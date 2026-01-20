from typing import List, Tuple, Optional, Dict
from src.pubmed_client import PubMedClient, RetrievedDocument
from src.query_builder import PubMedQueryBuilder, PICOQuery
from src.core.registry import ModelRegistry
from src.tools.dense_retriever import DenseRetriever, MedCPTRetriever, reciprocal_rank_fusion
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Prompts
# =============================================================================

PICO_DECOMPOSITION_PROMPT = """You are a medical query analyst. Decompose the following query into PICO components for PubMed search.

PICO = Population, Intervention, Comparison, Outcome

OUTPUT FORMAT (JSON):
{{
    "population": ["disease or condition terms"],
    "intervention": ["treatment, diagnostic, or topic terms"],
    "comparison": ["comparator terms if any, else empty list"],
    "outcome": ["outcome terms if any, else empty list"],
    "modifiers": ["stage, severity, age, etc."]
}}

GUIDELINES:
- Extract medical/disease terms for population (e.g., "Hirschsprung's Disease", "NSCLC", "melanoma")
- Extract intervention/topic terms (e.g., "treatment", "diagnosis", "prognosis")
- Keep terms short and specific for PubMed
- If a term has synonyms, include them (e.g., "therapy", "treatment")
"""

PICO_HUMAN_PROMPT = """Query: {query}

Decompose this into PICO components."""


# =============================================================================
# Reranker (Cross-Encoder)
# =============================================================================

class CrossEncoderReranker:
    """
    Two-stage reranker using sentence-transformers CrossEncoder.
    
    Uses a cross-encoder to deeply compare query-document pairs and reorder
    documents by semantic relevance.
    """
    
    # Medical/scientific cross-encoder model
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace model name. Defaults to ms-marco-MiniLM.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # Lazy loading
        logger.info(f"CrossEncoderReranker initialized (model: {self.model_name})")
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"CrossEncoder model loaded: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Reranking disabled.")
                self._model = False  # Mark as unavailable
        return self._model
    
    def rerank(
        self, 
        query: str, 
        documents: List[RetrievedDocument], 
        top_k: int = 10
    ) -> List[RetrievedDocument]:
        """
        Rerank documents by query relevance.
        
        Args:
            query: The search query.
            documents: List of retrieved documents.
            top_k: Number of top documents to return.
        
        Returns:
            Reranked list of documents (most relevant first).
        """
        if not documents:
            return []
        
        model = self._load_model()
        if not model:
            logger.warning("Reranker unavailable, returning original order")
            return documents[:top_k]
        
        # Create query-document pairs for scoring
        pairs = []
        for doc in documents:
            # Combine title and abstract for scoring
            doc_text = f"{doc.title}. {doc.abstract[:500]}"
            pairs.append([query, doc_text])
        
        # Debug: Log what specific query and doc are being scored
        if pairs:
            logger.info(f"Reranker Debug - Query: '{query}'")
            logger.info(f"Reranker Debug - First Doc: '{pairs[0][1][:100]}...'")
        
        # Score all pairs
        scores = model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Log reranking info
        logger.info(f"Reranked {len(documents)} docs. Top score: {scored_docs[0][1]:.3f}")
        
        return [doc for doc, score in scored_docs[:top_k]]


# =============================================================================
# Enhanced Retriever with Two-Stage Retrieval
# =============================================================================

class RetrieverTool:
    """
    Enhanced Retriever with Two-Stage Retrieval + Reranking.
    
    Pipeline:
        Query → LLM (PICO) → PubMed (Top 50) → CrossEncoder Rerank → Top 10
    """
    
    def __init__(
        self, 
        top_k: int = 10, 
        initial_pool_size: int = 50,
        use_query_builder: bool = True,
        use_reranker: bool = True,
        use_hybrid: bool = True,
        use_medcpt: bool = False,
        use_dynamic_k: bool = True
    ):
        """
        Initialize retriever with two-stage retrieval.
        
        Args:
            top_k: Final number of documents to return.
            initial_pool_size: Documents to fetch before reranking.
            use_query_builder: Use LLM + PICO decomposition.
            use_reranker: Use cross-encoder reranking.
            use_hybrid: Use hybrid retrieval (BM25 + Dense with RRF).
            use_medcpt: Use MedCPT for medical-specific retrieval (requires GPU).
            use_dynamic_k: Use dynamic top_k based on query complexity.
        """
        self.top_k = top_k
        self.initial_pool_size = initial_pool_size
        self.use_query_builder = use_query_builder
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        self.use_medcpt = use_medcpt
        self.use_dynamic_k = use_dynamic_k
        
        # PubMed client fetches larger pool for reranking
        self.client = PubMedClient(max_results=initial_pool_size)
        
        # PICO query builder
        if use_query_builder:
            self.query_builder = PubMedQueryBuilder(use_mesh=True, use_tiab=True)
            self.llm = ModelRegistry.get_llm(temperature=0.0, json_mode=True)
            self.parser = JsonOutputParser()
            self.pico_prompt = ChatPromptTemplate.from_messages([
                ("system", PICO_DECOMPOSITION_PROMPT),
                ("human", PICO_HUMAN_PROMPT)
            ])
        else:
            self.query_builder = None
        
        # Cross-encoder reranker
        if use_reranker:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        # Dense retriever for hybrid mode (MedCPT or standard)
        if use_hybrid:
            if use_medcpt:
                self.dense_retriever = MedCPTRetriever(use_cross_encoder=True)
                logger.info("Using MedCPT for biomedical retrieval")
            else:
                self.dense_retriever = DenseRetriever()
        else:
            self.dense_retriever = None
        
        logger.info(
            f"RetrieverTool initialized: pool={initial_pool_size}, top_k={top_k}, "
            f"query_builder={use_query_builder}, reranker={use_reranker}, hybrid={use_hybrid}, medcpt={use_medcpt}, dynamic_k={use_dynamic_k}"
        )
    
    def _decompose_to_pico(self, query: str) -> PICOQuery:
        """Use LLM to decompose natural language query into PICO format."""
        try:
            chain = self.pico_prompt | self.llm | self.parser
            result = chain.invoke({"query": query})
            
            pico = PICOQuery(
                population=result.get("population", []),
                intervention=result.get("intervention", []),
                comparison=result.get("comparison", []),
                outcome=result.get("outcome", []),
                modifiers=result.get("modifiers", []),
                humans_only=True
            )
            
            logger.info(f"PICO decomposition: P={pico.population}, I={pico.intervention}")
            return pico
            
        except Exception as e:
            logger.warning(f"PICO decomposition failed: {e}. Falling back to simple query.")
            return PICOQuery(population=[query], humans_only=True)
    
    def _analyze_query_complexity(self, query: str) -> str:
        """
        Analyze query complexity to determine optimal document count.
        
        Returns:
            'specific': Very targeted query (e.g., "EGFR T790M mutation resistance")
            'moderate': Standard medical query (e.g., "lung cancer treatment")
            'broad': General/exploratory query (e.g., "cancer research trends")
        """
        # Simple heuristic-based analysis (avoid extra LLM call)
        query_lower = query.lower()
        words = query_lower.split()
        
        # Indicators of specific queries
        specific_indicators = [
            # Specific mutations, genes, drugs
            any(term in query_lower for term in ['mutation', 'variant', 'polymorphism', 'snp']),
            # Specific drug names (often contain numbers or hyphens)
            any('-' in word or any(c.isdigit() for c in word) for word in words),
            # Mechanism of action terms
            any(term in query_lower for term in ['pathway', 'receptor', 'inhibitor', 'antagonist']),
            # Specific disease subtypes
            any(term in query_lower for term in ['stage iv', 'stage iii', 'metastatic', 'refractory']),
        ]
        
        # Indicators of broad queries
        broad_indicators = [
            # General overview terms
            any(term in query_lower for term in ['overview', 'general', 'introduction', 'what is']),
            # Multi-topic queries
            query_lower.count(' and ') >= 2 or query_lower.count(' or ') >= 2,
            # Very short queries (likely vague)
            len(words) <= 3,
            # Exploratory terms
            any(term in query_lower for term in ['trends', 'future', 'research directions', 'review']),
        ]
        
        specific_score = sum(specific_indicators)
        broad_score = sum(broad_indicators)
        
        if specific_score >= 2:
            return 'specific'
        elif broad_score >= 2:
            return 'broad'
        else:
            return 'moderate'
    
    def _get_dynamic_k(self, query: str) -> int:
        """
        Get dynamic top_k based on query complexity.
        
        Returns:
            Adjusted top_k value (3-20 range).
        """
        complexity = self._analyze_query_complexity(query)
        
        # Map complexity to top_k range
        k_map = {
            'specific': 5,    # Targeted: fewer, high-precision docs
            'moderate': 10,   # Standard: balanced
            'broad': 15       # Broad: more docs for comprehensive coverage
        }
        
        dynamic_k = k_map.get(complexity, self.top_k)
        logger.info(f"Dynamic K selection: complexity={complexity}, top_k={dynamic_k}")
        
        return dynamic_k

    
    def _build_optimized_query(self, query: str) -> str:
        """Build a PubMed-optimized query from natural language."""
        pico = self._decompose_to_pico(query)
        optimized_query = self.query_builder.build_query(pico)
        logger.info(f"Optimized PubMed query: {optimized_query[:100]}...")
        return optimized_query
    
    def _hybrid_rank(
        self, 
        query: str, 
        docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Hybrid ranking using Reciprocal Rank Fusion (RRF).
        
        Combines:
        - BM25 ranking (original PubMed order)
        - Dense semantic similarity ranking
        - Cross-encoder reranking (optional, applied after fusion)
        """
        # Create doc lookup
        doc_map: Dict[str, RetrievedDocument] = {doc.pmid: doc for doc in docs}
        
        # List 1: BM25 ranking (original PubMed order)
        bm25_ranked = [(doc.pmid, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        
        # List 2: Dense semantic ranking
        doc_texts = [f"{doc.title}. {doc.abstract[:500]}" for doc in docs]
        doc_ids = [doc.pmid for doc in docs]
        dense_results = self.dense_retriever.rank_documents(query, doc_texts, doc_ids)
        dense_ranked = [(pmid, score) for pmid, _, score in dense_results]
        
        # Apply RRF fusion
        fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked], k=60)
        
        # Reorder documents by fused score
        fused_docs = []
        for pmid, _ in fused:
            if pmid in doc_map:
                fused_docs.append(doc_map[pmid])
        
        logger.info(f"Hybrid RRF: Combined BM25 + Dense rankings")
        
        # Optional: Apply cross-encoder reranking on top of fused results
        if self.use_reranker and self.reranker and len(fused_docs) > self.top_k:
            fused_docs = self.reranker.rerank(query, fused_docs, top_k=self.top_k)
        else:
            fused_docs = fused_docs[:self.top_k]
        
        return fused_docs
    
    def __call__(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Two-stage retrieval: Fetch large pool → Rerank → Return top_k.
        
        Args:
            query: Natural language query.
        
        Returns:
            Tuple[List[str], List[str]]: (List of context strings, List of PMIDs)
        """
        try:
            # Determine effective top_k (dynamic or static)
            effective_top_k = self._get_dynamic_k(query) if self.use_dynamic_k else self.top_k
            
            # Stage 1: Build query and fetch large pool
            if self.use_query_builder and self.query_builder:
                search_query = self._build_optimized_query(query)
            else:
                search_query = query.strip()
            
            # Fetch initial pool
            docs = self.client.search(search_query, max_results=self.initial_pool_size)
            
            if not docs:
                logger.warning(f"No documents found for query: {query[:50]}...")
                if self.use_query_builder:
                    logger.info("Trying fallback with raw query...")
                    docs = self.client.search(query.strip(), max_results=self.initial_pool_size)
            
            if not docs:
                return [], []
            
            logger.info(f"Stage 1: Retrieved {len(docs)} documents from PubMed")
            
            # Stage 2: Hybrid ranking with RRF (if enabled)
            if self.use_hybrid and self.dense_retriever and len(docs) > 1:
                docs = self._hybrid_rank(query, docs)
                docs = docs[:effective_top_k]  # Apply dynamic K
                logger.info(f"Stage 2: Hybrid ranked to top {len(docs)} documents")
            # Stage 2b: Rerank (if hybrid not used)
            elif self.use_reranker and self.reranker and len(docs) > effective_top_k:
                docs = self.reranker.rerank(query, docs, top_k=effective_top_k)
                logger.info(f"Stage 2: Reranked to top {len(docs)} documents")
            else:
                docs = docs[:effective_top_k]
            
            # Format output
            list_docs = []
            list_doc_ids = []
            
            for doc in docs:
                context = f"Title: {doc.title}\nAbstract: {doc.abstract}\nDate: {doc.year}"
                list_docs.append(context)
                list_doc_ids.append(doc.pmid)
            
            logger.info(f"Final: Returning {len(list_docs)} documents")
            return list_docs, list_doc_ids
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return [], []
