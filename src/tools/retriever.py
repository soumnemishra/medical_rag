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
    "population": ["disease or condition terms - include synonyms and MeSH terms"],
    "intervention": ["treatment, diagnostic, topic, or mechanism terms"],
    "comparison": ["comparator terms if any, else empty list"],
    "outcome": ["outcome terms if any, else empty list"],
    "modifiers": ["stage, severity, age, clinical setting, etc."]
}}

GUIDELINES:
- Extract medical/disease terms for population (e.g., "Hirschsprung's Disease", "NSCLC", "melanoma")
- For NEW ONSET conditions, include: ["new onset X", "initial presentation X", "first episode X", "acute X"]
- For FOLLOW-UP/MONITORING queries, include: ["follow-up", "monitoring", "surveillance", "screening", "assessment"]
- For MANAGEMENT queries, include: ["management", "treatment", "therapy", "guidelines", "recommendations"]
- Include SYNONYMS: e.g., for "heart disease" include ["heart disease", "cardiac disease", "cardiovascular disease", "coronary artery disease", "heart failure"]
- For OCULAR conditions: include ["eye", "ocular", "ophthalmic", specific anatomical terms]
- For HYPERTENSION: include ["hypertensive emergency", "hypertensive crisis", "severe hypertension", "blood pressure management"]
- Extract intervention/topic terms (e.g., "treatment", "diagnosis", "prognosis", "pathophysiology", "etiology")
- Always include at least 3-5 terms per relevant category for comprehensive coverage
"""

PICO_HUMAN_PROMPT = """Query: {query}

Decompose this into PICO components. Be thorough - include synonyms, MeSH terms, and related terms for comprehensive PubMed coverage. For edge cases (follow-up, monitoring, new onset), include specific clinical terminology."""


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
        
        # Log reranking info with score threshold warning
        top_score = scored_docs[0][1]
        logger.info(f"Reranked {len(documents)} docs. Top score: {top_score:.3f}")
        
        # Warn if top score is very low (indicates weak relevance)
        LOW_RELEVANCE_THRESHOLD = -3.0
        if top_score < LOW_RELEVANCE_THRESHOLD:
            logger.warning(f"⚠️ LOW RELEVANCE WARNING: Top score {top_score:.3f} is below threshold {LOW_RELEVANCE_THRESHOLD}. Query may need better terms.")
        
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
        initial_pool_size: int = 75,  # Increased from 50 for better coverage
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
            
            logger.info(f"PICO decomposition for '{query[:50]}': P={pico.population}, I={pico.intervention}, M={pico.modifiers}, O={pico.outcome}")
            return pico
            
        except Exception as e:
            logger.warning(f"PICO decomposition failed: {e}. Falling back to simple query.")
            # Simple keyword extraction as fallback population
            keywords = [w for w in query.split() if len(w) > 3][:5]
            return PICOQuery(population=keywords if keywords else [query], humans_only=True)
    
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
            Adjusted top_k value (8-20 range).
        """
        complexity = self._analyze_query_complexity(query)
        
        # Map complexity to top_k range - INCREASED for more comprehensive answers
        k_map = {
            'specific': 8,    # Targeted: balanced precision with coverage
            'moderate': 12,   # Standard: good coverage
            'broad': 20       # Broad: maximum docs for comprehensive coverage
        }
        
        dynamic_k = k_map.get(complexity, self.top_k)
        logger.info(f"Dynamic K selection: complexity={complexity}, top_k={dynamic_k}")
        
        return dynamic_k
            
    def _extract_keywords(self, query: str) -> str:
        """Extract keywords from natural language for broad search."""
        # Remove common question words
        stop_words = {'what', 'is', 'the', 'are', 'latest', 'treatment', 'options', 'for', 'recommended', 'should', 'be', 'management', 'choice', 'of', 'in', 'with', 'patient', 'patients', 'current', 'how', 'long', 'term', 'rates'}
        words = query.lower().replace('?', '').replace(',', '').split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(keywords) if keywords else query

    
    def _build_optimized_query(self, query: str) -> str:
        """Build a PubMed-optimized query from natural language."""
        pico = self._decompose_to_pico(query)
        optimized_query = self.query_builder.build_query(pico)
        logger.info(f"Optimized PubMed query constructed: {optimized_query}")
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
    
    async def __call__(
        self, 
        query: str, 
        requires_guidelines: bool = False,
        sources: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Two-stage retrieval: Fetch large pool → Rerank → Return top_k.
        Supports filtering for guidelines and specific sources.
        """
        try:
            # Determine effective top_k (dynamic or static)
            effective_top_k = self._get_dynamic_k(query) if self.use_dynamic_k else self.top_k
            
            # LATENCY OPTIMIZATION: Skip PICO for very simple queries
            is_simple = len(query.split()) < 5 and not any(term in query.lower() for term in ["treatment", "management", "diagnosis", "dosage", "efficacy"])
            
            # Stage 1: Build base query
            if self.use_query_builder and self.query_builder and not is_simple:
                base_query = self._build_optimized_query(query)
            else:
                if is_simple:
                    logger.info(f"Fast-Path Retrieval: Skipping PICO for simple query: '{query}'")
                base_query = query.strip()
                
            docs = []
            
            # Clinical Guideline prioritization (Hard Rule: Guidelines > Papers)
            if requires_guidelines:
                guideline_filter = " AND (Practice Guideline[pt] OR Guideline[pt] OR Consensus Development Conference[pt])"
                guideline_query = base_query + guideline_filter
                
                logger.info("Fetching Clinical Guidelines...")
                guideline_docs = await self.client.search(guideline_query, max_results=10) # Fetch up to 10 guidelines
                if guideline_docs:
                    logger.info(f"Found {len(guideline_docs)} guidelines")
                    docs.extend(guideline_docs)
            
            # General Search (balance redundancy if guidelines found)
            remaining_slots = self.initial_pool_size - len(docs)
            if remaining_slots > 0:
                # Add rigorous evidence filter if high risk/sources specified
                filter_str = ""
                if sources and "rct" in sources:
                    filter_str += " AND (Randomized Controlled Trial[pt] OR Meta-Analysis[pt])"
                
                general_query = base_query + filter_str
                general_docs = await self.client.search(general_query, max_results=remaining_slots)
                docs.extend(general_docs)

            # Deduplicate by PMID
            unique_docs = []
            seen_pmids = set()
            for doc in docs:
                if doc.pmid not in seen_pmids:
                    unique_docs.append(doc)
                    seen_pmids.add(doc.pmid)
            docs = unique_docs
            
            if not docs:
                logger.warning(f"No documents found for optimized query. Trying Broad Strategy...")
                # Strategy 2: Keywords only (no field tags, no ANDing of modifiers)
                keywords = self._extract_keywords(query)
                logger.info(f"Broad Search Query: {keywords}")
                docs = await self.client.search(keywords, max_results=self.initial_pool_size)
                
            if not docs and self.use_query_builder:
                logger.warning("Still no docs. Trying simplest PICO (Population only)...")
                # Strategy 3: Population only
                pico = self._decompose_to_pico(query)
                if pico.population:
                    simple_pico_query = " OR ".join([f'"{p}"[tiab]' for p in pico.population[:3]])
                    logger.info(f"Population-only Query: {simple_pico_query}")
                    docs = await self.client.search(simple_pico_query, max_results=self.initial_pool_size)

            if not docs:
                logger.warning(f"Still no docs. Trying Emergency Fallback (AND-joined terms)...")
                # Strategy 4: AND-joined core terms (Last Resort)
                words = self._extract_keywords(query).split()
                if len(words) > 1:
                    emergency_query = " AND ".join([f'"{w}"[tiab]' for w in words[:4]])
                    logger.info(f"Emergency Query: {emergency_query}")
                    docs = await self.client.search(emergency_query, max_results=self.initial_pool_size)

            if not docs:
                return [], []
            
            logger.info(f"Stage 1: Retrieved {len(docs)} documents total")
            
            # Stage 2: Ranking
            # Note: We want to preserve guideline priority if possible, but reranker might reshuffle
            # Strategy: Rerank everything, but maybe boost guidelines? 
            # For now, let the semantic reranker decide relevance, assuming guidelines are relevant.
            
            if self.use_hybrid and self.dense_retriever and len(docs) > 1:
                docs = self._hybrid_rank(query, docs)
                docs = docs[:effective_top_k]
                logger.info(f"Stage 2: Hybrid ranked to top {len(docs)} documents")
            elif self.use_reranker and self.reranker and len(docs) > effective_top_k:
                docs = self.reranker.rerank(query, docs, top_k=effective_top_k)
                logger.info(f"Stage 2: Reranked to top {len(docs)} documents")
            else:
                docs = docs[:effective_top_k]
            
            # Format output
            list_docs = []
            list_doc_ids = []
            
            for doc in docs:
                # Add type indicator to context
                doc_type = " [GUIDELINE]" if "Guideline" in (doc.publication_types or []) else ""
                context = f"Title: {doc.title}{doc_type}\nAbstract: {doc.abstract}\nDate: {doc.year}"
                list_docs.append(context)
                list_doc_ids.append(doc.pmid)
            
            logger.info(f"Final: Returning {len(list_docs)} documents")
            return list_docs, list_doc_ids
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return [], []
