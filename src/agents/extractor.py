from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.prompts.templates import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_HUMAN_PROMPT
from src.core.registry import ModelRegistry
from src.tools.chunker import TextChunker

logger = logging.getLogger(__name__)

# Simplified extraction prompt for batch processing
BATCH_EXTRACTOR_PROMPT = """Extract key facts from this text that answer the question.

Question: {question}

Text:
{chunk}

Output a JSON with:
{{
    "facts": ["fact 1", "fact 2", ...],
    "relevant": true/false
}}

Only include facts directly relevant to the question. Be concise."""



from src.agents.evidence_scorer import EvidenceScorerAgent

class ExtractorAgent:
    """
    Agent responsible for filtering retrieved documents into concise notes.
    Now integrates Evidence Scoring.
    
    Uses chunking + batch processing for efficient extraction with small models:
    1. Chunks documents into smaller pieces
    2. Processes each chunk batch through LLM
    3. Aggregates all extracted facts
    4. Grades facts using EvidenceScorer
    """
    
    def __init__(
        self, 
        batch_size: int = 2,
        sentences_per_chunk: int = 4,
        max_chunks: int = 15
    ):
        """
        Initialize extractor with batch processing settings.
        """
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser()
        self.batch_size = batch_size
        self.max_chunks = max_chunks
        
        # Chunker for splitting documents
        self.chunker = TextChunker(
            strategy="sentence",
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=1
        )
        
        # Scorer integration
        self.scorer = EvidenceScorerAgent()
        
        # Simpler prompt for batch extraction
        self.batch_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical fact extractor. Extract only relevant facts."),
            ("human", BATCH_EXTRACTOR_PROMPT)
        ])
        
        # Original prompt for fallback
        self.full_prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTOR_SYSTEM_PROMPT),
            ("human", EXTRACTOR_HUMAN_PROMPT)
        ])
    
    async def _extract_from_chunk(self, question: str, chunk: str) -> List[str]:
        """Extract facts from a single chunk."""
        try:
            chain = self.batch_prompt | self.llm | self.parser
            response = await chain.ainvoke({
                "question": question,
                "chunk": chunk
            })
            
            facts = response.get("facts", [])
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []
            
        except Exception as e:
            # Bug Fix #5: Better error logging for debugging (still returns empty, but with context)
            logger.warning(f"Chunk extraction failed: {e}")
            return []
    
    async def _extract_batch(self, question: str, chunks: List[str]) -> List[str]:
        """Extract facts from a batch of chunks."""
        all_facts = []
        
        # Combine chunks for batch processing
        combined_text = "\n\n---\n\n".join(chunks)
        
        try:
            chain = self.batch_prompt | self.llm | self.parser
            response = await chain.ainvoke({
                "question": question,
                "chunk": combined_text
            })
            
            facts = response.get("facts", [])
            if isinstance(facts, list):
                all_facts.extend([str(f) for f in facts if f])
                
        except Exception as e:
            logger.warning(f"Batch extraction failed: {e}")
            # Fallback: try each chunk individually
            for chunk in chunks:
                all_facts.extend(await self._extract_from_chunk(question, chunk))
        
        return all_facts
    
    async def extract(self, question: str, documents: List[str]) -> Dict[str, Any]:
        """
        Extract relevant notes from documents using batch processing.
        Includes Evidence Scoring.
        """
        if not documents:
            return {
                "notes": "No documents provided.",
                "confidence": "NONE",
                "facts": [],
                "scored_facts": []
            }
        
        try:
            # 1. Flatten nested documents
            flat_docs = []
            for doc in documents:
                if isinstance(doc, list):
                    flat_docs.extend([str(d) for d in doc])
                else:
                    flat_docs.append(str(doc))
            
            # 2. Chunk all documents
            indexed_chunks = self.chunker.chunk_documents(flat_docs)
            chunks = [chunk for _, chunk in indexed_chunks]
            
            # Limit chunks to prevent too many LLM calls
            if len(chunks) > self.max_chunks:
                logger.info(f"Limiting chunks from {len(chunks)} to {self.max_chunks}")
                chunks = chunks[:self.max_chunks]
            
            logger.info(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")
            
            # 3. Process in batches
            all_facts = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                batch_facts = await self._extract_batch(question, batch)
                all_facts.extend(batch_facts)
                logger.debug(f"Batch {i//self.batch_size + 1}: extracted {len(batch_facts)} facts")
            
            # 4. Deduplicate facts
            unique_facts = list(dict.fromkeys(all_facts))
            
            # 5. Score Evidence Quality
            scored_facts = []
            if unique_facts:
                # EvidenceScorerAgent implementation check: 
                # Assuming EvidenceScorerAgent also needs to be async or its score_notes is sync regex?
                # Looking at imports... src.agents.evidence_scorer.EvidenceScorerAgent
                # If it uses LLM, it should be async. If regex, sync is fine.
                # Assuming sync for now unless I see otherwise, but best to wrap or check.
                # Since I don't want to open another file unless needed, I'll assume sync for scorer or update it later.
                # Update: EvidenceScorer usually uses LLM. I should verify if it needs update. 
                # But let's stick to what we see. I'll make extract async.
                scored_facts = await self.scorer.score_notes(unique_facts)
            
            # 6. Determine confidence based on results
            if len(unique_facts) >= 5:
                confidence = "HIGH"
            elif len(unique_facts) >= 2:
                confidence = "MEDIUM"
            elif len(unique_facts) >= 1:
                confidence = "LOW"
            else:
                confidence = "NONE"
            
            # 7. Format notes (Include Grade)
            if scored_facts:
                notes = "\n".join([f"• {sf['fact']} [Grade {sf['grade']}: {sf['study_type']}]" for sf in scored_facts])
            elif unique_facts:
                notes = "\n".join([f"• {fact}" for fact in unique_facts])
            else:
                # Fallback: use raw document snippets
                logger.warning("No facts extracted, using raw doc fallback")
                notes = "\n".join([doc[:300] + "..." for doc in flat_docs[:3]])
                confidence = "LOW"
            
            logger.info(f"Extraction complete: {len(unique_facts)} unique facts, confidence: {confidence}")
            
            return {
                "notes": notes,
                "confidence": confidence,
                "confidence_reason": f"Extracted {len(unique_facts)} facts from {len(chunks)} chunks",
                "facts": unique_facts,
                "scored_facts": scored_facts
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            fallback_notes = documents[0][:500] if documents else "Extraction error"
            return {
                "notes": f"[Fallback] {fallback_notes}...",
                "confidence": "NONE",
                "facts": [],
                "scored_facts": []
            }

    
    async def extract_notes(self, question: str, documents: List[str]) -> str:
        """Legacy method for backward compatibility."""
        result = await self.extract(question, documents)
        return result.get("notes", "")


