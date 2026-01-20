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


class ExtractorAgent:
    """
    Agent responsible for filtering retrieved documents into concise notes.
    
    Uses chunking + batch processing for efficient extraction with small models:
    1. Chunks documents into smaller pieces
    2. Processes each chunk batch through LLM
    3. Aggregates all extracted facts
    """
    
    def __init__(
        self, 
        batch_size: int = 2,
        sentences_per_chunk: int = 4,
        max_chunks: int = 15
    ):
        """
        Initialize extractor with batch processing settings.
        
        Args:
            batch_size: Number of chunks to process per LLM call.
            sentences_per_chunk: Sentences per chunk when splitting.
            max_chunks: Maximum chunks to process (prevents excessive LLM calls).
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
    
    def _extract_from_chunk(self, question: str, chunk: str) -> List[str]:
        """Extract facts from a single chunk."""
        try:
            chain = self.batch_prompt | self.llm | self.parser
            response = chain.invoke({
                "question": question,
                "chunk": chunk
            })
            
            facts = response.get("facts", [])
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
            return []
            
        except Exception as e:
            logger.warning(f"Chunk extraction failed: {e}")
            return []
    
    def _extract_batch(self, question: str, chunks: List[str]) -> List[str]:
        """Extract facts from a batch of chunks."""
        all_facts = []
        
        # Combine chunks for batch processing
        combined_text = "\n\n---\n\n".join(chunks)
        
        try:
            chain = self.batch_prompt | self.llm | self.parser
            response = chain.invoke({
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
                all_facts.extend(self._extract_from_chunk(question, chunk))
        
        return all_facts
    
    def extract(self, question: str, documents: List[str]) -> Dict[str, Any]:
        """
        Extract relevant notes from documents using batch processing.
        
        Pipeline:
        1. Flatten and chunk all documents
        2. Process chunks in batches
        3. Aggregate and deduplicate facts
        
        Args:
            question: The query.
            documents: List of document strings.
            
        Returns:
            Dict with notes, confidence, and facts.
        """
        if not documents:
            return {
                "notes": "No documents provided.",
                "confidence": "NONE",
                "facts": []
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
                batch_facts = self._extract_batch(question, batch)
                all_facts.extend(batch_facts)
                logger.debug(f"Batch {i//self.batch_size + 1}: extracted {len(batch_facts)} facts")
            
            # 4. Deduplicate facts (simple string comparison)
            unique_facts = list(dict.fromkeys(all_facts))
            
            # 5. Determine confidence based on results
            if len(unique_facts) >= 5:
                confidence = "HIGH"
            elif len(unique_facts) >= 2:
                confidence = "MEDIUM"
            elif len(unique_facts) >= 1:
                confidence = "LOW"
            else:
                confidence = "NONE"
            
            # 6. Format notes
            if unique_facts:
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
                "facts": unique_facts
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Fallback: return first document as-is
            fallback_notes = documents[0][:500] if documents else "Extraction error"
            return {
                "notes": f"[Fallback] {fallback_notes}...",
                "confidence": "NONE",
                "facts": []
            }
    
    def extract_notes(self, question: str, documents: List[str]) -> str:
        """Legacy method for backward compatibility."""
        result = self.extract(question, documents)
        return result.get("notes", "")


