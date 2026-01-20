"""
Text Chunking Utilities for MA-RAG Pipeline.

Provides sentence-based and section-based chunking for PubMed abstracts
to enable efficient processing by small language models.
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunk text into smaller pieces for efficient LLM processing.
    
    Strategies:
    - sentence: Split by sentences with overlap
    - section: Split by abstract sections (Background/Methods/Results/Conclusion)
    - fixed: Fixed character count chunks
    """
    
    def __init__(
        self, 
        strategy: str = "sentence",
        sentences_per_chunk: int = 3,
        overlap_sentences: int = 1,
        max_chunk_chars: int = 500
    ):
        """
        Initialize chunker.
        
        Args:
            strategy: "sentence", "section", or "fixed"
            sentences_per_chunk: Number of sentences per chunk (for sentence strategy)
            overlap_sentences: Overlap between chunks (for sentence strategy)
            max_chunk_chars: Maximum characters per chunk (for fixed strategy)
        """
        self.strategy = strategy
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.max_chunk_chars = max_chunk_chars
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting pattern
        # Handles: . ! ? followed by space or end
        # Avoids splitting on: Dr. Mr. Mrs. etc.
        abbreviations = r'(?<!\b[A-Z][a-z]?)(?<!\b[Dd]r)(?<!\b[Mm]r)(?<!\b[Mm]rs)(?<!\b[Mm]s)(?<!\be\.g)(?<!\bi\.e)'
        pattern = abbreviations + r'[.!?]\s+'
        
        sentences = re.split(pattern, text)
        # Filter empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text using sliding window over sentences."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.sentences_per_chunk:
            return [text] if text.strip() else []
        
        chunks = []
        step = self.sentences_per_chunk - self.overlap_sentences
        step = max(1, step)  # Ensure at least 1 sentence step
        
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            if chunk_sentences:
                chunk = '. '.join(chunk_sentences)
                if not chunk.endswith('.'):
                    chunk += '.'
                chunks.append(chunk)
            
            # Stop if we've covered all sentences
            if i + self.sentences_per_chunk >= len(sentences):
                break
        
        return chunks
    
    def _chunk_by_section(self, text: str) -> List[str]:
        """
        Chunk by detecting abstract sections.
        Common patterns: BACKGROUND:, METHODS:, RESULTS:, CONCLUSION:
        """
        # Section header patterns
        section_patterns = [
            r'\b(BACKGROUND|Background|INTRODUCTION|Introduction):?\s*',
            r'\b(METHODS?|Methods?|MATERIALS?\s+AND\s+METHODS?):?\s*',
            r'\b(RESULTS?|Results?|FINDINGS?|Findings?):?\s*',
            r'\b(CONCLUSIONS?|Conclusions?|DISCUSSION|Discussion):?\s*',
        ]
        
        # Try to find sections
        combined_pattern = '|'.join(section_patterns)
        parts = re.split(f'({combined_pattern})', text, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            # No sections found, fall back to sentence chunking
            return self._chunk_by_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if part and not re.match(combined_pattern, part, re.IGNORECASE):
                if part.strip():
                    current_chunk += part.strip() + " "
            elif current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk by fixed character count with word boundary awareness."""
        if len(text) <= self.max_chunk_chars:
            return [text] if text.strip() else []
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > self.max_chunk_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        if self.strategy == "sentence":
            return self._chunk_by_sentences(text)
        elif self.strategy == "section":
            return self._chunk_by_section(text)
        elif self.strategy == "fixed":
            return self._chunk_by_fixed_size(text)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using sentence")
            return self._chunk_by_sentences(text)
    
    def chunk_documents(self, documents: List[str]) -> List[Tuple[int, str]]:
        """
        Chunk multiple documents, preserving document index.
        
        Args:
            documents: List of document texts.
            
        Returns:
            List of (doc_index, chunk_text) tuples.
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk(doc)
            for chunk in chunks:
                all_chunks.append((doc_idx, chunk))
        
        logger.info(f"Chunked {len(documents)} docs into {len(all_chunks)} chunks")
        return all_chunks
    
    def batch_chunks(
        self, 
        chunks: List[Tuple[int, str]], 
        batch_size: int = 3
    ) -> List[List[Tuple[int, str]]]:
        """
        Group chunks into batches for processing.
        
        Args:
            chunks: List of (doc_index, chunk_text) tuples.
            batch_size: Number of chunks per batch.
            
        Returns:
            List of batches, each batch is a list of tuples.
        """
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of {batch_size} chunks each")
        return batches


# Convenience function
def chunk_abstracts(
    abstracts: List[str], 
    sentences_per_chunk: int = 4,
    overlap: int = 1
) -> List[Tuple[int, str]]:
    """
    Convenience function to chunk PubMed abstracts.
    
    Args:
        abstracts: List of abstract texts.
        sentences_per_chunk: Sentences per chunk.
        overlap: Sentence overlap between chunks.
        
    Returns:
        List of (abstract_index, chunk_text) tuples.
    """
    chunker = TextChunker(
        strategy="sentence",
        sentences_per_chunk=sentences_per_chunk,
        overlap_sentences=overlap
    )
    return chunker.chunk_documents(abstracts)
