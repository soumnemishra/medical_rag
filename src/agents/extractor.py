import asyncio
import logging
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.prompts.templates import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_HUMAN_PROMPT,
    BATCH_EXTRACTOR_SYSTEM_PROMPT,
    BATCH_EXTRACTOR_HUMAN_PROMPT,
)
from src.core.registry import ModelRegistry
from src.tools.chunker import TextChunker
from src.agents.evidence_scorer import EvidenceScorerAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Output schema — enforces structure on LLM chunk extraction output  #
# ------------------------------------------------------------------ #

class ChunkExtractionOutput(BaseModel):
    """
    Pydantic schema for per-chunk LLM extraction.

    Why: bare JsonOutputParser() silently returns {} when the LLM uses
    the wrong key name (e.g. "extracted" instead of "facts").
    This schema makes that failure loud and immediately visible in logs.
    """
    facts:    List[str] = Field(
        default_factory=list,
        description="List of relevant medical facts extracted from the chunk"
    )
    relevant: bool = Field(
        default=False,
        description="Whether this chunk contained information relevant to the question"
    )


# ------------------------------------------------------------------ #
#  ExtractorAgent                                                     #
# ------------------------------------------------------------------ #

class ExtractorAgent:
    """
    Filters retrieved PubMed documents into graded, deduplicated fact notes.

    Pipeline per extract() call:
      1. Flatten nested document lists
      2. Chunk documents into sentence-boundary pieces
      3. Select chunks distributed across ALL documents (not front-loaded)
      4. Extract facts in parallel — batch_size controls concurrency, not
         string concatenation (the old design silently exceeded model context)
      5. Semantic deduplication — removes near-duplicate facts that exact
         string matching misses
      6. Grade each fact A/B/C via EvidenceScorerAgent
      7. Compute confidence from BEST grade present, not raw fact count
      8. Return structured output with extraction_meta for planner memory
    """

    def __init__(
        self,
        batch_size: int = 2,
        sentences_per_chunk: int = 4,
        max_chunks: int = 15,
        dedup_threshold: float = 0.92,
    ):
        """
        Args:
            batch_size:          Max concurrent LLM calls during extraction.
                                 Controls parallelism — NOT string concatenation.
            sentences_per_chunk: How many sentences per chunk. 4 works well
                                 for small models (3B) without truncation.
            max_chunks:          Total chunks to process. Spread across all
                                 documents, not front-loaded from document 1.
            dedup_threshold:     Cosine similarity above which two facts are
                                 considered duplicates. 0.92 = 92% similar.
        """
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        # Fail loud at startup — not silently on first query
        if self.llm is None:
            raise RuntimeError(
                "ExtractorAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.batch_size      = batch_size
        self.max_chunks      = max_chunks
        self.dedup_threshold = dedup_threshold

        # Chunker — sentence-based with overlap so facts spanning
        # a chunk boundary are not lost
        self.chunker = TextChunker(
            strategy="sentence",
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=1,
        )

        self.scorer = EvidenceScorerAgent()

        # Pydantic-validated parser — wrong key names now throw visibly
        self.parser = JsonOutputParser(pydantic_object=ChunkExtractionOutput)

        # Build chains ONCE in __init__ — not reconstructed per call
        self.batch_prompt = ChatPromptTemplate.from_messages([
            ("system", BATCH_EXTRACTOR_SYSTEM_PROMPT),
            ("human",  BATCH_EXTRACTOR_HUMAN_PROMPT),
        ])
        self.full_prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTOR_SYSTEM_PROMPT),
            ("human",  EXTRACTOR_HUMAN_PROMPT),
        ])

        self.batch_chain = self.batch_prompt | self.llm | self.parser
        self.full_chain  = self.full_prompt  | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    async def _extract_from_chunk(self, question: str, chunk: str) -> List[str]:
        """
        Extract facts from a single chunk via one LLM call.

        Each chunk is processed independently so the model receives
        at most sentences_per_chunk sentences — well within the context
        window of a 3B model.

        Returns empty list on failure — the caller tracks failed chunks
        via exceptions from asyncio.gather(return_exceptions=True).
        """
        try:
            response = await self.batch_chain.ainvoke({
                "question": question,
                "chunk":    chunk,
            })
            facts = response.get("facts", [])
            return [str(f) for f in facts if f] if isinstance(facts, list) else []
        except Exception as e:
            logger.warning(f"Chunk extraction failed: {e}")
            raise  # re-raise so gather can count failures

    def _select_chunks(
        self,
        indexed_chunks: List[Tuple[int, str]],
        max_chunks: int,
    ) -> List[str]:
        """
        Select up to max_chunks distributed evenly across all source documents.

        Why not [:max_chunks]:
            If document 1 produces 20 chunks and max_chunks=15, naive slicing
            gives you 15 chunks from document 1 and ZERO from documents 2–8.
            The retriever ranked those documents by relevance — ignoring them
            defeats the entire retrieval stage.

        Strategy: round-robin across documents, taking up to max_per_doc
        chunks from each, until max_chunks is reached.

        Args:
            indexed_chunks: List of (doc_index, chunk_text) from TextChunker
            max_chunks:     Hard upper limit on total chunks returned
        """
        by_doc: Dict[int, List[str]] = defaultdict(list)
        for doc_idx, chunk in indexed_chunks:
            by_doc[doc_idx].append(chunk)

        n_docs      = len(by_doc)
        max_per_doc = max(1, max_chunks // n_docs)

        selected = []
        for doc_idx in sorted(by_doc.keys()):
            selected.extend(by_doc[doc_idx][:max_per_doc])
            if len(selected) >= max_chunks:
                break

        return selected[:max_chunks]

    def _deduplicate_facts(self, facts: List[str]) -> List[str]:
        """
        Semantic deduplication using cosine similarity on embeddings.

        Why not dict.fromkeys():
            Exact string dedup misses near-duplicates like:
              "Metformin reduces HbA1c by 1.5%"
              "HbA1c reduction of ~1.5% observed with metformin"
            Both survive exact dedup. Both contain the same clinical fact.

        Algorithm:
            Embed all facts → keep a fact only if its cosine similarity
            to ALL already-kept facts is below dedup_threshold.

        Falls back to exact dedup if sentence-transformers unavailable.

        Args:
            facts: Raw list of extracted facts (may contain duplicates)
        """
        if len(facts) <= 1:
            return facts

        try:
            from sentence_transformers import SentenceTransformer, util
            import torch

            # Reuse the same small model already in use by RetrieverTool
            # Do NOT load a new model here — expensive and redundant
            model      = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(facts, convert_to_tensor=True)

            kept_indices = [0]
            for i in range(1, len(facts)):
                kept_embeds = embeddings[kept_indices]
                sims        = util.cos_sim(embeddings[i], kept_embeds)
                if sims.max().item() < self.dedup_threshold:
                    kept_indices.append(i)

            logger.info(
                f"Dedup: {len(facts)} raw facts → {len(kept_indices)} unique "
                f"(threshold={self.dedup_threshold})"
            )
            return [facts[i] for i in kept_indices]

        except ImportError:
            logger.warning(
                "sentence-transformers not available for semantic dedup — "
                "falling back to exact string dedup"
            )
            return list(dict.fromkeys(facts))

    def _compute_confidence(
        self, scored_facts: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Derive confidence from the BEST evidence grade present, not fact count.

        Why fact count is wrong:
            5 case reports (Grade C) → count-based = "HIGH" confidence
            1 meta-analysis (Grade A) → count-based = "LOW" confidence
            Clinically this is backwards.

        Grade hierarchy: A > B > C
            A = Systematic reviews, meta-analyses, large RCTs, guidelines
            B = Small RCTs, cohort studies, case-control
            C = Case reports, expert opinion, animal/in-vitro studies

        Returns:
            (confidence_level, human_readable_reason)
        """
        if not scored_facts:
            return "NONE", "No facts extracted from retrieved documents"

        grades = [sf.get("grade", "C") for sf in scored_facts]

        if "A" in grades:
            count  = grades.count("A")
            return "HIGH", f"Contains {count} Grade-A fact(s) from systematic review/RCT evidence"
        elif "B" in grades:
            count  = grades.count("B")
            return "MEDIUM", f"Contains {count} Grade-B fact(s) from cohort/small RCT evidence"
        else:
            count  = len(grades)
            return "LOW", f"Only {count} Grade-C fact(s) from case reports or expert opinion"

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def extract(
        self,
        question:  str,
        documents: List[str],
    ) -> Dict[str, Any]:
        """
        Main extraction pipeline. Called by execution_node in executor.py.

        Args:
            question:  The step question being answered (used to filter relevant facts)
            documents: List of retrieved document strings (abstracts or full-text)

        Returns dict with:
            notes            — formatted bullet points with grade tags for RAG agent
            confidence       — HIGH | MEDIUM | LOW | NONE (grade-driven)
            confidence_reason — human-readable explanation of confidence level
            facts            — deduplicated plain fact strings
            scored_facts     — facts with grade, study_type, confidence_score
            extraction_meta  — chunk counts for PlannerAgent._format_memory()
            is_error         — True only on complete pipeline failure
        """
        if not documents:
            return {
                "notes":      "No documents provided.",
                "confidence": "NONE",
                "facts":      [],
                "scored_facts": [],
            }

        try:
            # ── 1. Flatten nested document lists ──────────────────────
            flat_docs: List[str] = []
            for doc in documents:
                if isinstance(doc, list):
                    flat_docs.extend([str(d) for d in doc])
                else:
                    flat_docs.append(str(doc))

            # ── 2. Chunk with sentence boundaries + 1-sentence overlap ─
            indexed_chunks = self.chunker.chunk_documents(flat_docs)

            # ── 3. Distribute chunks across all docs (not front-loaded) ─
            chunks = self._select_chunks(indexed_chunks, self.max_chunks)
            logger.info(
                f"Extraction: {len(flat_docs)} docs → "
                f"{len(indexed_chunks)} total chunks → "
                f"{len(chunks)} selected for processing"
            )

            # ── 4. Parallel extraction with concurrency limit ──────────
            #
            # batch_size controls MAX CONCURRENT LLM calls, not string
            # concatenation. Each chunk gets its own independent LLM call
            # so the model never receives more than sentences_per_chunk
            # sentences at once.
            #
            # asyncio.Semaphore(batch_size) ensures at most batch_size
            # calls run simultaneously — prevents overwhelming the model
            # server while still being much faster than sequential.
            sem = asyncio.Semaphore(self.batch_size)

            async def _bounded_extract(chunk: str) -> List[str]:
                async with sem:
                    return await self._extract_from_chunk(question, chunk)

            raw_results = await asyncio.gather(
                *[_bounded_extract(c) for c in chunks],
                return_exceptions=True,
            )

            all_facts: List[str] = []
            failed_chunks = 0
            for result in raw_results:
                if isinstance(result, Exception):
                    failed_chunks += 1
                    logger.warning(f"Chunk failed: {result}")
                elif isinstance(result, list):
                    all_facts.extend(result)

            logger.info(
                f"Raw extraction: {len(all_facts)} facts from "
                f"{len(chunks) - failed_chunks}/{len(chunks)} chunks succeeded"
            )

            # ── 5. Semantic deduplication ──────────────────────────────
            unique_facts = self._deduplicate_facts(all_facts)

            # ── 6. Grade each fact A/B/C ───────────────────────────────
            scored_facts: List[Dict] = []
            if unique_facts:
                scored_facts = await self.scorer.score_notes(unique_facts)

            # ── 7. Confidence from BEST grade, not fact count ──────────
            confidence, confidence_reason = self._compute_confidence(scored_facts)

            # ── 8. Format notes with grade tags for RAG agent ─────────
            if scored_facts:
                notes = "\n".join([
                    f"• {sf['fact']} [Grade {sf['grade']}: {sf['study_type']}]"
                    for sf in scored_facts
                ])
            elif unique_facts:
                # Scorer failed but we have facts — still useful
                notes = "\n".join([f"• {f}" for f in unique_facts])
            else:
                # Nothing extracted — raw doc fallback so RAG agent
                # has SOMETHING rather than an empty context
                logger.warning(
                    "No facts extracted from any chunk — using raw doc fallback. "
                    "Consider broadening the retrieval query."
                )
                notes      = "\n".join([doc[:300] + "..." for doc in flat_docs[:3]])
                confidence = "NONE"
                confidence_reason = "Extraction produced zero facts; raw snippets used as fallback"

            logger.info(
                f"Extraction complete | facts={len(unique_facts)} | "
                f"confidence={confidence} | failed_chunks={failed_chunks}"
            )

            return {
                "notes":             notes,
                "confidence":        confidence,
                "confidence_reason": confidence_reason,
                "facts":             unique_facts,
                "scored_facts":      scored_facts,
                # extraction_meta feeds PlannerAgent._format_memory()
                # so the planner can learn "step 2 only got 1 fact from
                # 15 chunks — broaden the query next time"
                "extraction_meta": {
                    "docs_processed":   len(flat_docs),
                    "chunks_selected":  len(chunks),
                    "chunks_failed":    failed_chunks,
                    "facts_raw":        len(all_facts),
                    "facts_unique":     len(unique_facts),
                },
            }

        except Exception as e:
            logger.error(f"Extraction pipeline failed: {e}", exc_info=True)
            fallback_notes = documents[0][:500] if documents else "Extraction error"
            return {
                "notes":         f"[Fallback] {fallback_notes}...",
                "confidence":    "NONE",
                "facts":         [],
                "scored_facts":  [],
                "is_error":      True,    # consistent with executor error tagging
                "error_message": str(e),
            }

    async def extract_notes(self, question: str, documents: List[str]) -> str:
        """
        Legacy compatibility wrapper.
        Callers that used the old string-return API continue to work.
        New callers should use extract() to get the full structured output.
        """
        result = await self.extract(question, documents)
        return result.get("notes", "")