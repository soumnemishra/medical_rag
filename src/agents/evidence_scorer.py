import asyncio
import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.prompts.templates import EVIDENCE_SCORER_SYSTEM_PROMPT, EVIDENCE_SCORER_HUMAN_PROMPT
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)

BATCH_SIZE = 5  # facts per LLM call — prevents model context overflow


# ------------------------------------------------------------------ #
#  Output schema                                                      #
# ------------------------------------------------------------------ #

class ScoredNote(BaseModel):
    """Schema for a single scored fact."""
    fact:       str   = Field(description="Original fact text")
    study_type: str   = Field(default="Unspecified", description="RCT/Cohort/Meta-analysis/etc")
    grade:      str   = Field(default="C",           description="A, B, or C")
    confidence: float = Field(default=0.0,           description="0.0–1.0 confidence in the grade")

class ScoredNotesOutput(BaseModel):
    """
    Pydantic schema enforcing the LLM output structure.

    Why: bare JsonOutputParser() silently returns [] when the LLM uses
    'notes' or 'scored_facts' instead of 'scored_notes'. Every fact
    then defaults to Grade C with no visible error. This schema makes
    wrong key names throw immediately with a clear validation message.
    """
    scored_notes: List[ScoredNote] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  EvidenceScorerAgent                                                #
# ------------------------------------------------------------------ #

class EvidenceScorerAgent:
    """
    Grades the quality of extracted medical facts using a study-type
    classification scheme (A/B/C evidence hierarchy).

    Grade A — Systematic reviews, meta-analyses, large RCTs, guidelines
    Grade B — Small RCTs, cohort studies, case-control studies
    Grade C — Case reports, expert opinion, animal/in-vitro studies

    Processing:
      - Facts are batched (BATCH_SIZE=5) to prevent model context overflow
      - Batches run in PARALLEL via asyncio.gather() — not sequentially
      - Failed batches fall back to Grade C rather than crashing
    """

    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)

        # Fail loud at startup — not silently on first scoring call
        if self.llm is None:
            raise RuntimeError(
                "EvidenceScorerAgent: Flash LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=ScoredNotesOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_SCORER_SYSTEM_PROMPT),
            ("human",  EVIDENCE_SCORER_HUMAN_PROMPT),
        ])

        # Build chain ONCE — not reconstructed on every _score_batch call
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    async def _score_batch(self, notes: List[str]) -> List[Dict[str, Any]]:
        """
        Score one batch of facts via a single LLM call.

        On failure: returns Grade C defaults for every fact in the batch
        so the pipeline continues rather than crashing. The warning log
        makes the failure visible for debugging.
        """
        if not notes:
            return []

        notes_str = "\n".join([f"- {note}" for note in notes])

        try:
            result       = await self.chain.ainvoke({"notes": notes_str})
            scored_notes = result.get("scored_notes", [])

            return [
                {
                    "fact":       item.get("fact",       "Unknown Fact"),
                    "study_type": item.get("study_type", "Unspecified"),
                    "grade":      item.get("grade",      "C"),
                    "confidence": float(item.get("confidence", 0.0)),
                }
                for item in scored_notes
            ]

        except Exception as e:
            logger.warning(
                f"Batch scoring failed (falling back to Grade C for {len(notes)} facts): {e}"
            )
            # Grade C fallback — conservative default when scoring fails
            return [
                {"fact": note, "grade": "C", "study_type": "Unknown", "confidence": 0.0}
                for note in notes
            ]

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def score_notes(self, notes: List[str]) -> List[Dict[str, Any]]:
        """
        Score a list of extracted facts in parallel batches.

        Why parallel: scoring batches are completely independent of each
        other. Sequential execution (the original design) wasted 3–4s on
        20 facts. asyncio.gather() runs all batches simultaneously.

        Args:
            notes: List of fact strings from ExtractorAgent.

        Returns:
            List of dicts: {fact, study_type, grade, confidence}
            Preserves input order — batch i maps to output slice i.
        """
        if not notes:
            return []

        # Split into batches
        batches = [
            notes[i: i + BATCH_SIZE]
            for i in range(0, len(notes), BATCH_SIZE)
        ]

        total = len(batches)
        logger.info(
            f"Scoring {len(notes)} facts in {total} parallel batch(es) "
            f"of up to {BATCH_SIZE}"
        )

        # Fire all batches simultaneously
        batch_results = await asyncio.gather(
            *[self._score_batch(batch) for batch in batches],
            return_exceptions=True,
        )

        all_scored: List[Dict[str, Any]] = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                # gather with return_exceptions=True — one batch failing
                # should not kill the others
                logger.warning(f"Batch {i + 1}/{total} raised unhandled: {result}")
                all_scored.extend([
                    {"fact": note, "grade": "C", "study_type": "Unknown", "confidence": 0.0}
                    for note in batches[i]
                ])
            else:
                all_scored.extend(result)

        grade_summary = {
            "A": sum(1 for s in all_scored if s["grade"] == "A"),
            "B": sum(1 for s in all_scored if s["grade"] == "B"),
            "C": sum(1 for s in all_scored if s["grade"] == "C"),
        }
        logger.info(
            f"Scoring complete: {len(all_scored)} facts graded — "
            f"A={grade_summary['A']} B={grade_summary['B']} C={grade_summary['C']}"
        )

        return all_scored