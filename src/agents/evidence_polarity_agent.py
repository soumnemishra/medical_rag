import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from src.state.state import GraphState, EvidencePolarity
from src.prompts.templates import (
    EVIDENCE_POLARITY_SYSTEM_PROMPT,
    EVIDENCE_POLARITY_HUMAN_PROMPT,
)
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  EvidencePolarityAgent                                             #
# ------------------------------------------------------------------ #

class EvidencePolarityAgent:
    """
    Detects the directional polarity of retrieved evidence relative to
    the original question: support / refute / mixed / insufficient.

    This runs AFTER the executor completes all plan steps, reading the
    current execution's extracted notes from GraphState.

    IMPORTANT: reads from state["step_notes"] and state["step_docs_ids"]
    — the CURRENT query's retrieved evidence — NOT from state["past_exp"]
    which holds data from previous FAILED retry attempts.

    The polarity result is written to state["evidence_polarity"] and is
    consumed by:
      - RagAgent: adds conflict note when polarity is "refute"/"mixed"
      - SafetyCriticAgent: enforces disclaimer when polarity is "refute"
      - SummaryAgent: flags contradictory evidence for "mixed"

    Failure is NON-BLOCKING — a polarity failure returns "insufficient"
    and lets the pipeline continue rather than aborting the query.
    """

    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)

        # Fail loud at startup, not silently mid-query
        if self.llm is None:
            raise RuntimeError(
                "EvidencePolarityAgent: Flash LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=EvidencePolarity)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_POLARITY_SYSTEM_PROMPT),
            ("human",  EVIDENCE_POLARITY_HUMAN_PROMPT),
        ])

        # Build chain ONCE in __init__ — not reconstructed per analyze() call
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _format_evidence(self, state: GraphState) -> str:
        """
        Build a readable evidence string from the CURRENT execution's
        retrieved notes and document IDs.

        Why NOT past_exp:
            state["past_exp"] stores results from PREVIOUS FAILED query
            attempts — it is the planner's retry memory. On the first
            attempt it is always empty.

            The current execution's evidence lives in:
              state["step_notes"]     — ExtractorAgent output this query
              state["step_docs_ids"]  — PMIDs retrieved this query
              state["step_output"]    — RagAgent answers this query

        Returns a formatted string, or a clear "no evidence" message
        so the LLM doesn't hallucinate a polarity from nothing.
        """
        step_notes   = state.get("step_notes",    [])
        step_doc_ids = state.get("step_docs_ids", [])
        step_outputs = state.get("step_output",   [])

        if not step_notes and not step_outputs:
            return "No evidence retrieved in current execution."

        lines = []
        count = 1

        # Format extracted notes with their associated PMIDs
        for i, note in enumerate(step_notes):
            if not note or str(note).strip() == "":
                continue

            # Get corresponding doc IDs for this step
            doc_ids = step_doc_ids[i] if i < len(step_doc_ids) else []
            if isinstance(doc_ids, list) and doc_ids:
                ids_str = f" (PMIDs: {', '.join(str(d) for d in doc_ids)})"
            else:
                ids_str = ""

            note_text = str(note)[:1000]  # truncate very long notes
            lines.append(f"Evidence item {count}{ids_str}:\n{note_text}")
            count += 1

        # Also include step answers as supporting context
        for i, output in enumerate(step_outputs):
            if isinstance(output, dict) and not output.get("is_error", False):
                answer = output.get("answer", "")
                if answer and answer.strip():
                    lines.append(f"Step {i + 1} answer:\n{answer[:500]}")

        if not lines:
            return "No relevant evidence text found in current execution."

        return "\n\n".join(lines)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """
        Runs the chain with up to 2 retries on transient failures.
        Separated into its own method for independent testability.
        """
        return await self.chain.ainvoke(inputs)

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def analyze(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyse the current execution's retrieved evidence and classify
        its polarity relative to the original question.

        Reads from:
            state["original_question"]  — the query being answered
            state["step_notes"]         — current execution extracted notes
            state["step_docs_ids"]      — current execution PMIDs
            state["step_output"]        — current execution step answers

        Writes to GraphState:
            evidence_polarity: {
                "polarity":   "support" | "refute" | "mixed" | "insufficient",
                "confidence": float 0.0–1.0,
                "reasoning":  str — one-sentence explanation
            }

        Downstream consumers of evidence_polarity:
            - RagAgent QA_HUMAN_PROMPT: {evidence_polarity} placeholder
            - SafetyCriticAgent: adds disclaimer when polarity = "refute"
            - SummaryAgent: flags conflict when polarity = "mixed"
        """
        try:
            question       = state["original_question"]
            evidence_text  = self._format_evidence(state)

            logger.info(
                f"Analysing evidence polarity for: '{question[:80]}...' "
                f"({len(state.get('step_notes', []))} step notes)"
            )

            try:
                result = await self._invoke_chain({
                    "question": question,
                    "evidence": evidence_text,
                })
            except RetryError as e:
                raise ValueError(f"Polarity chain failed after 2 attempts: {e}") from e

            polarity   = result.get("polarity",   "insufficient")
            confidence = float(result.get("confidence", 0.0))
            reasoning  = result.get("reasoning",  "")

            # Validate polarity value — LLM sometimes returns unexpected strings
            valid_polarities = {"support", "refute", "mixed", "insufficient"}
            if polarity not in valid_polarities:
                logger.warning(
                    f"LLM returned unexpected polarity '{polarity}' — "
                    f"defaulting to 'insufficient'"
                )
                polarity = "insufficient"

            logger.info(
                f"Evidence polarity: {polarity} "
                f"(confidence={confidence:.2f}) — {reasoning}"
            )

            return {
                "evidence_polarity": {
                    "polarity":   polarity,
                    "confidence": confidence,
                    "reasoning":  reasoning,
                }
            }

        except Exception as e:
            logger.error(f"Evidence polarity analysis failed: {e}", exc_info=True)
            # NON-BLOCKING — return safe default, pipeline continues
            return {
                "evidence_polarity": {
                    "polarity":   "insufficient",
                    "confidence": 0.0,
                    "reasoning":  f"Analysis failed: {e}",
                }
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def evidence_polarity_node(state: GraphState) -> Dict[str, Any]:
    """
    Thin wrapper called by the LangGraph StateGraph.
    Polarity is computed AFTER executor completes all plan steps,
    so step_notes and step_docs_ids are fully populated.
    """
    try:
        agent = AgentRegistry.get_instance().evidence_polarity
        return await agent.analyze(state)
    except Exception as e:
        logger.error(f"evidence_polarity_node crashed: {e}", exc_info=True)
        # Non-blocking — pipeline must not abort because polarity failed
        return {
            "evidence_polarity": {
                "polarity":   "insufficient",
                "confidence": 0.0,
                "reasoning":  f"Node crash: {e}",
            }
        }