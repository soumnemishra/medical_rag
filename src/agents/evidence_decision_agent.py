import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from src.state.state import GraphState
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Output schema                                                      #
# ------------------------------------------------------------------ #

class EvidenceDecisionOutput(BaseModel):
    """
    Pydantic schema for the LLM decision output.
    Enforces the three valid decision values — any other string throws
    immediately rather than silently routing to the wrong branch.
    """
    decision: str = Field(
        description="accept | reretrieve_diverse | reretrieve_counter"
    )
    reasoning: str = Field(
        description="Brief reasoning for the decision"
    )


# ------------------------------------------------------------------ #
#  Prompt — lives here until moved to templates.py                   #
# ------------------------------------------------------------------ #
# TODO: move to templates.py and import, so PROMPT_VERSION tracks it.

EVIDENCE_DECISION_SYSTEM_PROMPT = """You are a Quality Control Agent for a medical QA system.
Your goal is to decide whether the current evidence is sufficient to answer the question or if more searching is needed.

INPUTS:
1. Question
2. Evidence Polarity (support/refute/mixed/insufficient) and Confidence (0.0-1.0)
3. Retry Count (how many reretrieve attempts have already been made)

DECISION RULES:
1. ACCEPT:
   - IF Polarity is 'support' or 'refute' AND Confidence >= 0.7.
   - IF Retry Count >= 1 (max retries reached — always accept to avoid infinite loop).
2. RERETRIEVE_COUNTER:
   - IF Polarity is 'mixed' (clear conflicting evidence between sources).
3. RERETRIEVE_DIVERSE:
   - IF Polarity is 'insufficient' OR Confidence < 0.7.

OUTPUT FORMAT (JSON):
{{
  "decision": "accept" | "reretrieve_diverse" | "reretrieve_counter",
  "reasoning": "One sentence explanation"
}}
"""

EVIDENCE_DECISION_HUMAN_PROMPT = """Question: {question}
Polarity: {polarity} (Confidence: {confidence})
Retry Count: {retry_count}"""


# ------------------------------------------------------------------ #
#  EvidenceDecisionAgent                                             #
# ------------------------------------------------------------------ #

class EvidenceDecisionAgent:
    """
    Quality gate that decides whether the current retrieved evidence is
    good enough to proceed, or whether retrieval should be retried with
    a different strategy.

    Three possible decisions:
      accept             — evidence is sufficient, proceed to synthesis
      reretrieve_diverse — evidence is thin/insufficient, search broader
      reretrieve_counter — evidence is conflicting, search for counter-evidence

    Key design decisions:
      - Max 1 retry enforced HERE (not delegated to retrieval node)
      - PMID overlap check uses CURRENT execution's step_docs_ids,
        not past_exp (which is retry memory, often empty on first attempt)
      - retry_count incremented in the return dict — guaranteed, not hoped for
    """

    def __init__(self):
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)

        if self.llm is None:
            raise RuntimeError(
                "EvidenceDecisionAgent: Flash LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=EvidenceDecisionOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_DECISION_SYSTEM_PROMPT),
            ("human",  EVIDENCE_DECISION_HUMAN_PROMPT),
        ])

        # Build chain ONCE — not reconstructed per decide() call
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _calculate_overlap(self, state: GraphState) -> float:
        """
        Calculate PMID overlap across steps in the CURRENT execution.

        Why NOT past_exp:
            past_exp stores results from previous FAILED retry attempts.
            On the first query attempt it is always empty — overlap would
            always be 0.0 and diverse retrieval would never trigger.

            The current execution's retrieved PMIDs live in:
                state["step_docs_ids"] — List[List[str]], one per step

        What we're detecting:
            If Step 1 and Step 2 retrieved the same 8 papers, the plan
            decomposition isn't generating diverse enough sub-questions.
            Overlap > 60% → trigger diverse re-retrieval.

        Returns float 0.0–1.0 (fraction of last step's PMIDs seen before)
        """
        step_docs = state.get("step_docs_ids", [])

        if len(step_docs) < 2:
            return 0.0

        # Current step = last entry in step_docs_ids
        last = step_docs[-1]
        current_ids = set(
            str(d) for d in (last if isinstance(last, list) else [last])
        )

        if not current_ids:
            return 0.0

        # Previous steps = everything before the last
        previous_ids: set = set()
        for step in step_docs[:-1]:
            for doc_id in (step if isinstance(step, list) else [step]):
                previous_ids.add(str(doc_id))

        if not previous_ids:
            return 0.0

        overlap = len(current_ids & previous_ids) / len(current_ids)
        logger.debug(f"PMID overlap: {overlap:.2f} ({len(current_ids & previous_ids)}/{len(current_ids)})")
        return overlap

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """
        Runs the decision chain with up to 2 retries.
        Isolated for independent testability.
        """
        return await self.chain.ainvoke(inputs)

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def decide(self, state: GraphState) -> Dict[str, Any]:
        """
        Evaluate current evidence quality and decide next action.

        Reads from GraphState:
            original_question    — the query being answered
            evidence_polarity    — set by EvidencePolarityAgent
            step_docs_ids        — current execution's retrieved PMIDs
            retry_count          — how many reretrieve cycles done so far

        Returns GraphState update:
            evidence_decision    — "accept" | "reretrieve_diverse" | "reretrieve_counter"
            retry_count          — incremented when reretrieve is triggered
                                   (guaranteed here, not delegated to retrieval node)
        """
        try:
            retry_count = state.get("retry_count", 0)

            # HARD CONSTRAINT: max 1 retry to prevent infinite retrieval loop.
            # This guard lives HERE so it's always enforced regardless of
            # what the LLM or retrieval node does.
            if retry_count >= 1:
                logger.info(f"Max retries reached ({retry_count}) — forcing accept.")
                return {"evidence_decision": "accept"}

            evidence_polarity = state.get("evidence_polarity", {})
            polarity   = evidence_polarity.get("polarity",   "insufficient") \
                         if isinstance(evidence_polarity, dict) else "insufficient"
            confidence = float(evidence_polarity.get("confidence", 0.0)) \
                         if isinstance(evidence_polarity, dict) else 0.0

            # CODE-BASED RULE: high PMID overlap → diverse retrieval
            # This runs before the LLM call so it's fast and deterministic.
            overlap = self._calculate_overlap(state)
            if overlap > 0.6:
                logger.info(
                    f"High PMID redundancy ({overlap:.2f}) across steps — "
                    f"triggering diverse retrieval."
                )
                return {
                    "evidence_decision": "reretrieve_diverse",
                    "retry_count":       retry_count + 1,  # guaranteed increment
                }

            # LLM-BASED DECISION for everything else
            try:
                result = await self._invoke_chain({
                    "question":    state["original_question"],
                    "polarity":    polarity,
                    "confidence":  confidence,
                    "retry_count": retry_count,
                })
            except RetryError as e:
                raise ValueError(f"Decision chain failed after 2 attempts: {e}") from e

            decision = result.get("decision", "accept")

            # Validate decision value — LLM may return unexpected strings
            valid = {"accept", "reretrieve_diverse", "reretrieve_counter"}
            if decision not in valid:
                logger.warning(f"LLM returned invalid decision '{decision}' — defaulting to accept")
                decision = "accept"

            logger.info(
                f"Evidence decision: {decision} | "
                f"polarity={polarity} confidence={confidence:.2f} | "
                f"reason: {result.get('reasoning', '')}"
            )

            # Only increment retry_count when we're actually retrying
            if decision != "accept":
                return {
                    "evidence_decision": decision,
                    "retry_count":       retry_count + 1,
                }

            return {"evidence_decision": decision}

        except Exception as e:
            logger.error(f"Evidence decision failed: {e}", exc_info=True)
            # Safe default — accept so pipeline doesn't stall
            return {"evidence_decision": "accept"}


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def evidence_decision_node(state: GraphState) -> Dict[str, Any]:
    """
    Thin wrapper called by LangGraph StateGraph.
    Uses registry singleton — no model reload per call.
    """
    try:
        agent = AgentRegistry.get_instance().evidence_decision
        return await agent.decide(state)
    except Exception as e:
        logger.error(f"evidence_decision_node crashed: {e}", exc_info=True)
        return {"evidence_decision": "accept"}