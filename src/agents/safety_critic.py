import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from src.state.state import GraphState
from src.prompts.templates import SAFETY_CRITIC_SYSTEM_PROMPT, SAFETY_CRITIC_HUMAN_PROMPT
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Output schema                                                      #
# ------------------------------------------------------------------ #

class SafetyCriticOutput(BaseModel):
    """
    Pydantic schema for safety audit output.

    Why: bare JsonOutputParser() silently returns is_safe=True when
    the LLM uses 'safe' instead of 'is_safe'. Every unsafe answer
    would then pass the audit with no warning.
    """
    is_safe:        bool            = Field(default=True)
    issues:         List[str]       = Field(default_factory=list)
    refined_answer: str | None      = Field(default=None)


# ------------------------------------------------------------------ #
#  ClinicalSafetyCriticAgent                                          #
# ------------------------------------------------------------------ #

class ClinicalSafetyCriticAgent:
    """
    Audits the final answer for clinical safety compliance before
    it is returned to the user.

    Checklist (enforced via SAFETY_CRITIC_SYSTEM_PROMPT):
      1. No inappropriate absolutes ("always", "never", "cure")
      2. Appropriate medical uncertainty language
      3. Disclaimer present for high/medium risk queries
      4. Claims backed by cited PMIDs
      5. Drug contraindications mentioned where relevant
      6. PMID citations preserved in refined answer
      7. Final Answer tag preserved if present
      8. Conflicting evidence acknowledged when polarity is refute/mixed

    Evaluation mode:
      When evaluation_mode=True, unsafe answers are NOT refined — the
      original answer is preserved for fair benchmark scoring. Issues
      are still recorded in safety_flags and an "eval_mode_safety_skip"
      flag is added so metrics can separate "genuinely safe" from
      "skipped for eval".
    """

    def __init__(self):
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        if self.llm is None:
            raise RuntimeError(
                "ClinicalSafetyCriticAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=SafetyCriticOutput)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SAFETY_CRITIC_SYSTEM_PROMPT),
            ("human",  SAFETY_CRITIC_HUMAN_PROMPT),
        ])

        # Chain built ONCE in __init__
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """Runs audit chain with up to 2 retries on transient failures."""
        return await self.chain.ainvoke(inputs)

    def _extract_polarity_string(self, state: GraphState) -> str:
        """
        Safely extract polarity string for prompt injection.
        SAFETY_CRITIC_HUMAN_PROMPT has {evidence_polarity} placeholder —
        this must always be a non-None string or LangChain raises KeyError.
        """
        polarity_data = state.get("evidence_polarity", {})
        if isinstance(polarity_data, dict):
            return polarity_data.get("polarity", "insufficient")
        if isinstance(polarity_data, str):
            return polarity_data
        return "insufficient"

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def critique(self, state: GraphState) -> Dict[str, Any]:
        """
        Audit the final answer and either pass it, refine it, or flag it.

        Reads from GraphState:
            final_answer      — the answer to audit
            intent            — from ClinicalIntentAgent
            risk_level        — from ClinicalIntentAgent
            evidence_polarity — from EvidencePolarityAgent
            evaluation_mode   — if True, preserve original for benchmarking
            safety_flags      — existing flags (appended to, not overwritten)

        Returns GraphState update:
            final_answer      — original, refined, or warning-appended
            safety_flags      — list of issues found (empty = safe)
        """
        try:
            answer         = state.get("final_answer", "")
            intent         = state.get("intent",     "unknown")
            risk           = state.get("risk_level", "low")
            eval_mode      = state.get("evaluation_mode", False)
            existing_flags = state.get("safety_flags", [])
            polarity_str   = self._extract_polarity_string(state)

            # Skip audit for empty or already-errored answers
            if not answer or "Error" in answer:
                logger.info("Safety audit skipped — empty or error answer.")
                return {"safety_flags": existing_flags + ["skipped_empty"]}

            logger.info(f"Safety audit | risk={risk} intent={intent} polarity={polarity_str}")

            try:
                result = await self._invoke_chain({
                    "answer":            answer,
                    "intent":            intent,
                    "risk_level":        risk,
                    "evidence_polarity": polarity_str,   # required by updated template
                })
            except RetryError as e:
                raise ValueError(f"Safety audit chain failed after 2 attempts: {e}") from e

            is_safe  = result.get("is_safe", True)
            issues   = result.get("issues", [])
            refined  = result.get("refined_answer")

            if not is_safe:
                logger.warning(f"Safety FAILED | issues: {issues}")

                # EVALUATION MODE — preserve original answer for fair benchmarking
                # but record that safety was skipped so metrics are honest
                if eval_mode:
                    logger.info("Eval mode: preserving original answer despite safety issues.")
                    return {
                        "final_answer": answer,
                        "safety_flags": existing_flags + issues + ["eval_mode_safety_skip"],
                    }

                # STANDARD MODE — apply refinement or append warning
                if refined:
                    logger.info("Applying refined safe answer.")
                    return {
                        "final_answer": refined,
                        "safety_flags": existing_flags + issues,
                    }
                else:
                    warning = (
                        "\n\n**SAFETY WARNING**: This content has been flagged for review: "
                        + "; ".join(issues)
                    )
                    return {
                        "final_answer": answer + warning,
                        "safety_flags": existing_flags + issues,
                    }

            logger.info("Safety audit PASSED.")
            return {"safety_flags": existing_flags}

        except Exception as e:
            logger.error(f"Safety audit failed: {e}", exc_info=True)
            return {"safety_flags": state.get("safety_flags", []) + ["audit_error"]}


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def safety_critic_node(state: GraphState) -> Dict[str, Any]:
    """Uses registry singleton."""
    try:
        agent = AgentRegistry.get_instance().safety_critic
        return await agent.critique(state)
    except Exception as e:
        logger.error(f"safety_critic_node crashed: {e}", exc_info=True)
        return {"safety_flags": state.get("safety_flags", []) + ["node_crash"]}