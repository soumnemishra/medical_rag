

#Router agents says how should we answer this question it talks about the execution strategy 
import logging
from typing import Dict, Any

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

class AnswerPolicy(BaseModel):
    format:           str  = Field(default="standard", description="standard | yes_no")
    force_commitment: bool = Field(default=False)
    allow_disclaimer: bool = Field(default=True)

class RouterDecision(BaseModel):
    """
    Pydantic schema for router LLM output.
    RouterOutput was imported but never wired — fixed here.
    Wrong key name now throws immediately instead of routing
    everything to multihop silently.
    """
    execution_mode:            str          = Field(description="direct_qa | disambiguation | multihop")
    requires_planning:         bool         = Field(default=True)
    requires_extraction:       bool         = Field(default=True)
    requires_evidence_grading: bool         = Field(default=True)
    answer_policy:             AnswerPolicy = Field(default_factory=AnswerPolicy)


# ------------------------------------------------------------------ #
#  Prompts
#  NOTE: Dynamic per-query values MUST be in the HUMAN prompt.
#  System prompt is static — template vars there cause KeyError.
#  ROUTER_SYSTEM_PROMPT and ROUTER_HUMAN_PROMPT should be moved to
#  templates.py in the next templates update.
# ------------------------------------------------------------------ #

ROUTER_SYSTEM_PROMPT = """You are the Master Router for a medical RAG system.
Your job is to route queries to the most efficient execution path.

MODES:
1. "direct_qa" — Simple, unambiguous, single-hop questions.
   Examples: "What is diabetes?", "Side effects of aspirin?"
   Criteria: Definitional, general knowledge, no comparison needed.

2. "disambiguation" — Specific but potentially ambiguous, needs precise retrieval.
   Examples: "Effect of Drug X on Gene Y?", "Does study Z support this?"
   Criteria: Single-hop but high noise risk; specific literature required.

3. "multihop" — Complex, multi-step reasoning across documents.
   Examples: "Compare X and Y treatments", "Best management for patient with A and B?"
   Criteria: Comparative, mechanistic chains, patient-specific management.

OUTPUT FORMAT (JSON):
{{
    "execution_mode": "direct_qa" | "disambiguation" | "multihop",
    "requires_planning": true,
    "requires_extraction": true,
    "requires_evidence_grading": true,
    "answer_policy": {{
        "format": "standard" | "yes_no",
        "force_commitment": false,
        "allow_disclaimer": true
    }}
}}
"""

# Dynamic values (intent, risk_level, question) are in the HUMAN prompt
# so LangChain fills them at invocation time, not chain-build time.
ROUTER_HUMAN_PROMPT = """Query: {question}
Intent: {intent}
Risk Level: {risk_level}"""


# ------------------------------------------------------------------ #
#  RouterAgent                                                        #
# ------------------------------------------------------------------ #

class RouterAgent:
    """
    Routes incoming queries to the correct execution pipeline.

    Three modes:
      direct_qa      — fast path, single RAG call, no planning
      disambiguation — single RAG call with careful retrieval
      multihop       — full plan → executor → aggregate pipeline

    Falls back to multihop on any failure — the safest default for
    a medical system is to do MORE reasoning, not less.
    """

    def __init__(self):
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        if self.llm is None:
            raise RuntimeError(
                "RouterAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        self.parser = JsonOutputParser(pydantic_object=RouterDecision)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human",  ROUTER_HUMAN_PROMPT),
        ])

        # Chain built ONCE in __init__
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        return await self.chain.ainvoke(inputs)

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def route(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyse query complexity and return routing decision.

        Reads:  original_question, intent, risk_level
        Writes: router_output (RouterOutput TypedDict in GraphState)
        """
        question   = state["original_question"]
        intent     = state.get("intent",     "informational")
        risk_level = state.get("risk_level", "low")

        logger.info(
            f"Routing: '{question[:60]}...' "
            f"[intent={intent} risk={risk_level}]"
        )

        try:
            try:
                result = await self._invoke_chain({
                    "question":   question,
                    "intent":     intent,
                    "risk_level": risk_level,
                })
            except RetryError as e:
                raise ValueError(f"Router chain failed after 2 attempts: {e}") from e

            mode = result.get("execution_mode", "multihop")
            valid_modes = {"direct_qa", "disambiguation", "multihop"}
            if mode not in valid_modes:
                logger.warning(f"Invalid mode '{mode}' — defaulting to multihop")
                mode = "multihop"

            policy = result.get("answer_policy", {})
            if isinstance(policy, AnswerPolicy):
                policy = policy.dict()

            router_output = {
                "execution_mode":            mode,
                "requires_planning":         result.get("requires_planning",         mode == "multihop"),
                "requires_extraction":       result.get("requires_extraction",       mode != "direct_qa"),
                "requires_evidence_grading": result.get("requires_evidence_grading", mode != "direct_qa"),
                "answer_policy": {
                    "format":           policy.get("format",           "standard"),
                    "force_commitment": policy.get("force_commitment", False),
                    "allow_disclaimer": policy.get("allow_disclaimer", True),
                },
                "execution_budget": None,
            }

            logger.info(f"Router decision: {mode}")
            return {"router_output": router_output}

        except Exception as e:
            logger.error(f"Router failed: {e} — falling back to multihop", exc_info=True)
            return {
                "router_output": {
                    "execution_mode":            "multihop",
                    "requires_planning":         True,
                    "requires_extraction":       True,
                    "requires_evidence_grading": True,
                    "answer_policy": {
                        "format": "standard", "force_commitment": False, "allow_disclaimer": True
                    },
                    "execution_budget": None,
                }
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def router_node(state: GraphState) -> Dict[str, Any]:
    """Uses registry singleton — no LLM reload per call."""
    try:
        agent = AgentRegistry.get_instance().router
        return await agent.route(state)
    except Exception as e:
        logger.error(f"router_node crashed: {e}", exc_info=True)
        return {
            "router_output": {
                "execution_mode": "multihop",
                "requires_planning": True, "requires_extraction": True,
                "requires_evidence_grading": True,
                "answer_policy": {"format": "standard", "force_commitment": False, "allow_disclaimer": True},
                "execution_budget": None,
            }
        }