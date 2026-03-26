import logging
from typing import Dict, Any, List, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state.state import PlanExecState
from src.prompts.templates import (
    STEP_DEFINER_SYSTEM_PROMPT,
    STEP_DEFINER_HUMAN_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_HUMAN_PROMPT,
)
from src.core.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Output schemas                                                     #
# ------------------------------------------------------------------ #

class StepTaskOutput(BaseModel):
    type: str = Field(default="question-answering", description="question-answering | aggregate")
    task: str = Field(description="Detailed query string for this step")

class PlanSummaryOutput(BaseModel):
    output:         str         = Field(default="Successful")
    answer:         str         = Field(default="")
    final_decision: str | None  = Field(default=None)
    score:          int         = Field(default=0)


# ------------------------------------------------------------------ #
#  StepDefinerAgent                                                   #
# ------------------------------------------------------------------ #

class StepDefinerAgent:
    """
    Decides what to do at each point in plan execution:
      - If steps remain  → define the next step question (_next_step)
      - If all done      → summarise all outputs into final answer (_summarize)

    Critical fix: plan is now List[Dict] (from updated PlannerAgent),
    not List[str]. All string operations on plan items now go through
    _get_step_question() which handles both formats gracefully.

    LLM split:
      step_chain    → light LLM (fast routing decision, not clinical reasoning)
      summary_chain → HEAVY LLM (final clinical answer with citations)
    """

    def __init__(self):
        light_llm = ModelRegistry.get_light_llm(temperature=0.0, json_mode=True)
        heavy_llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        if heavy_llm is None:
            raise RuntimeError(
                "StepDefinerAgent: Heavy LLM unavailable. "
                "Summary chain requires the heavy model."
            )

        # Step definition — routing decision — light model is fine
        self.step_parser  = JsonOutputParser(pydantic_object=StepTaskOutput)
        self.step_prompt  = ChatPromptTemplate.from_messages([
            ("system", STEP_DEFINER_SYSTEM_PROMPT),
            ("human",  STEP_DEFINER_HUMAN_PROMPT),
        ])
        self.step_chain   = self.step_prompt | light_llm | self.step_parser

        # Summarisation — final clinical answer with citations — HEAVY model
        # Using light model here was the wrong tradeoff: the summary IS the
        # output the user sees. It needs the best reasoning available.
        self.summary_parser = JsonOutputParser(pydantic_object=PlanSummaryOutput)
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human",  SUMMARY_HUMAN_PROMPT),
        ])
        self.summary_chain  = self.summary_prompt | heavy_llm | self.summary_parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _get_step_question(step: Union[Dict, str]) -> str:
        """
        Safely extract the question string from a plan step.

        Updated PlannerAgent returns steps as dicts:
            {"id": 1, "question": "...", "depends_on": [], "step_type": "..."}

        Old format was plain strings. This method handles both so the
        system is backward-compatible during the migration.
        """
        if isinstance(step, dict):
            return step.get("question", str(step))
        return str(step)

    @staticmethod
    def _flatten_doc_ids(raw: Any) -> List[str]:
        """Flatten step_docs_ids entry regardless of nesting shape."""
        if isinstance(raw, list):
            result = []
            for item in raw:
                if isinstance(item, list):
                    result.extend([str(i) for i in item])
                else:
                    result.append(str(item))
            return result
        return [str(raw)] if raw else []

    # ---------------------------------------------------------------- #
    #  Core routing logic                                                #
    # ---------------------------------------------------------------- #

    async def define_task(self, state: PlanExecState) -> Dict[str, Any]:
        """
        Entry point called by task_definer_node in the executor graph.
        Decides: more steps to run → _next_step, else → _summarize.
        """
        try:
            plan          = state.get("plan", [])
            step_output   = state.get("step_output", [])
            step_question = state.get("step_question", [])
            step_docs_ids = state.get("step_docs_ids", [])
            original_q    = state.get("original_question", "")

            # Stop when we have an output for every plan step
            should_stop = len(step_output) >= len(plan)

            if should_stop:
                return await self._summarize(
                    original_q, plan, step_output, step_question, step_docs_ids
                )
            else:
                return await self._next_step(plan, step_output)

        except Exception as e:
            logger.error(f"define_task failed: {e}", exc_info=True)
            return {
                "stop": True,
                "plan_summary": {
                    "output": "Failed",
                    "answer": f"Task definition error: {e}",
                    "score":  0,
                },
            }

    async def _next_step(
        self,
        plan:         List[Union[Dict, str]],
        step_outputs: List[Dict],
    ) -> Dict[str, Any]:
        """Define the next step question from the plan."""
        current_idx = len(step_outputs)

        if not plan or current_idx >= len(plan):
            logger.error(
                f"Invalid plan state: plan_len={len(plan)} step_idx={current_idx}"
            )
            return {
                "stop": True,
                "plan_summary": {
                    "output": "Failed",
                    "answer": "Plan execution error: invalid step index",
                    "score":  0,
                },
            }

        # Works with both List[str] and List[Dict] plan formats
        cur_step    = plan[current_idx]
        cur_step_q  = self._get_step_question(cur_step)
        plan_str    = ", ".join([self._get_step_question(s) for s in plan])

        logger.info(f"Defining step {current_idx + 1}/{len(plan)}: '{cur_step_q}'")

        # Build memory string from completed step outputs
        memory = ""
        for i, output in enumerate(step_outputs):
            step_q = self._get_step_question(plan[i]) if i < len(plan) else "Unknown"
            memory += f"Task: {step_q}\nAnswer: {output.get('answer', '')}\n\n"

        try:
            result = await self.step_chain.ainvoke({
                "plan":     f"[{plan_str}]",
                "cur_step": cur_step_q,
                "memory":   memory,
            })

            # Carry over step_type from planner if available
            if isinstance(cur_step, dict):
                result["step_type"] = cur_step.get("step_type", "question-answering")
                result["id"]        = cur_step.get("id", current_idx + 1)

            if not result.get("task"):
                logger.warning(f"StepDefiner returned empty task — using plan question as fallback")
                result["task"] = cur_step_q

        except Exception as e:
            logger.warning(f"StepDefiner LLM failed: {e} — using fallback task")
            result = {
                "type":      "question-answering",
                "task":      cur_step_q,
                "step_type": self._get_step_question(cur_step)
                             if not isinstance(cur_step, dict)
                             else cur_step.get("step_type", "question-answering"),
            }

        logger.info(f"Next task: type={result.get('type')} task='{result.get('task', '')[:80]}'")
        return {"step_question": [result]}

    async def _summarize(
        self,
        question:      str,
        plan:          List[Union[Dict, str]],
        step_outputs:  List[Dict],
        step_questions: List[Dict],
        step_docs_ids: List = None,
    ) -> Dict[str, Any]:
        """
        Synthesise all step outputs into the final answer with PMID citations.
        Uses the HEAVY LLM — this is the output the user receives.
        """
        logger.info("Summarising plan execution results...")

        memory     = ""
        all_pmids  = set()

        for i, output in enumerate(step_outputs):
            # Skip error steps — don't synthesise from failures
            if output.get("is_error", False):
                continue

            step_q     = self._get_step_question(plan[i]) if i < len(plan) else "Unknown"
            detailed_q = step_questions[i].get("task", "") if i < len(step_questions) else ""
            answer     = output.get("answer", "")
            score      = output.get("rating", 0)

            # Collect PMIDs — flatten whatever shape step_docs_ids[i] is
            raw_ids  = (step_docs_ids[i] if step_docs_ids and i < len(step_docs_ids) else [])
            flat_ids = self._flatten_doc_ids(raw_ids)
            all_pmids.update(flat_ids)
            pmids_str = ", ".join(flat_ids[:5]) if flat_ids else "None"

            memory += (
                f"Task: {step_q}\n"
                f"Detailed Query: {detailed_q}\n"
                f"Answer: {answer}\n"
                f"Confidence: {score}\n"
                f"Source PMIDs: {pmids_str}\n\n"
            )

        plan_str = ", ".join([self._get_step_question(s) for s in plan])
        logger.info(f"Summary memory ({len(memory)} chars) | PMIDs: {len(all_pmids)}")

        try:
            result = await self.summary_chain.ainvoke({
                "question": question,
                "plan":     f"[{plan_str}]",
                "memory":   memory,
            })
        except Exception as e:
            logger.error(f"Summary chain failed: {e}", exc_info=True)
            result = {
                "output": "Failed",
                "answer": f"Summarisation error: {e}",
                "score":  0,
            }

        if isinstance(result, dict):
            result["cited_pmids"] = list(all_pmids)

            # Append Final Answer tag if LLM put it in final_decision
            # but forgot to include it in the answer body
            final_decision = result.get("final_decision")
            if (
                final_decision
                and final_decision.lower() in ("yes", "no", "maybe")
                and "**Final Answer:" not in result.get("answer", "")
            ):
                result["answer"] += f"\n\n**Final Answer: {final_decision.lower()}**"
                logger.info(f"Appended Final Answer tag: {final_decision}")

        return {"plan_summary": result, "stop": True}


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def step_node(state: PlanExecState) -> Dict[str, Any]:
    """Uses registry singleton — no LLM reload per call."""
    try:
        agent = AgentRegistry.get_instance().step_definer
        return await agent.define_task(state)
    except Exception as e:
        logger.error(f"step_node crashed: {e}", exc_info=True)
        return {
            "stop": True,
            "plan_summary": {"output": "Failed", "answer": f"Node crash: {e}", "score": 0},
        }