from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import GraphState, PlanFormat
from src.prompts.templates import PLANNING_SYSTEM_PROMPT, PLANNING_HUMAN_PROMPT
from src.core.registry import ModelRegistry
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import logging

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Agent responsible for breaking down a complex medical query into a
    structured, dependency-aware execution plan.

    Reads intent, risk_level, and needs_guidelines from GraphState so the
    plan is tailored to the clinical context set by ClinicalIntentAgent.

    Output plan steps carry 'id', 'question', 'depends_on', and 'step_type'
    fields so the Executor can run independent steps in parallel.
    """

    def __init__(self):
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        # Fail loud at startup — not silently on first query
        if self.llm is None:
            raise RuntimeError(
                "PlannerAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        # PlanFormat enforces 'id', 'question', 'depends_on', 'step_type'
        self.parser = JsonOutputParser(pydantic_object=PlanFormat)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNING_SYSTEM_PROMPT),
            ("human", PLANNING_HUMAN_PROMPT)
        ])

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """
        Runs the prompt → LLM → parser chain with up to 2 retries.
        Retries on transient API errors AND on JsonOutputParser parse failures.
        Keeping this as a separate method makes it independently testable.
        """
        chain = self.prompt | self.llm | self.parser
        return await chain.ainvoke(inputs)

    def _format_memory(self, past_exp: List[Dict[str, Any]]) -> str:
        """
        Convert past execution attempts into a human-readable string the
        Planner LLM can learn from.

        Surfaces: which steps ran, whether they succeeded, how many docs were
        found, and a 'lesson' field so the LLM can avoid repeating bad plans.
        """
        if not past_exp:
            return "No past experience."

        lines = []
        for i, exp in enumerate(past_exp):
            lines.append(f"Attempt {i + 1}:")

            # Extract lists once — avoids repeated .get() and index mismatches
            plan_steps   = exp.get("plan", [])
            step_outputs = exp.get("step_output", [])
            step_docs    = exp.get("step_docs_ids", [])

            for j, step in enumerate(plan_steps):
                step_q  = step.get("question", step) if isinstance(step, dict) else step
                result  = step_outputs[j] if j < len(step_outputs) else {}
                docs    = step_docs[j]    if j < len(step_docs)    else []
                status  = result.get("success", "unknown") if isinstance(result, dict) else "unknown"
                lines.append(
                    f"  Step {j + 1}: '{step_q}' → {status}, {len(docs)} docs found"
                )

            summary   = exp.get("plan_summary", {})
            score_val = summary.get("score", 0) if isinstance(summary, dict) else 0
            lines.append(f"  Overall score: {score_val}/1.0")
            lines.append(f"  Lesson: {exp.get('lesson', 'none recorded')}")

        return "\n".join(lines)

    def _validate_depends_on(self, steps: List[Dict]) -> List[Dict]:
        """
        Strips invalid, self-referencing, or cyclic dependency IDs.

        Why this matters: if the LLM returns depends_on: [5] but only 3 steps
        exist, the parallel executor waits forever for step 5 — a silent
        deadlock.  If a cycle exists (step 2 → step 3 → step 2), same result.

        On cycle detection we strip ALL depends_on and fall back to sequential
        execution — slower but never deadlocked.
        """
        valid_ids = {s["id"] for s in steps}

        # Pass 1 — remove IDs that don't exist or are self-references
        for step in steps:
            step["depends_on"] = [
                d for d in step.get("depends_on", [])
                if d in valid_ids and d != step["id"]
            ]

        # Pass 2 — cycle detection via DFS
        dep_map = {s["id"]: set(s["depends_on"]) for s in steps}

        def has_cycle() -> bool:
            visited, in_stack = set(), set()

            def dfs(node: int) -> bool:
                visited.add(node)
                in_stack.add(node)
                for neighbour in dep_map.get(node, set()):
                    if neighbour not in visited:
                        if dfs(neighbour):
                            return True
                    elif neighbour in in_stack:
                        return True
                in_stack.discard(node)
                return False

            return any(dfs(s["id"]) for s in steps if s["id"] not in visited)

        if has_cycle():
            logger.warning(
                "Dependency cycle detected in plan — stripping all depends_on "
                "and falling back to sequential execution."
            )
            for step in steps:
                step["depends_on"] = []

        return steps

    def _normalize_steps(self, raw_steps: List[Any]) -> List[Dict]:
        """
        Coerce whatever the LLM returned into the canonical step format:
            {"id": int, "question": str, "depends_on": List[int], "step_type": str}

        Handles three LLM output variants:
          - Correct dict with all fields       → kept as-is
          - Dict missing 'id' or 'question'   → repaired
          - Raw string                         → wrapped in dict
          - Anything else                      → str() conversion
        """
        steps = []
        for raw in raw_steps:
            if isinstance(raw, dict):
                if "id" not in raw or "question" not in raw:
                    steps.append({
                        "id":         len(steps) + 1,
                        "question":   raw.get("question", str(raw)),
                        "depends_on": raw.get("depends_on", []),
                        "step_type":  raw.get("step_type", "question-answering"),
                    })
                else:
                    # Ensure depends_on and step_type exist even in a good dict
                    raw.setdefault("depends_on", [])
                    raw.setdefault("step_type", "question-answering")
                    steps.append(raw)
            elif isinstance(raw, str):
                steps.append({
                    "id":         len(steps) + 1,
                    "question":   raw,
                    "depends_on": [],
                    "step_type":  "question-answering",
                })
            else:
                steps.append({
                    "id":         len(steps) + 1,
                    "question":   str(raw),
                    "depends_on": [],
                    "step_type":  "question-answering",
                })
        return steps

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    async def plan(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a dependency-aware execution plan for the given query.

        Reads from GraphState:
          - original_question   the user's query
          - past_exp            previous failed attempts (for retry learning)
          - intent              from ClinicalIntentAgent
          - risk_level          from ClinicalIntentAgent
          - needs_guidelines    from ClinicalIntentAgent

        Returns GraphState update with:
          - plan                List of structured step dicts
          - plan_complexity     "simple" | "moderate" | "complex"
          - plan_error          only on failure
          - safety_flags        appended on failure
        """
        try:
            question        = state["original_question"]
            past_exp        = state.get("past_exp", [])
            intent          = state.get("intent", "informational")
            risk_level      = state.get("risk_level", "low")
            needs_guidelines = state.get("needs_guidelines", False)

            memory = self._format_memory(past_exp)

            logger.info(
                f"PLANNING | question='{question}' "
                f"intent={intent} risk={risk_level} guidelines={needs_guidelines}"
            )

            try:
                result = await self._invoke_chain({
                    "question":         question,
                    "memory":           memory,
                    "intent":           intent,
                    "risk_level":       risk_level,
                    "needs_guidelines": needs_guidelines,
                })
            except RetryError as retry_err:
                # Both retry attempts failed — surface cleanly
                raise ValueError(
                    f"LLM chain failed after 2 attempts: {retry_err}"
                ) from retry_err

            # Normalise whatever shape the LLM returned
            raw_steps = result.get("plan", [])
            steps = self._normalize_steps(raw_steps)

            # Guard: empty plan should never reach the executor
            if not steps:
                logger.error(
                    "Planner produced zero steps. "
                    "LLM likely returned wrong JSON key. "
                    f"Raw result keys: {list(result.keys())}"
                )
                return {
                    "plan":         [],
                    "safety_flags": state.get("safety_flags", []) + ["empty_plan"],
                }

            # Validate dependency graph — strips bad IDs, detects cycles
            steps = self._validate_depends_on(steps)

            logger.info(
                f"Plan ready ({len(steps)} steps, "
                f"complexity={result.get('complexity', 'moderate')}): "
                f"{[s['question'] for s in steps]}"
            )

            return {
                "plan":             steps,
                "plan_complexity":  result.get("complexity", "moderate"),
            }

        except Exception as e:
            logger.error(f"Planning failed: {e}", exc_info=True)
            return {
                "plan":         [],
                "plan_error":   str(e),
                "safety_flags": state.get("safety_flags", []) + ["planning_failed"],
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                              #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Thin wrapper called by the LangGraph StateGraph.
    Gets the singleton PlannerAgent from the registry so the LLM and prompt
    are initialised once and reused across queries.
    """
    try:
        agent = AgentRegistry.get_instance().planner
        return await agent.plan(state)
    except Exception as e:
        logger.critical(f"planner_node crashed: {e}", exc_info=True)
        return {
            "plan":         [],
            "plan_error":   str(e),
            "safety_flags": state.get("safety_flags", []) + ["planner_node_crash"],
        }