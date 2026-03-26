import asyncio
import logging
import operator
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state.state import PlanExecState, GraphState
from src.prompts.templates import AGGREGATE_SYSTEM_PROMPT, AGGREGATE_HUMAN_PROMPT
from src.core.registry import ModelRegistry
from src.agents.registry import AgentRegistry
from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Output schema for the aggregate step                               #
# ------------------------------------------------------------------ #

class AggregateOutput(BaseModel):
    """
    Pydantic schema for the aggregate LLM output.
    Enforces structure so a wrong key name throws immediately
    instead of silently returning an empty dict.
    """
    summary:    str       = Field(description="Synthesised answer across all step findings with inline citations")
    analysis:   str       = Field(description="Reasoning process showing how step findings were combined")
    confidence: str       = Field(description="HIGH | MEDIUM | LOW — based on evidence quality")
    citations:  List[str] = Field(default_factory=list, description="List of PMID strings cited")


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _smart_truncate(text: str, max_chars: int = 500) -> str:
    """
    Truncate text at the last sentence boundary before max_chars.

    Why: c[:500] cuts mid-sentence. The RAG agent then receives
    a context ending in '...concomitant use of warfarin with' which
    degrades answer coherence. This finds the last '.' before the
    limit so the context always ends at a complete thought.
    """
    if len(text) <= max_chars:
        return text
    boundary = text[:max_chars].rfind('.')
    return text[:boundary + 1] if boundary > 0 else text[:max_chars]


def _collect_prior_context(state: PlanExecState) -> tuple[str, str, List[str]]:
    """
    Gather all prior step findings, evidence notes, and doc IDs
    from state for use by the aggregate step.

    Returns:
        findings_block  — formatted step answers (errors excluded)
        notes_block     — raw evidence notes per step
        all_doc_ids     — flat list of all cited PMIDs so far
    """
    prior_outputs = state.get("step_output", [])
    prior_notes   = state.get("step_notes", [])

    findings_block = "\n\n".join([
        f"Finding from step {i + 1}:\n{out.get('answer', '')}"
        for i, out in enumerate(prior_outputs)
        if not out.get("is_error", False)
    ])

    notes_block = "\n\n".join([
        f"Evidence notes step {i + 1}:\n{note}"
        for i, note in enumerate(prior_notes)
        if note
    ])

    all_doc_ids = [
        doc_id
        for ids_per_step in state.get("step_docs_ids", [])
        for doc_id in (ids_per_step if isinstance(ids_per_step, list) else [])
    ]

    return findings_block, notes_block, all_doc_ids


# ------------------------------------------------------------------ #
#  Module-level singleton — graph is built once, reused forever      #
# ------------------------------------------------------------------ #

_executor_graph = None


def get_executor_graph():
    """
    Returns the compiled executor graph.

    Why singleton: build_executor_graph() used to be called on every
    query, which reloaded RetrieverTool sentence-transformer models
    (~2-5s each time) and recompiled the LangGraph. This builds the
    graph exactly once per process lifetime.
    """
    global _executor_graph
    if _executor_graph is None:
        _executor_graph = _build_executor_graph()
    return _executor_graph


# Keep old name as an alias so any existing call sites don't break
def build_executor_graph():
    return get_executor_graph()


# ------------------------------------------------------------------ #
#  Graph builder — called once                                        #
# ------------------------------------------------------------------ #

def _build_executor_graph():
    """
    Internal builder. Only called once by get_executor_graph().

    Architecture:
        START
          └─► task_definer_node
                ├─► END          (when stop=True)
                └─► execution_node
                      └─► task_definer_node  (loop)

    For multi-step plans with dependency metadata (set by PlannerAgent),
    use run_parallel_plan() from the main graph instead of this loop —
    it fires independent steps with asyncio.gather() for ~50% latency
    reduction on complex queries.
    """

    # ── Agents from registry (singleton — not re-created here) ─────
    registry       = AgentRegistry.get_instance()
    step_definer   = registry.step_definer
    rag_agent      = registry.rag
    extractor_agent = registry.extractor

    # RetrieverTool should live in the registry.
    # Use registry.retriever if available, else fall back to direct init.
    retriever_tool = getattr(registry, 'retriever', None) or RetrieverTool()

    # ── LLM and aggregate chain ─────────────────────────────────────
    llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

    if llm is None:
        raise RuntimeError(
            "Executor: Heavy LLM failed to load. "
            "Check Ollama is running or GOOGLE_API_KEY is set."
        )

    aggregate_parser = JsonOutputParser(pydantic_object=AggregateOutput)
    aggregate_prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATE_SYSTEM_PROMPT),
        ("human",  AGGREGATE_HUMAN_PROMPT),
    ])

    # ── Node definitions ────────────────────────────────────────────

    async def task_definer_node(state: PlanExecState) -> Dict[str, Any]:
        """
        Delegates to StepDefinerAgent which reads the plan, picks the
        next un-executed step, writes it to step_question, and sets
        stop=True when all steps are done.
        """
        return await step_definer.define_task(state)

    async def execution_node(state: PlanExecState) -> Dict[str, Any]:
        """
        Executes one plan step — either aggregate synthesis or RAG QA.

        RAG path:   retrieve → extract → answer
        Aggregate:  synthesise across all prior step outputs

        All errors are tagged with is_error=True so they can be filtered
        out by the aggregator rather than being included as real findings.
        """
        step_id = -1  # tracked for error reporting

        try:
            step_questions = state.get("step_question", [])
            if not step_questions:
                logger.error("execution_node: no step_question in state")
                return _make_error_output(
                    "No step_question found in state", step_id
                )

            current_task_info = step_questions[-1]
            task_type = current_task_info.get("type", "question-answering")
            query     = current_task_info.get("task", "")
            step_id   = current_task_info.get("id", -1)

            logger.info(f"Executing step {step_id} | type={task_type} | query='{query[:80]}...'")

            # ── Aggregate path ──────────────────────────────────────
            if task_type == "aggregate":
                findings_block, notes_block, all_doc_ids = _collect_prior_context(state)

                chain    = aggregate_prompt | llm | aggregate_parser
                response = await chain.ainvoke({
                    "question":       query,
                    "step_findings":  findings_block,   # prior step answers
                    "evidence_notes": notes_block,      # raw extracted notes
                    "doc_ids":        str(all_doc_ids), # for citation
                })

                return {
                    "step_output":   [response],
                    "step_docs_ids": all_doc_ids,
                    "step_notes":    [response.get("analysis", "")],
                }

            # ── RAG path ────────────────────────────────────────────
            clinical_guidelines_needed = state.get("needs_guidelines", False)
            contexts, doc_ids = await retriever_tool(
                query, requires_guidelines=clinical_guidelines_needed
            )

            if not contexts:
                logger.warning(f"Step {step_id}: retriever returned 0 documents")
                notes_text = "No documents found for this query."
            else:
                # Per-step fast-path decision using step_type from planner,
                # not total plan length (the old condition was always wrong
                # for multi-step plans).
                current_step_type     = current_task_info.get("step_type", "question-answering")
                is_simple_informational = (
                    state.get("intent") == "informational"
                    and current_step_type == "simple"
                )

                if is_simple_informational:
                    logger.info(f"Step {step_id}: fast-path (simple informational)")
                    notes_text = "\n\n".join([
                        _smart_truncate(c) for c in contexts[:3]
                    ])
                else:
                    logger.info(f"Step {step_id}: running ExtractorAgent")
                    extraction_result = await extractor_agent.extract(query, contexts)
                    if isinstance(extraction_result, dict):
                        notes_text = extraction_result.get("notes", "")
                    else:
                        notes_text = str(extraction_result)

            logger.info(f"Step {step_id} notes preview: '{notes_text[:150]}...'")

            rag_result = await rag_agent.query({
                "question":  query,
                "documents": [notes_text],
                "doc_ids":   doc_ids,
            })

            answer_state          = rag_result["final_raw_answer"]
            answer_state["step_id"] = step_id  # tag for downstream tracking

            logger.info(
                f"Step {step_id} answer: '{answer_state.get('answer','')[:150]}...'"
            )

            return {
                "step_output":   [answer_state],
                "step_docs_ids": doc_ids,
                "step_notes":    [notes_text] if notes_text else [""],
            }

        except Exception as e:
            logger.error(f"execution_node step {step_id} failed: {e}", exc_info=True)
            return _make_error_output(str(e), step_id)

    # ── Conditional edge ────────────────────────────────────────────

    def check_stop(state: PlanExecState) -> str:
        """Exit the loop when StepDefinerAgent sets stop=True."""
        if state.get("stop"):
            return END
        return "execution_node"

    # ── Graph wiring ────────────────────────────────────────────────

    workflow = StateGraph(PlanExecState)

    workflow.add_node("task_definer",  task_definer_node)
    workflow.add_node("execution_node", execution_node)

    workflow.add_edge(START, "task_definer")

    workflow.add_conditional_edges(
        "task_definer",
        check_stop,
        {END: END, "execution_node": "execution_node"},
    )

    workflow.add_edge("execution_node", "task_definer")

    return workflow.compile()


# ------------------------------------------------------------------ #
#  Parallel plan runner — use this for multi-step plans              #
# ------------------------------------------------------------------ #

async def _execute_single_step(
    step:          Dict[str, Any],
    state:         GraphState,
    prior_results: Dict[int, Dict],
) -> Dict[str, Any]:
    """
    Executes one plan step independently.
    Called concurrently by run_parallel_plan via asyncio.gather().

    'prior_results' contains outputs of already-completed steps so
    that aggregate steps can read them without accessing shared state.
    """
    registry        = AgentRegistry.get_instance()
    rag_agent       = registry.rag
    extractor_agent = registry.extractor
    retriever_tool  = getattr(registry, 'retriever', None) or RetrieverTool()

    llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
    aggregate_parser = JsonOutputParser(pydantic_object=AggregateOutput)
    aggregate_prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATE_SYSTEM_PROMPT),
        ("human",  AGGREGATE_HUMAN_PROMPT),
    ])

    query     = step.get("question", "")
    step_id   = step.get("id", -1)
    step_type = step.get("step_type", "question-answering")

    try:
        # ── Aggregate step ─────────────────────────────────────────
        if step_type == "aggregate":
            valid_prior = [
                v for v in prior_results.values()
                if not v.get("is_error", False)
            ]
            findings_block = "\n\n".join([
                f"Finding from step {r.get('step_id','?')}:\n{r.get('answer','')}"
                for r in valid_prior
            ])
            notes_block = "\n\n".join([
                r.get("notes", "") for r in valid_prior if r.get("notes")
            ])
            all_doc_ids = [
                d for r in valid_prior for d in r.get("doc_ids", [])
            ]

            chain    = aggregate_prompt | llm | aggregate_parser
            response = await chain.ainvoke({
                "question":       query,
                "step_findings":  findings_block,
                "evidence_notes": notes_block,
                "doc_ids":        str(all_doc_ids),
            })
            return {
                "answer":   response.get("summary", ""),
                "analysis": response.get("analysis", ""),
                "doc_ids":  all_doc_ids,
                "notes":    response.get("analysis", ""),
                "step_id":  step_id,
                "success":  "Yes",
                "is_error": False,
            }

        # ── RAG step ───────────────────────────────────────────────
        clinical_guidelines_needed = state.get("needs_guidelines", False)
        contexts, doc_ids = await retriever_tool(
            query, requires_guidelines=clinical_guidelines_needed
        )

        if not contexts:
            notes_text = "No documents found."
        else:
            is_simple = (
                state.get("intent") == "informational"
                and step_type == "simple"
            )
            if is_simple:
                notes_text = "\n\n".join([_smart_truncate(c) for c in contexts[:3]])
            else:
                extraction_result = await extractor_agent.extract(query, contexts)
                notes_text = (
                    extraction_result.get("notes", "")
                    if isinstance(extraction_result, dict)
                    else str(extraction_result)
                )

        rag_result   = await rag_agent.query({
            "question":  query,
            "documents": [notes_text],
            "doc_ids":   doc_ids,
        })
        answer_state = rag_result["final_raw_answer"]

        return {
            "answer":   answer_state.get("answer", ""),
            "analysis": answer_state.get("analysis", ""),
            "doc_ids":  doc_ids,
            "notes":    notes_text,
            "step_id":  step_id,
            "success":  "Yes",
            "is_error": False,
        }

    except Exception as e:
        logger.error(f"Step {step_id} failed in parallel runner: {e}", exc_info=True)
        return _make_error_output(str(e), step_id)


async def run_parallel_plan(state: GraphState) -> Dict[str, Any]:
    """
    Dependency-aware parallel plan executor.

    Replaces the sequential LangGraph loop for plans that have
    depends_on metadata (output by the updated PlannerAgent).

    Algorithm:
      1. Find all steps whose dependencies are already completed.
      2. Fire them ALL simultaneously with asyncio.gather().
      3. Mark them complete, repeat until all steps done.

    Example — 4 step plan:
      Steps 1, 2, 3 have depends_on=[] → fire together  (~15s)
      Step 4 has depends_on=[1,2,3]    → fires after    (~15s)
      Total: ~30s instead of ~60s sequential

    asyncio.gather(return_exceptions=True) means one step failing
    does NOT cancel other running steps.
    """
    plan      = state.get("plan", [])
    results   = {}    # step_id → output dict
    completed = set()

    if not plan:
        logger.error("run_parallel_plan called with empty plan")
        return {"step_output": [], "step_docs_ids": [], "step_notes": []}

    while len(completed) < len(plan):
        ready = [
            s for s in plan
            if s["id"] not in completed
            and all(dep in completed for dep in s.get("depends_on", []))
        ]

        if not ready:
            logger.error(
                "Parallel executor deadlocked — no ready steps. "
                f"Completed: {completed}, Remaining: "
                f"{[s['id'] for s in plan if s['id'] not in completed]}"
            )
            break

        logger.info(
            f"Parallel batch: firing steps {[s['id'] for s in ready]} simultaneously"
        )

        # Fire all ready steps at the same time
        outputs = await asyncio.gather(*[
            _execute_single_step(s, state, results)
            for s in ready
        ], return_exceptions=True)

        for step, output in zip(ready, outputs):
            if isinstance(output, Exception):
                logger.error(f"Step {step['id']} raised unhandled: {output}")
                results[step["id"]] = _make_error_output(str(output), step["id"])
            else:
                results[step["id"]] = output
            completed.add(step["id"])

    # Preserve the original plan order in the final output lists
    ordered = [results[s["id"]] for s in plan if s["id"] in results]

    return {
        "step_output":   ordered,
        "step_docs_ids": [r.get("doc_ids", []) for r in ordered],
        "step_notes":    [r.get("notes", "")   for r in ordered],
    }


# ------------------------------------------------------------------ #
#  Shared error output factory                                        #
# ------------------------------------------------------------------ #

def _make_error_output(error_msg: str, step_id: int) -> Dict[str, Any]:
    """
    Produces a tagged error dict that aggregators can detect and skip.

    Why: previously errors used {"answer": "Error executing step."}
    which got synthesised into the final clinical answer alongside real
    findings.  The is_error flag lets aggregators filter them out.
    """
    return {
        "step_output": [{
            "answer":        "",       # empty — nothing to synthesise
            "is_error":      True,     # aggregator filters on this
            "error_message": error_msg,
            "step_id":       step_id,
            "success":       "No",
            "rating":        0,
        }],
        "step_docs_ids": [],
        "step_notes":    [""],
    }