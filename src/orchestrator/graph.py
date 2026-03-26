'''this file is brain orchestrator  on the entire system 
it builds a multiagent workflow graph like a pipeline of decision making and task execution

think of this like a hspital work flow system where patient comes in and goes through different
 departments (triage, diagnosis, treatment) before discharge.

 patient come in == start 
 Nurse checks symptoms (clinical intent)
 Receptionist routes patient to right department (router): 
 --> simple case : asnwers doctor 
 --> complex case : specialist team 

 Specialist team ananlsyes and makes a step by step plan and executes it (planar+ executor)
 Reports checked for quality (evidence polarity and decision)

 final review for safety and discharge (safety critic)
 patient gets answer 
'''
# think logging like a cctv  tracing the things up 
import logging  #logging is essential for production . we need to see where the query fails during productions 

# type hinting we use to specify tht our wrapper node returns dictionary with string keys and anytype values 
#type hinting tells the devloper what kind of data is returned 
######### makes code readable and avoids bugs 
from typing import Dict, Any

# STATEGRAPH IS THE CLASS WHERE WE DEFINE THE WORKFLOW .start and end node are special nodes that represents the begining and the end 
####### sate graph builds workflow ################ 
# where in state graph start is the entry node and end is the exit node 

##################  lets take example of  google maps #################
############## NODES - ARE CITIES OR LANDMARKS   EDGES : ROADS  START : SOURCE   END : DESTINATION 
from langgraph.graph import StateGraph, START, END
############ GRAPH STATE : defines the data structures that fow through the graph and shared by all agents 
########### think of like pateient file shared between the doctors in the hospital 
from src.state.state import GraphState, PlanExecState
from src.agents.planner               import planner_node
from src.agents.executor              import get_executor_graph   # singleton — not build_executor_graph
from src.agents.clinical_intent       import clinical_intent_node
from src.agents.safety_critic         import safety_critic_node
from src.agents.router_agent          import router_node
from src.agents.rag_node              import rag_direct_node
from src.agents.evidence_polarity_agent  import evidence_polarity_node
from src.agents.decision_alignment    import decision_alignment_node
from src.agents.evidence_decision_agent import evidence_decision_node
from src.agents.supplemental_retrieval_node import supplemental_retrieval_node

logger = logging.getLogger(__name__)


def build_graph():

    """
    the build graph() is the full system (orchestra or brain)
    executor() is the subgraph that executes the

    the build graph builds the entire workflow pipeline 
    Builds the main MA-RAG Orchestration Graph.

    Full pipeline:

        START
          └─► clinical_intent
                └─► router
                      ├─► rag_direct  (direct_qa mode)
                      └─► planner     (disambiguation / multihop)
                            └─► [empty plan guard] → safety_critic → END
                            └─► executor
                                  └─► evidence_polarity
                                        └─► evidence_decision
                                              ├─► decision_alignment → safety_critic → END
                                              └─► supplemental_retrieval
                                                    └─► rag_direct  (retry path)
                                                          └─► evidence_polarity → ... → END

    Key design notes:
      - get_executor_graph() is a singleton — models load once, not on every call.
      - Empty plan from PlannerAgent is caught BEFORE the executor to prevent
        silent empty-answer failures.
      - executor_wrapper_node lifts step_output / step_docs_ids / step_notes
        to GraphState top level so EvidencePolarityAgent can read them.
      - After supplemental_retrieval, rag_direct re-runs on new documents.
        The retry path then goes through the full evidence pipeline again.
        EvidenceDecisionAgent forces 'accept' when retry_count >= 1 so
        the loop terminates after one retry.
    """

    # Singleton executor graph — built once, reused across all queries
    executor_graph = get_executor_graph()

    # ---------------------------------------------------------------- #
    #  Executor wrapper node                                            #
    # ---------------------------------------------------------------- #

    async def executor_wrapper_node(state: GraphState) -> Dict[str, Any]:
        """
        Runs the nested executor graph for multi-step plan execution.

        Critical responsibilities:
          1. Safely read plan — use .get() not [] to avoid KeyError
          2. Propagate clinical context into executor sub-state
          3. Lift step_output / step_docs_ids / step_notes to GraphState
             top level so EvidencePolarityAgent can read them directly
          4. Append cited PMIDs to final_answer if not already present
        """
        plan              = state.get("plan", [])
        original_question = state["original_question"]
        intent            = state.get("intent",              "informational")
        risk_level        = state.get("risk_level",          "low")
        needs_guidelines  = state.get("needs_guidelines",    False)
        req_disclaimer    = state.get("requires_disclaimer", False)

        input_state: PlanExecState = {
            "original_question":  original_question,
            "plan":               plan,
            "step_question":      [],
            "step_output":        [],
            "step_docs_ids":      [],
            "step_notes":         [],
            "plan_summary":       {"output": "", "answer": "", "score": 0},
            "stop":               False,
            "intent":             intent,
            "risk_level":         risk_level,
            "needs_guidelines":   needs_guidelines,
            "requires_disclaimer": req_disclaimer,
        }

        logger.info(f"Executor starting | {len(plan)} step(s) | question='{original_question[:60]}'")
        full_output = await executor_graph.ainvoke(input_state)

        plan_summary = full_output.get("plan_summary", {})
        final_answer = plan_summary.get("answer", "No answer generated.")

        # Append reference section if PMIDs not already cited inline
        cited_pmids = plan_summary.get("cited_pmids", [])
        if cited_pmids and "PMID" not in final_answer and "References" not in final_answer:
            refs = "\n\n**References:**\n" + "\n".join(
                [f"- PMID:{pmid}" for pmid in cited_pmids[:10]]
            )
            final_answer += refs

        # ── CRITICAL: surface step data to GraphState top level ─────
        # EvidencePolarityAgent reads state["step_notes"] and
        # state["step_docs_ids"] from GraphState directly.
        # Without lifting these from full_output, the polarity agent
        # always gets empty lists and returns "insufficient" for every
        # multihop query — breaking the entire evidence quality pipeline.
        return {
            "past_exp":      [full_output],
            "final_answer":  final_answer,
            "step_output":   full_output.get("step_output",   []),
            "step_docs_ids": full_output.get("step_docs_ids", []),
            "step_notes":    full_output.get("step_notes",    []),
        }

    # ---------------------------------------------------------------- #
    #  Build graph                                                      #
    # ---------------------------------------------------------------- #

    workflow = StateGraph(GraphState)

    workflow.add_node("clinical_intent",        clinical_intent_node)
    workflow.add_node("router",                  router_node)
    workflow.add_node("planner",                 planner_node)
    workflow.add_node("executor",                executor_wrapper_node)
    workflow.add_node("rag_direct",              rag_direct_node)
    workflow.add_node("evidence_polarity",       evidence_polarity_node)
    workflow.add_node("evidence_decision",       evidence_decision_node)
    workflow.add_node("supplemental_retrieval",  supplemental_retrieval_node)
    workflow.add_node("decision_alignment",      decision_alignment_node)
    workflow.add_node("safety_critic",           safety_critic_node)

    # ── Entry ────────────────────────────────────────────────────────
    workflow.add_edge(START, "clinical_intent")
    workflow.add_edge("clinical_intent", "router")

    # ── Router → execution path ──────────────────────────────────────
    def route_decision(state: GraphState) -> str:
        mode = state.get("router_output", {}).get("execution_mode", "multihop")
        if mode == "direct_qa":
            return "rag_direct"
        # "disambiguation" and "multihop" both use the full planning pipeline
        return "planner"

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"rag_direct": "rag_direct", "planner": "planner"},
    )

    # ── Planner → empty plan guard ────────────────────────────────────
    # If PlannerAgent returns plan=[] or plan_error is set, skip the
    # executor entirely and go straight to safety_critic which will
    # surface the error in the final answer.
    def check_plan_valid(state: GraphState) -> str:
        plan = state.get("plan", [])
        if not plan or state.get("plan_error"):
            logger.warning(
                f"Empty or failed plan — skipping executor. "
                f"plan_error={state.get('plan_error')}"
            )
            return "safety_critic"
        return "executor"

    workflow.add_conditional_edges(
        "planner",
        check_plan_valid,
        {"executor": "executor", "safety_critic": "safety_critic"},
    )

    # ── Complex path: executor → polarity ────────────────────────────
    workflow.add_edge("executor", "evidence_polarity")

    # ── Direct path: rag_direct → polarity ───────────────────────────
    # Both direct and retry paths feed into polarity
    workflow.add_edge("rag_direct", "evidence_polarity")

    # ── Evidence quality gate ─────────────────────────────────────────
    workflow.add_edge("evidence_polarity", "evidence_decision")

    def check_evidence_decision(state: GraphState) -> str:
        decision = state.get("evidence_decision", "accept")
        if decision == "accept":
            return "decision_alignment"
        return "supplemental_retrieval"

    workflow.add_conditional_edges(
        "evidence_decision",
        check_evidence_decision,
        {
            "decision_alignment":    "decision_alignment",
            "supplemental_retrieval": "supplemental_retrieval",
        },
    )

    # ── Retry loop ────────────────────────────────────────────────────
    # After supplemental_retrieval fetches new docs (stored in
    # current_documents / current_doc_ids), rag_direct_node reads them
    # via state.get("current_documents", []).
    # The retry then flows: rag_direct → evidence_polarity → evidence_decision
    # → (retry_count >= 1 forces accept) → decision_alignment → safety_critic
    #
    # Note: multihop queries use a single RAG call on supplemental docs
    # rather than re-running the full plan. This is an intentional
    # tradeoff — better evidence in one focused call vs. full re-planning.
    workflow.add_edge("supplemental_retrieval", "rag_direct")

    # ── Final pipeline ────────────────────────────────────────────────
    workflow.add_edge("decision_alignment", "safety_critic")
    workflow.add_edge("safety_critic", END)

    return workflow.compile()