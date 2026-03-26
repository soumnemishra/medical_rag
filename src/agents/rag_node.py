import logging
from typing import Dict, Any

from src.state.state import GraphState, RagState
from src.agents.rag import RagAgent

logger = logging.getLogger(__name__)


async def rag_direct_node(state: GraphState) -> Dict[str, Any]:
    """
    Node for the 'Direct QA' execution path — used when the router
    decides the query is simple enough to skip the full plan/executor
    pipeline and answer directly.

    Key design decisions vs original:
      1. Uses AgentRegistry singleton — no RetrieverTool reload per call.
      2. Does NOT write to past_exp. past_exp is the planner's retry
         memory. Injecting a fake 'Direct QA' entry would corrupt the
         planner's learning on any subsequent retry attempt.
      3. Passes evidence_polarity to RagAgent so the QA prompt receives
         the {evidence_polarity} variable now required by QA_HUMAN_PROMPT.
      4. Writes step_output and step_docs_ids so SafetyCriticAgent and
         DecisionAlignmentAgent have the data they need.
      5. force_commitment no-op removed — dead code in a medical system
         is a maintenance hazard.

    Contract:
        Input:  GraphState (set by RouterAgent, ClinicalIntentAgent)
        Output: Updates final_answer, step_output, step_docs_ids
                Does NOT touch past_exp.
    """
    logger.info("Direct QA path executing (rag_direct_node)...")

    question = state["original_question"]

    # ── Build RagState from GraphState ─────────────────────────────
    # RagAgent expects: question, documents, doc_ids, evidence_polarity
    # Pre-fetched documents from SupplementalRetrievalNode (if present)
    # are forwarded so retrieval is not duplicated.

    evidence_polarity = state.get("evidence_polarity", {})

    rag_input: RagState = {
        "question":         question,
        "documents":        state.get("current_documents", []),
        "doc_ids":          state.get("current_doc_ids", []),
        "notes":            [],
        "final_raw_answer": {},
        "intent":           state.get("intent",      "informational"),
        "risk_level":       state.get("risk_level",  "low"),
        "safety_flags":     state.get("safety_flags", []),
        # Pass through polarity so RagAgent feeds {evidence_polarity}
        # in QA_HUMAN_PROMPT — without this, LangChain raises KeyError
        "evidence_polarity": evidence_polarity,
    }

    try:
        from src.agents.registry import AgentRegistry
        agent = AgentRegistry.get_instance().rag
        result = await agent.query(rag_input)

        final_raw  = result.get("final_raw_answer", {})
        answer_text = final_raw.get("answer", "No answer generated.")
        doc_ids     = result.get("doc_ids", [])

        # Build a step_output entry so SafetyCriticAgent and
        # DecisionAlignmentAgent have consistent data to read.
        # Format matches what ExecutorAgent produces — one dict per step.
        step_output_entry = {
            "analysis": final_raw.get("analysis", "Direct QA execution"),
            "answer":   answer_text,
            "success":  final_raw.get("success", "Yes"),
            "rating":   final_raw.get("rating",  8),
            "is_error": final_raw.get("is_error", False),
        }

        logger.info(
            f"Direct QA complete | "
            f"answer='{answer_text[:100]}...' | "
            f"docs={len(doc_ids)}"
        )

        # IMPORTANT: do NOT write to past_exp.
        # past_exp is the planner's retry memory. Writing a fake 'Direct QA'
        # entry would cause PlannerAgent._format_memory() to believe a full
        # plan was attempted and succeeded, making retry plans incoherent.
        return {
            "final_answer":  answer_text,
            "step_output":   [step_output_entry],
            "step_docs_ids": doc_ids,
            "step_notes":    result.get("notes", []),
        }

    except Exception as e:
        logger.error(f"rag_direct_node failed: {e}", exc_info=True)
        return {
            "final_answer":  "I encountered an error during direct search.",
            "step_output":   [{
                "answer":   "Error in direct QA path.",
                "success":  "No",
                "is_error": True,
                "error_message": str(e),
            }],
            "step_docs_ids": [],
            "step_notes":    [],
        }