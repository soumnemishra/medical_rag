import logging
from typing import Dict, Any, Set, List

from src.state.state import GraphState

logger = logging.getLogger(__name__)


async def supplemental_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes a second retrieval pass when EvidenceDecisionAgent requests it.

    Two strategies:
      reretrieve_counter  — search for counter-evidence / opposing findings
      reretrieve_diverse  — search more broadly to reduce retrieval gaps

    Key fixes vs original:
      1. seen_pmids built from state["step_docs_ids"] (current execution),
         NOT from past_exp (retry memory, empty on first attempt).
         This was the third occurrence of this bug across the codebase.

      2. retry_count NOT incremented here. EvidenceDecisionAgent already
         increments it when it returns a reretrieve decision. Double-
         incrementing caused the retry guard (retry_count >= 1) to fire
         after zero actual retrieval attempts.

      3. RetrieverTool from registry singleton — no model reload per call.

      4. Query modification uses PubMed-appropriate terms, not natural
         language prefixes that eSearch ignores.

    Contract:
        Reads:  original_question, evidence_decision, step_docs_ids
        Writes: current_documents, current_doc_ids, evidence_decision="accept"
        Does NOT write: retry_count (owned by EvidenceDecisionAgent)
    """
    logger.info("Supplemental retrieval executing...")

    question = state["original_question"]
    decision = state.get("evidence_decision", "accept")

    # ── 1. Collect PMIDs from CURRENT execution to filter duplicates ──
    #
    # BUG HISTORY: Original code read from past_exp (retry memory).
    # past_exp is empty on the first query attempt, so seen_pmids was
    # always set() and filtering never happened — supplemental retrieval
    # returned the exact same documents as the original retrieval.
    #
    # Fix: read from step_docs_ids which contains the PMIDs retrieved
    # during THIS execution's plan steps.
    seen_pmids: Set[str] = set()
    for step_ids in state.get("step_docs_ids", []):
        ids = step_ids if isinstance(step_ids, list) else [step_ids]
        for pmid in ids:
            seen_pmids.add(str(pmid))

    logger.info(f"Filtering against {len(seen_pmids)} already-seen PMIDs")

    # ── 2. Build PubMed-appropriate search query ──────────────────────
    #
    # BUG HISTORY: Original used "evidence against {question}" and
    # "{question} review overview" — freeform English prefixes that
    # PubMed's eSearch ignores. PubMed expects MeSH terms and field tags.
    #
    # Strategy: append clinical modifier terms that PubMed indexes well.
    # For production, route through query_builder.py which handles MeSH
    # injection and Boolean assembly.
    if decision == "reretrieve_counter":
        # Add terms that surface opposing/adverse findings
        search_query = f"{question} adverse effects contraindications risks"
        logger.info(f"Counter-evidence strategy: '{search_query[:80]}'")
    elif decision == "reretrieve_diverse":
        # Add terms that broaden scope to systematic reviews and guidelines
        search_query = f"{question} systematic review meta-analysis guidelines"
        logger.info(f"Diverse retrieval strategy: '{search_query[:80]}'")
    else:
        search_query = question

    try:
        # ── 3. Use registry singleton — no model reload ───────────────
        from src.agents.registry import AgentRegistry
        retriever = AgentRegistry.get_instance().retriever

        # Retrieve with larger pool so filtering still yields enough docs
        docs, doc_ids = await retriever(search_query, top_k=20)

        # ── 4. Filter out already-seen PMIDs ──────────────────────────
        filtered_docs:  List[str] = []
        filtered_ids:   List[str] = []

        for doc_text, pmid in zip(docs, doc_ids):
            if str(pmid) not in seen_pmids:
                filtered_docs.append(doc_text)
                filtered_ids.append(pmid)

        # Cap at 10 unique documents
        final_docs = filtered_docs[:10]
        final_ids  = filtered_ids[:10]

        logger.info(
            f"Supplemental retrieval: {len(docs)} retrieved → "
            f"{len(final_docs)} unique after filtering"
        )

        return {
            # DO NOT increment retry_count here.
            # EvidenceDecisionAgent already incremented it when it
            # returned the reretrieve decision. Incrementing again would
            # cause the retry_count >= 1 guard to fire prematurely.
            "current_documents": final_docs,
            "current_doc_ids":   final_ids,
            # Reset to "accept" so the graph doesn't loop again
            # (graph conditional edges also guard this, but belt-and-suspenders)
            "evidence_decision": "accept",
        }

    except Exception as e:
        logger.error(f"Supplemental retrieval failed: {e}", exc_info=True)
        # Fail open — pipeline continues with whatever docs it has
        return {
            "evidence_decision": "accept",
        }