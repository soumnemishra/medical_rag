import logging
from typing import Dict, Any, List, Set
from src.state.state import GraphState
from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)

async def supplemental_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes a second retrieval pass if EvidenceDecisionAgent requests it.
    Filters out PMIDs already seen in past_exp.
    Updates 'current_documents' (context) and increments 'retry_count'.
    """
    logger.info("Executing Supplemental Retrieval...")
    
    question = state["original_question"]
    decision = state.get("evidence_decision", "accept")
    retry_count = state.get("retry_count", 0)
    past_exp = state.get("past_exp", [])
    
    # 1. Collect seen PMIDs to avoid redundancy
    seen_pmids: Set[str] = set()
    for exp in past_exp:
        for ids in exp.get("step_docs_ids", []):
            if isinstance(ids, list):
                seen_pmids.update(ids)
            else:
                seen_pmids.add(str(ids))
    
    # 2. Modify Query based on Decision
    search_query = question
    if decision == "reretrieve_counter":
        # Attempt to find conflicting evidence
        search_query = f"evidence against {question}"
        logger.info(f"Supplemental Strategy: Counter-Evidence ('{search_query}')")
    elif decision == "reretrieve_diverse":
        # Broader search
        search_query = f"{question} review overview"
        logger.info(f"Supplemental Strategy: Diverse/Broad ('{search_query}')")
        
    try:
        # 3. Execute Retrieval
        # We use a larger top_k to allow for filtering
        retriever = RetrieverTool(top_k=20, initial_pool_size=50)
        docs, doc_ids = await retriever(search_query) # Returns (List[str], List[str])
        
        # 4. Filter Redundant Docs
        filtered_docs = []
        filtered_ids = []
        
        for doc_text, pmid in zip(docs, doc_ids):
            if pmid not in seen_pmids:
                filtered_docs.append(doc_text)
                filtered_ids.append(pmid)
                
        # Take top 10 unique
        final_docs = filtered_docs[:10]
        final_ids = filtered_ids[:10]
        
        logger.info(f"Supplemental Retrieval: Found {len(docs)} docs, {len(final_docs)} unique.")
        
        return {
            "retry_count": retry_count + 1,
            "current_documents": final_docs,
            "current_doc_ids": final_ids,
            # Reset decision to prevent loops (though graph edge should handle this)
            "evidence_decision": "accept" 
        }
        
    except Exception as e:
        logger.error(f"Supplemental Retrieval Failed: {e}")
        return {
            "retry_count": retry_count + 1,
            "evidence_decision": "accept" # Fail open
        }
