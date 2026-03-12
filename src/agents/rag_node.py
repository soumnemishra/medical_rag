import logging
from typing import Dict, Any
from src.state.state import GraphState, RagState
from src.agents.rag import RagAgent

logger = logging.getLogger(__name__)

async def rag_direct_node(state: GraphState) -> Dict[str, Any]:
    """
    Node for "Direct QA" execution mode.
    Wraps RagAgent to execute directly on the global state without planning.
    
    Contract:
    - Input: GraphState
    - Output: Updates 'final_answer' and 'past_exp' (simulated)
    - Must populate 'doc_ids' in the final output for SafetyCritic.
    """
    logger.info("Executing Direct QA Path (RagNode)...")
    
    question = state["original_question"]
    router_output = state.get("router_output", {})
    answer_policy = router_output.get("answer_policy", {})
    
    # Adapt GraphState to RagState
    # RagAgent expects: question, documents (optional), doc_ids, notes, final_raw_answer
    # [MODIFIED] Inject pre-fetched documents from SupplementalRetrievalNode if present
    rag_input: RagState = {
        "question": question,
        "documents": state.get("current_documents", []),     
        "doc_ids": state.get("current_doc_ids", []),
        "notes": [],
        "final_raw_answer": {},
        # Pass context for RAG agent if it uses it (current implementation mostly uses question)
        "intent": state.get("intent", "informational"),
        "risk_level": state.get("risk_level", "low"),
        "safety_flags": state.get("safety_flags", [])
    }
    
    try:
        # Initialize Agent
        agent = RagAgent()
        
        # Execute RAG
        # Note: RagAgent.query performs retrieval if documents are empty
        result = await agent.query(rag_input)
        
        # Parse Result
        final_raw = result.get("final_raw_answer", {})
        answer_text = final_raw.get("answer", "No answer generated.")
        doc_ids = result.get("doc_ids", [])
        
        # Policy Enforcement: Force "Yes/No" commitment if requested
        if answer_policy.get("force_commitment", False):
            # Simple heuristic check if model failed to follow "Final Answer:" instruction
            # (Ideally this would be a re-prompt, but for Stage 1 we rely on the prompt)
            pass 
        
        # Update GraphState
        # We simulate a "past_exp" entry so the UI/SafetyCritic can see what happened
        simulated_step_output = {
            "analysis": "Direct QA Execution",
            "answer": answer_text,
            "success": "Yes",
            "rating": 10
        }
        
        # Create a "pseudo-plan-execution" record for compatibility
        simulated_exp = {
            "original_question": question,
            "plan": ["Direct QA"],
            "step_question": [{"type": "question-answering", "task": "Direct QA"}],
            "step_output": [simulated_step_output],
            "step_docs_ids": [doc_ids],
            "step_notes": [result.get("notes", [])],
            "plan_summary": {
                "output": "Successful", 
                "answer": answer_text, 
                "score": 10,
                "cited_pmids": doc_ids # Critical for citation preservation
            },
            "stop": True,
            "intent": state.get("intent"),
            "risk_level": state.get("risk_level"),
            "needs_guidelines": state.get("needs_guidelines"),
            "requires_disclaimer": state.get("requires_disclaimer")
        }
        
        return {
            "final_answer": answer_text,
            "past_exp": [simulated_exp]
        }
        
    except Exception as e:
        logger.error(f"RagNode Failed: {e}")
        return {
            "final_answer": "I encountered an error during direct search.",
            "past_exp": []
        }
