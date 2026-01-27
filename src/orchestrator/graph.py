import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from src.state.state import GraphState, PlanExecState
from src.agents.planner import planner_node
from src.agents.executor import build_executor_graph
from src.agents.clinical_intent import clinical_intent_node
from src.agents.safety_critic import safety_critic_node

logger = logging.getLogger(__name__)

def build_graph():
    """
    Builds the main MA-RAG Orchestration Graph.
    
    Structure:
    START -> Clinical Intent -> Planner -> Executor -> Safety Critic -> END
    
    The Executor is a nested graph that runs the iterative planning/execution loop.
    """
    
    # Pre-build the nested executor graph
    executor_graph = build_executor_graph()
    
    async def executor_wrapper_node(state: GraphState) -> Dict[str, Any]:
        """
        Wraps the nested executor graph.
        Prepares input state for executor and captures output.
        """
        plan = state["plan"]
        original_question = state["original_question"]
        
        # Propagate clinical context to executor
        intent = state.get("intent", "informational")
        risk_level = state.get("risk_level", "low")
        needs_guidelines = state.get("needs_guidelines", False)
        requires_disclaimer = state.get("requires_disclaimer", False)
        
        # Initialize sub-graph state
        input_state: PlanExecState = {
            "original_question": original_question,
            "plan": plan,
            "step_question": [],
            "step_output": [],
            "step_docs_ids": [],
            "step_notes": [],
            # Initialize with empty summary
            "plan_summary": {"output": "", "answer": "", "score": 0},
            "stop": False,
            # Clinical Context
            "intent": intent,
            "risk_level": risk_level,
            "needs_guidelines": needs_guidelines,
            "requires_disclaimer": requires_disclaimer
        }
        
        logger.info("Invoking Executor Graph...")
        # Use ainvoke for async graph execution
        full_output = await executor_graph.ainvoke(input_state)
        
        # Capture the result
        # The executor returns the full state state. We want to archive it in 'past_exp'
        # Bug Fix #7: Safe access for plan_summary with fallback
        plan_summary = full_output.get("plan_summary", {})
        final_answer = plan_summary.get("answer", "No answer generated")
        
        # Bug Fix #8: Append cited PMIDs as references if not already in answer
        cited_pmids = plan_summary.get("cited_pmids", [])
        if cited_pmids and "PMID" not in final_answer and "References" not in final_answer:
            refs_section = "\n\n**References:**\n" + "\n".join([f"- PMID:{pmid}" for pmid in cited_pmids[:10]])
            final_answer += refs_section
        
        return {
            "past_exp": [full_output],
            "final_answer": final_answer
        }

    # Build Main Graph
    workflow = StateGraph(GraphState)
    
    workflow.add_node("clinical_intent", clinical_intent_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_wrapper_node)
    workflow.add_node("safety_critic", safety_critic_node)
    
    # Define Edge Flow
    workflow.add_edge(START, "clinical_intent")
    workflow.add_edge("clinical_intent", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "safety_critic")
    workflow.add_edge("safety_critic", END)
    
    return workflow.compile()
