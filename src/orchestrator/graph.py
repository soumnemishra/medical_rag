import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from src.state.state import GraphState, PlanExecState
from src.agents.planner import planner_node
from src.agents.executor import build_executor_graph

logger = logging.getLogger(__name__)

def build_graph():
    """
    Builds the main MA-RAG Orchestration Graph.
    
    Structure:
    START -> Planner -> Executor -> END
    
    The Executor is a nested graph that runs the iterative planning/execution loop.
    """
    
    # Pre-build the nested executor graph
    executor_graph = build_executor_graph()
    
    def executor_wrapper_node(state: GraphState) -> Dict[str, Any]:
        """
        Wraps the nested executor graph.
        Prepares input state for executor and captures output.
        """
        plan = state["plan"]
        original_question = state["original_question"]
        
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
            "stop": False
        }
        
        logger.info("Invoking Executor Graph...")
        full_output = executor_graph.invoke(input_state)
        
        # Capture the result
        # The executor returns the full state state. We want to archive it in 'past_exp'
        return {
            "past_exp": [full_output],
            "final_answer": full_output["plan_summary"]["answer"]
        }

    # Build Main Graph
    workflow = StateGraph(GraphState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_wrapper_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)
    
    return workflow.compile()
