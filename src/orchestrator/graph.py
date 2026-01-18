# FILE: src/orchestrator/graph.py
"""
LangGraph-based orchestrator for the Agentic RAG system.

Implements a state machine with Plan → Retrieve → Extract → Synthesize flow,
with confidence-based backtracking (TC-RAG pattern).

Example Usage:
    >>> from src.orchestrator.graph import build_agentic_graph, run_query
    >>> graph = build_agentic_graph()
    >>> result = await run_query(graph, "What is the treatment for stage II melanoma?")
    >>> print(result["final_answer"])
"""

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.orchestrator.state import (
    AgentStep,
    GraphState,
    create_initial_state,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Router Functions (Conditional Edges)
# =============================================================================

def route_after_plan(state: GraphState) -> str:
    """
    Route after planning step.
    
    Returns:
        Next node: "retrieve" if plan exists, "done" if empty plan
    """
    if state["plan"] and len(state["plan"]) > 0:
        logger.info(f"Plan created with {len(state['plan'])} steps")
        return "retrieve"
    else:
        logger.warning("No plan created, ending")
        return "done"


def route_after_synthesis(state: GraphState) -> str:
    """
    Route after synthesis step - check confidence for backtracking.
    
    Returns:
        "backtrack" if low confidence, "done" if confident
    """
    confidence = state.get("final_confidence", 0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    
    # Check if we should backtrack (TC-RAG pattern)
    if confidence < 6 and iteration < max_iter:
        logger.info(f"Low confidence ({confidence}/10), backtracking (iter {iteration})")
        return "backtrack"
    
    logger.info(f"Confidence sufficient ({confidence}/10), completing")
    return "done"


def route_after_backtrack(state: GraphState) -> str:
    """
    Route after backtracking - retry planning or give up.
    
    Returns:
        "plan" to retry, "done" to give up
    """
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    
    if iteration < max_iter:
        logger.info(f"Retrying plan (iteration {iteration + 1})")
        return "plan"
    
    logger.warning("Max iterations reached, completing with best effort")
    return "done"


# =============================================================================
# Placeholder Node Functions (To be replaced by agents)
# =============================================================================

def plan_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder for Planner Agent.
    
    Decomposes the question into PICO components and sub-tasks.
    Will be replaced by actual agent implementation.
    """
    logger.info(f"[PLAN] Processing: {state['original_question'][:50]}...")
    
    # Placeholder: In real implementation, this calls the planner agent
    return {
        "plan": ["Search for relevant literature", "Extract key findings", "Synthesize answer"],
        "pico_query": {
            "population": [],
            "intervention": [],
            "modifiers": [],
        },
        "current_step": AgentStep.RETRIEVE,
        "reasoning_trace": [{
            "phase": "PLAN",
            "thought": "Decomposed query into search plan",
            "iteration": state.get("iteration_count", 0),
        }],
    }


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder for Retriever Agent.
    
    Performs hybrid retrieval (PubMed + MedCPT).
    Will be replaced by actual agent implementation.
    """
    logger.info("[RETRIEVE] Searching PubMed...")
    
    # Placeholder: In real implementation, this calls the retriever agent
    return {
        "current_step": AgentStep.EXTRACT,
        "reasoning_trace": [{
            "phase": "RETRIEVE",
            "thought": "Retrieved documents from PubMed",
            "iteration": state.get("iteration_count", 0),
        }],
    }


def extract_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder for Extractor Agent.
    
    Filters noise from retrieved documents.
    Will be replaced by actual agent implementation.
    """
    logger.info("[EXTRACT] Filtering relevant content...")
    
    # Placeholder: In real implementation, this calls the extractor agent
    return {
        "current_step": AgentStep.SYNTHESIZE,
        "reasoning_trace": [{
            "phase": "EXTRACT",
            "thought": "Extracted relevant passages from documents",
            "iteration": state.get("iteration_count", 0),
        }],
    }


def synthesize_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder for Synthesizer Agent.
    
    Generates final answer in SOAP format.
    Will be replaced by actual agent implementation.
    """
    logger.info("[SYNTHESIZE] Generating answer...")
    
    # Placeholder: In real implementation, this calls the synthesizer agent
    iteration = state.get("iteration_count", 0)
    
    # Simulate increasing confidence with iterations
    confidence = min(7 + iteration * 2, 10)
    
    return {
        "final_answer": "Placeholder answer - agents not yet implemented",
        "final_confidence": confidence,
        "current_step": AgentStep.DONE,
        "reasoning_trace": [{
            "phase": "SYNTHESIZE",
            "thought": f"Generated answer with confidence {confidence}/10",
            "iteration": iteration,
        }],
    }


def backtrack_node(state: GraphState) -> Dict[str, Any]:
    """
    Handle backtracking - pop memory stack and adjust strategy.
    """
    logger.info("[BACKTRACK] Re-planning with new strategy...")
    
    iteration = state.get("iteration_count", 0) + 1
    memory_stack = list(state.get("memory_stack", []))
    
    # Pop last attempt and add to memory
    memory_stack.append(f"Attempt {iteration}: confidence too low, retrying")
    
    return {
        "iteration_count": iteration,
        "memory_stack": memory_stack,
        "current_step": AgentStep.PLAN,
        "reasoning_trace": [{
            "phase": "BACKTRACK",
            "thought": f"Low confidence, retrying (attempt {iteration})",
            "iteration": iteration,
        }],
    }


def done_node(state: GraphState) -> Dict[str, Any]:
    """
    Final node - prepare output.
    """
    logger.info("[DONE] Completing query processing")
    
    return {
        "current_step": AgentStep.DONE,
    }


# =============================================================================
# Graph Builder
# =============================================================================

def build_agentic_graph(
    planner: Optional[Callable] = None,
    retriever: Optional[Callable] = None,
    extractor: Optional[Callable] = None,
    synthesizer: Optional[Callable] = None,
) -> StateGraph:
    """
    Build the agentic RAG state graph.
    
    Args:
        planner: Optional custom planner node function
        retriever: Optional custom retriever node function
        extractor: Optional custom extractor node function
        synthesizer: Optional custom synthesizer node function
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph with state type
    graph = StateGraph(GraphState)
    
    # Add nodes (use custom or placeholder)
    graph.add_node("plan", planner or plan_node)
    graph.add_node("retrieve", retriever or retrieve_node)
    graph.add_node("extract", extractor or extract_node)
    graph.add_node("synthesize", synthesizer or synthesize_node)
    graph.add_node("backtrack", backtrack_node)
    graph.add_node("done", done_node)
    
    # Add edges
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan)
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "synthesize")
    graph.add_conditional_edges("synthesize", route_after_synthesis)
    graph.add_conditional_edges("backtrack", route_after_backtrack)
    graph.add_edge("done", END)
    
    logger.info("Built agentic RAG graph with 6 nodes")
    
    return graph.compile()


# =============================================================================
# Query Runner
# =============================================================================

async def run_query(
    graph: StateGraph,
    question: str,
    max_iterations: int = 3,
) -> GraphState:
    """
    Run a query through the agentic RAG graph.
    
    Args:
        graph: Compiled LangGraph
        question: User's medical question
        max_iterations: Maximum backtrack iterations
    
    Returns:
        Final GraphState with answer and reasoning trace
    """
    # Create initial state
    initial_state = create_initial_state(question)
    initial_state["max_iterations"] = max_iterations
    
    logger.info(f"Running query: {question[:50]}...")
    
    # Invoke graph
    result = await graph.ainvoke(initial_state)
    
    logger.info(
        f"Query complete. Confidence: {result.get('final_confidence', 0)}/10, "
        f"Iterations: {result.get('iteration_count', 0)}"
    )
    
    return result


def run_query_sync(
    graph: StateGraph,
    question: str,
    max_iterations: int = 3,
) -> GraphState:
    """
    Synchronous version of run_query for non-async contexts.
    """
    initial_state = create_initial_state(question)
    initial_state["max_iterations"] = max_iterations
    
    logger.info(f"Running query (sync): {question[:50]}...")
    
    result = graph.invoke(initial_state)
    
    return result


# =============================================================================
# Visualization Helper
# =============================================================================

def get_graph_diagram(graph: StateGraph) -> str:
    """
    Get Mermaid diagram of the graph for visualization.
    
    Returns:
        Mermaid diagram string
    """
    return """
flowchart TB
    START --> plan
    plan -->|has plan| retrieve
    plan -->|no plan| done
    retrieve --> extract
    extract --> synthesize
    synthesize -->|low confidence| backtrack
    synthesize -->|confident| done
    backtrack -->|retry| plan
    backtrack -->|max iterations| done
    done --> END
"""
