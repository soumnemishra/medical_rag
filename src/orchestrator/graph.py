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


def build_full_agentic_graph() -> StateGraph:
    """
    Build the agentic RAG graph with all real agent implementations.
    
    This is the recommended way to build the graph for production use.
    Uses:
    - PlannerAgent: PICO decomposition + query planning
    - RetrieverAgent: PubMed search with PICO optimization
    - ExtractorAgent: Noise filtering from documents
    - SynthesizerAgent: Medical answer synthesis with confidence
    
    Returns:
        Compiled LangGraph StateGraph with all agents
    """
    # Import agent nodes (lazy to avoid circular deps)
    from src.agents.planner import planner_node_sync
    from src.agents.retriever import retriever_node_sync
    from src.agents.extractor import extractor_node_sync
    from src.agents.synthesizer import synthesizer_node_sync
    
    logger.info("Building full agentic graph with all agent implementations")
    
    return build_agentic_graph(
        planner=planner_node_sync,
        retriever=retriever_node_sync,
        extractor=extractor_node_sync,
        synthesizer=synthesizer_node_sync,
    )


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


# =============================================================================
# Enhanced Graph Builder (MA-RAG Style)
# =============================================================================

def build_enhanced_graph() -> StateGraph:
    """
    Build the enhanced agentic RAG graph with MA-RAG style step execution.
    
    This version uses:
    - Step-wise plan execution (loop through each step)
    - Task classification (search vs aggregate)
    - Per-step confidence tracking
    - Plan summarization for final answer
    
    Flow:
        PLAN → STEP_EXECUTOR (loop) → SUMMARIZE → DONE/BACKTRACK
    
    Returns:
        Compiled LangGraph StateGraph with enhanced features
    """
    from src.agents.planner import planner_node_sync
    from src.agents.plan_summarizer import summarizer_node_sync as plan_summarizer_sync
    from src.orchestrator.step_executor import build_step_executor, create_executor_initial_state
    
    logger.info("Building enhanced agentic graph with step executor")
    
    # Build step executor subgraph
    step_executor = build_step_executor()
    
    def plan_node_enhanced(state: GraphState) -> Dict[str, Any]:
        """Plan with the real planner agent."""
        return planner_node_sync(state)
    
    def execute_plan_node(state: GraphState) -> Dict[str, Any]:
        """Execute plan steps using step executor subgraph."""
        logger.info(f"[EXECUTE_PLAN] Running step executor for {len(state.get('plan', []))} steps")
        
        # Create step executor state
        exec_state = create_executor_initial_state(
            question=state["original_question"],
            plan=state.get("plan", []),
        )
        
        # Run step executor
        result = step_executor.invoke(exec_state)
        
        # Extract results
        step_outputs = result.get("step_outputs", [])
        retrieved_docs = result.get("retrieved_docs", [])
        extracted_notes = result.get("extracted_notes", [])
        sources = []
        
        # Collect sources from step outputs
        for step in step_outputs:
            if isinstance(step, dict):
                docs = step.get("docs_used", [])
                sources.extend(docs)
        
        logger.info(f"[EXECUTE_PLAN] Completed with {len(step_outputs)} step outputs")
        
        return {
            "step_outputs": step_outputs,
            "retrieved_docs": retrieved_docs,
            "extracted_notes": extracted_notes,
            "sources": list(set(sources)),  # Deduplicate
            "reasoning_trace": [{
                "phase": "EXECUTE_PLAN",
                "thought": f"Executed {len(step_outputs)} plan steps",
                "details": f"Retrieved {len(retrieved_docs)} docs, {len(extracted_notes)} notes",
                "iteration": state.get("iteration_count", 0),
            }],
        }
    
    def summarize_plan_node(state: GraphState) -> Dict[str, Any]:
        """Summarize plan execution with plan summarizer."""
        return plan_summarizer_sync(state)
    
    def route_after_plan_enhanced(state: GraphState) -> str:
        """Route after planning."""
        if state.get("plan") and len(state["plan"]) > 0:
            logger.info(f"Plan created with {len(state['plan'])} steps")
            return "execute_plan"
        else:
            logger.warning("No plan created, ending")
            return "done"
    
    def route_after_summary(state: GraphState) -> str:
        """Route after summarization - check confidence."""
        confidence = state.get("final_confidence", 0)
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 3)
        
        if confidence < 6 and iteration < max_iter:
            logger.info(f"Low confidence ({confidence}/10), backtracking")
            return "backtrack"
        
        logger.info(f"Confidence sufficient ({confidence}/10)")
        return "done"
    
    # Build enhanced graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("plan", plan_node_enhanced)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("summarize", summarize_plan_node)
    graph.add_node("backtrack", backtrack_node)
    graph.add_node("done", done_node)
    
    # Add edges
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan_enhanced)
    graph.add_edge("execute_plan", "summarize")
    graph.add_conditional_edges("summarize", route_after_summary)
    graph.add_conditional_edges("backtrack", route_after_backtrack)
    graph.add_edge("done", END)
    
    logger.info("Built enhanced agentic graph with step executor")
    
    return graph.compile()


def get_enhanced_graph_diagram() -> str:
    """Get Mermaid diagram for enhanced graph."""
    return """
flowchart TB
    START --> plan["PLAN (PICO)"]
    plan -->|has plan| execute["STEP EXECUTOR"]
    plan -->|no plan| done
    
    subgraph execute["STEP EXECUTOR (Loop)"]
        define["task_definer"] --> exec_step{"Task Type?"}
        exec_step -->|search| search["RAG Search"]
        exec_step -->|aggregate| agg["Aggregate"]
        search --> next["next step?"]
        agg --> next
        next -->|more steps| define
        next -->|done| out["outputs"]
    end
    
    execute --> summarize["PLAN SUMMARIZER"]
    summarize -->|high confidence| done
    summarize -->|low confidence| backtrack
    backtrack -->|retry| plan
    backtrack -->|max tries| done
    done --> END
"""

