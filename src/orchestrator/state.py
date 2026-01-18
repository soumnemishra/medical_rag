# FILE: src/orchestrator/state.py
"""
State definitions for the Agentic RAG system.

Implements LangGraph-compatible state with TC-RAG style memory stack
for backtracking and confidence-based decision making.

Example Usage:
    >>> from src.orchestrator.state import AgentState, AgentStep
    >>> state = AgentState(query="What is the treatment for melanoma?")
    >>> state.push("Searched for melanoma treatment")
    >>> state.should_backtrack()
    False
"""

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentStep(str, Enum):
    """Current step in the agentic RAG pipeline."""
    PLAN = "plan"
    RETRIEVE = "retrieve"
    EXTRACT = "extract"
    SYNTHESIZE = "synthesize"
    BACKTRACK = "backtrack"
    DONE = "done"


# =============================================================================
# Pydantic Models for Structured LLM Outputs
# =============================================================================

class PlanOutput(BaseModel):
    """Structured output from the Planner agent."""
    analysis: str = Field(description="Step-by-step analysis of the question")
    steps: List[str] = Field(description="Ordered list of sub-tasks to complete")
    pico_population: List[str] = Field(default_factory=list, description="PICO population terms")
    pico_intervention: List[str] = Field(default_factory=list, description="PICO intervention terms")
    pico_modifiers: List[str] = Field(default_factory=list, description="PICO modifiers (stage, age, etc.)")


class StepTask(BaseModel):
    """A single step task to execute."""
    task_type: str = Field(description="Type: 'search', 'aggregate', or 'synthesize'")
    task_query: str = Field(description="The specific query or task to execute")


class QAAnswer(BaseModel):
    """Structured answer output with confidence scoring."""
    analysis: str = Field(description="Step-by-step reasoning")
    answer: str = Field(description="The answer to the question")
    success: str = Field(description="Yes/No - whether the answer was found")
    confidence: int = Field(
        default=5, 
        ge=0, 
        le=10, 
        description="Confidence score 0-10"
    )
    sources: List[str] = Field(default_factory=list, description="Source PMIDs")


class PlanSummary(BaseModel):
    """Summary of plan execution with final answer."""
    output: str = Field(description="Execution status: Successful/Unsuccessful")
    answer: str = Field(description="Final answer to the original question")
    confidence: int = Field(default=5, ge=0, le=10, description="Overall confidence")
    reasoning: str = Field(default="", description="Summary of reasoning chain")


# =============================================================================
# LangGraph State Definitions
# =============================================================================

class RetrievedDoc(TypedDict):
    """A retrieved document from PubMed."""
    pmid: str
    title: str
    abstract: str
    year: Optional[int]
    relevance_score: float


class StepOutput(TypedDict):
    """Output from a single step execution."""
    step_name: str
    task: str
    answer: QAAnswer
    docs_used: List[str]


class PlanExecutorState(TypedDict):
    """State for the plan execution subgraph."""
    original_question: str
    plan: List[str]
    current_step_idx: int
    step_tasks: Annotated[List[StepTask], operator.add]
    step_outputs: Annotated[List[StepOutput], operator.add]
    retrieved_docs: Annotated[List[RetrievedDoc], operator.add]
    extracted_notes: Annotated[List[str], operator.add]
    stop: bool


class RAGState(TypedDict):
    """State for the RAG subgraph (Retrieve → Extract → Generate)."""
    question: str
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]
    final_answer: Optional[QAAnswer]


class GraphState(TypedDict):
    """
    Main graph state for the Agentic RAG system.
    
    Combines MA-RAG patterns with medical-specific fields.
    Uses Annotated types with operator.add for state accumulation.
    """
    # Core query
    original_question: str
    
    # Planning
    plan: List[str]
    pico_query: Optional[Dict[str, List[str]]]
    
    # Execution tracking
    current_step: AgentStep
    iteration_count: int
    max_iterations: int
    
    # Memory (TC-RAG style)
    memory_stack: List[str]
    past_attempts: Annotated[List[PlanExecutorState], operator.add]
    
    # Retrieval
    retrieved_docs: Annotated[List[RetrievedDoc], operator.add]
    extracted_notes: Annotated[List[str], operator.add]
    
    # Outputs
    step_outputs: Annotated[List[StepOutput], operator.add]
    final_answer: Optional[str]
    final_confidence: int
    sources: List[str]
    
    # Reasoning trace (for explainability)
    reasoning_trace: Annotated[List[Dict[str, Any]], operator.add]


# =============================================================================
# State Management Utilities
# =============================================================================

@dataclass
class AgentStateManager:
    """
    Manager for AgentState with TC-RAG style memory operations.
    
    Provides push/pop operations for backtracking and
    confidence-based decision making.
    """
    
    state: GraphState = field(default_factory=lambda: GraphState(
        original_question="",
        plan=[],
        pico_query=None,
        current_step=AgentStep.PLAN,
        iteration_count=0,
        max_iterations=3,
        memory_stack=[],
        past_attempts=[],
        retrieved_docs=[],
        extracted_notes=[],
        step_outputs=[],
        final_answer=None,
        final_confidence=0,
        sources=[],
        reasoning_trace=[],
    ))
    
    confidence_threshold: float = 0.6
    
    def push(self, item: str) -> None:
        """TC-RAG: Push to memory stack."""
        self.state["memory_stack"].append(item)
    
    def pop(self) -> Optional[str]:
        """TC-RAG: Pop from memory stack (backtrack)."""
        if self.state["memory_stack"]:
            return self.state["memory_stack"].pop()
        return None
    
    def should_backtrack(self) -> bool:
        """
        Check if we should backtrack based on confidence.
        
        Returns True if:
        - Confidence is below threshold
        - We haven't exceeded max iterations
        """
        confidence_ratio = self.state["final_confidence"] / 10.0
        return (
            confidence_ratio < self.confidence_threshold 
            and self.state["iteration_count"] < self.state["max_iterations"]
        )
    
    def increment_iteration(self) -> None:
        """Increment iteration count."""
        self.state["iteration_count"] += 1
    
    def add_reasoning_step(
        self, 
        phase: str, 
        thought: str, 
        details: Optional[str] = None
    ) -> None:
        """Add a step to the reasoning trace."""
        self.state["reasoning_trace"].append({
            "phase": phase,
            "thought": thought,
            "details": details,
            "iteration": self.state["iteration_count"],
        })
    
    def get_context_summary(self, max_items: int = 5) -> str:
        """Get a summary of current context for LLM prompts."""
        parts = []
        
        # Recent docs
        if self.state["retrieved_docs"]:
            docs = self.state["retrieved_docs"][-max_items:]
            parts.append("Recent documents:")
            for doc in docs:
                parts.append(f"  - PMID:{doc['pmid']}: {doc['title'][:80]}...")
        
        # Recent notes
        if self.state["extracted_notes"]:
            notes = self.state["extracted_notes"][-max_items:]
            parts.append("\nExtracted notes:")
            for note in notes:
                parts.append(f"  - {note[:100]}...")
        
        return "\n".join(parts) if parts else "No context yet."


def create_initial_state(question: str) -> GraphState:
    """Create an initial state for a new query."""
    return GraphState(
        original_question=question,
        plan=[],
        pico_query=None,
        current_step=AgentStep.PLAN,
        iteration_count=0,
        max_iterations=3,
        memory_stack=[],
        past_attempts=[],
        retrieved_docs=[],
        extracted_notes=[],
        step_outputs=[],
        final_answer=None,
        final_confidence=0,
        sources=[],
        reasoning_trace=[],
    )
