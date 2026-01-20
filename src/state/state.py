from typing import List, Annotated, Optional, TypedDict
import operator
from pydantic import BaseModel, Field

# =============================================================================
# Data Models (Pydantic)
# =============================================================================

class QAAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    success: str = Field(description="Binary output (Yes or No), indicate if you can answer or not")
    rating: int = Field(default=None, description="Confidence rating 0-10. More evidence = higher score")

class PlanFormat(BaseModel):
    analysis: str = Field(description="Your analysis. Think step-by-step")
    step: List[str] = Field(description="Different steps to follow, should be in sorted order")

class StepTaskFormat(BaseModel):
    type: str = Field(description="Type of task: 'aggregate' or 'question-answering'")
    task: str = Field(description="The detailed task to do in this step")

class PlanSummaryFormat(BaseModel): 
    output: str = Field(description="Your output summary, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confidence score")

# =============================================================================
# State Definitions (TypedDict)
# =============================================================================

class QAAnswerState(TypedDict):
    analysis: str
    answer: str
    success: str
    rating: int

class StepTaskState(TypedDict):
    type: str
    task: str

class PlanSummaryState(TypedDict):
    output: str
    answer: str
    score: int

class RagState(TypedDict):
    """
    State for RAG execution on a single query.
    """
    question: str
    documents: List[str] # Optional: Pre-fetched docs/notes
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: QAAnswerFormat

class PlanExecState(TypedDict):
    """
    State for the nested Plan Executor Graph.
    Manages the execution of a single plan (sequence of steps).
    """
    original_question: str
    plan: List[str]  # The plan to follow
    step_question: Annotated[List[StepTaskState], operator.add]  # List of sub-tasks
    step_output: Annotated[List[QAAnswerState], operator.add]    # Output of each sub-task
    step_docs_ids: Annotated[List[List[str]], operator.add]      # Retrieved Doc IDs per step
    step_notes: Annotated[List[List[str]], operator.add]         # Notes per step
    plan_summary: PlanSummaryState                               # Final summary of this plan
    stop: bool = False

class GraphState(TypedDict):
    """
    Main Global State for the MA-RAG process.
    Manages the high-level loop of Planning -> Execution -> Comparison.
    """
    original_question: str
    plan: List[str] 
    past_exp: Annotated[List[PlanExecState], operator.add]       # History of past plan executions
    final_answer: str
