from typing import List, Annotated, Optional, TypedDict, Literal, Dict, Any
import operator
from pydantic import BaseModel, Field

# =============================================================================
# Pydantic Models — LLM output validation
# =============================================================================
# Pydantic: validates LLM JSON output at runtime. The LLM speaks messy text;
# Pydantic enforces that it fills out a structured form correctly.
#
# TypedDict: defines the structure of the shared state dictionaries that
# agents read/write. Lightweight — LangGraph updates state many times per
# query; Pydantic validation on every update would be too slow.
#
# Pipeline:
#   LLM text → Pydantic (validate) → dict → TypedDict state (store)
# =============================================================================

class QAAnswerFormat(BaseModel):
    analysis: str = Field(description="Chain-of-thought reasoning about the question and context")
    answer:   str = Field(description="The final answer")
    success:  str = Field(description="Yes or No — can the question be answered from the evidence?")
    rating:   int = Field(default=0, description="Confidence 0–10. More/better evidence = higher score")

class PlanStep(BaseModel):
    """Single step in an execution plan. id and depends_on enable parallel execution."""
    id:         int      = Field(description="Unique step identifier")
    question:   str      = Field(description="The sub-question for this step")
    depends_on: List[int] = Field(default_factory=list, description="IDs of steps that must complete first")
    step_type:  str      = Field(default="question-answering", description="question-answering | aggregate | simple")

class PlanFormat(BaseModel):
    """Full plan output from PlannerAgent."""
    plan:        List[PlanStep] = Field(description="Ordered list of plan steps")
    total_steps: int            = Field(description="Total number of steps")
    complexity:  str            = Field(description="simple | moderate | complex")

class StepTaskFormat(BaseModel):
    type: str = Field(description="aggregate or question-answering")
    task: str = Field(description="Detailed query for this step")

class PlanSummaryFormat(BaseModel):
    output: str = Field(description="Successful or Failed")
    answer: str = Field(description="Final synthesised answer with citations")
    score:  int = Field(description="Confidence score 0–10")

class ClinicalIntentFormat(BaseModel):
    intent:              str   = Field(description="informational | diagnostic | therapeutic | mechanism")
    risk_level:          str   = Field(description="low | medium | high")
    requires_disclaimer: bool  = Field(description="Mandatory for therapeutic/diagnostic queries")
    needs_guidelines:    bool  = Field(description="True if guidelines are required")
    confidence:          float = Field(ge=0.0, le=1.0, description="Classification certainty 0.0–1.0")
    reasoning:           str   = Field(description="One sentence explanation of the classification")

class EvidencePolarity(BaseModel):
    """
    Pydantic model — NOT TypedDict — because it is used as:
        JsonOutputParser(pydantic_object=EvidencePolarity)
    JsonOutputParser requires a Pydantic BaseModel subclass.
    A TypedDict here raises TypeError at agent initialisation.
    """
    polarity:   str   = Field(default="insufficient",
                              description="support | refute | mixed | insufficient")
    confidence: float = Field(default=0.0, description="0.0–1.0")
    reasoning:  str   = Field(default="", description="One sentence explanation")

    def to_dict(self) -> dict:
        return {"polarity": self.polarity, "confidence": self.confidence, "reasoning": self.reasoning}


# =============================================================================
# State TypedDicts — shared memory between agents
# =============================================================================

class QAAnswerState(TypedDict):
    analysis: str
    answer:   str
    success:  str
    rating:   int

class StepTaskState(TypedDict):
    type: str
    task: str

class PlanSummaryState(TypedDict):
    output: str
    answer: str
    score:  int

class ClinicalIntentState(TypedDict):
    intent:              str
    risk_level:          str
    requires_disclaimer: bool
    needs_guidelines:    bool
    confidence:          float
    reasoning:           str

class RouterOutput(TypedDict):
    execution_mode:            Literal["direct_qa", "disambiguation", "multihop"]
    requires_planning:         bool
    requires_extraction:       bool
    requires_evidence_grading: bool
    answer_policy:             Dict[str, Any]
    execution_budget:          Optional[Dict[str, int]]


class RagState(TypedDict):
    """Temporary state for a single RAG execution (one step or direct QA)."""
    question:         str
    documents:        List[str]
    doc_ids:          List[str]
    notes:            List[str]
    final_raw_answer: QAAnswerState
    intent:           str
    risk_level:       str
    safety_flags:     List[str]
    # evidence_polarity passed through so QA_HUMAN_PROMPT {evidence_polarity}
    # placeholder is filled — without this field RagAgent raises KeyError
    evidence_polarity: Dict[str, Any]


class PlanExecState(TypedDict):
    """
    State for the nested Plan Executor Graph (executor.py).
    Lives only during plan execution — discarded afterwards.
    """
    original_question:  str
    # List[Any] supports both old List[str] and new List[Dict] step formats
    plan:               List[Any]
    step_question:      Annotated[List[StepTaskState],  operator.add]
    step_output:        Annotated[List[QAAnswerState],  operator.add]
    step_docs_ids:      Annotated[List[List[str]],      operator.add]
    # FIXED: flat List[str] — was List[List[str]] which created ragged nesting.
    # executor.py returns {"step_notes": ["text"]} (List[str]);
    # operator.add appends correctly to Annotated[List[str], operator.add].
    step_notes:         Annotated[List[str],            operator.add]
    plan_summary:       PlanSummaryState
    stop:               bool
    # Clinical context passed down from GraphState
    intent:             str
    risk_level:         str
    needs_guidelines:   bool
    requires_disclaimer: bool


class GraphState(TypedDict):
    """
    Master shared state — the 'shared notebook' every agent reads and writes.

    Fields are grouped by which agent writes them:
      ClinicalIntentAgent → intent, risk_level, requires_disclaimer,
                            needs_guidelines, confidence, reasoning, safety_flags
      RouterAgent         → router_output
      PlannerAgent        → plan, plan_complexity, plan_error
      ExecutorWrapper     → past_exp, final_answer,
                            step_output, step_docs_ids, step_notes
      RagDirectNode       → final_answer, step_output, step_docs_ids, step_notes
      EvidencePolarityAgent → evidence_polarity
      EvidenceDecisionAgent → evidence_decision, retry_count
      SupplementalRetrieval → current_documents, current_doc_ids
      DecisionAlignment   → final_answer (may modify), safety_flags
      SafetyCritic        → final_answer (may refine), safety_flags
    """
    # ── Core query ──────────────────────────────────────────────────
    original_question: str

    # ── Plan ────────────────────────────────────────────────────────
    # List[Any]: supports both List[str] (legacy) and List[Dict] (updated planner)
    plan:             List[Any]
    plan_complexity:  str                  # "simple" | "moderate" | "complex"
    plan_error:       Optional[str]        # non-None when planner fails

    # ── Execution history ────────────────────────────────────────────
    # Accumulates across retries — PlannerAgent reads this for retry learning
    past_exp:         Annotated[List[Dict[str, Any]], operator.add]
    final_answer:     str

    # ── Current execution step data (surfaced from executor / rag_direct) ──
    # EvidencePolarityAgent and DecisionAlignmentAgent read these directly
    # from GraphState — they must exist here, not only inside past_exp
    step_output:      List[Dict[str, Any]]
    step_docs_ids:    List[Any]
    step_notes:       List[str]

    # ── Clinical intent classification ───────────────────────────────
    intent:              str
    risk_level:          str
    requires_disclaimer: bool
    needs_guidelines:    bool
    confidence:          float   # from ClinicalIntentAgent
    reasoning:           str     # from ClinicalIntentAgent

    # ── Routing ─────────────────────────────────────────────────────
    router_output:    RouterOutput

    # ── Evidence quality pipeline ────────────────────────────────────
    evidence_polarity:  Dict[str, Any]   # {"polarity": str, "confidence": float, "reasoning": str}
    evidence_decision:  Literal["accept", "reretrieve_diverse", "reretrieve_counter"]
    retry_count:        int

    # ── Supplemental retrieval context ───────────────────────────────
    current_documents:  List[str]
    current_doc_ids:    List[str]

    # ── Safety / audit ───────────────────────────────────────────────
    safety_flags:       List[str]

    # ── Evaluation ───────────────────────────────────────────────────
    # When True, SafetyCritic preserves unsafe answers for benchmark scoring
    evaluation_mode:    bool