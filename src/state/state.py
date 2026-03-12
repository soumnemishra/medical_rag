

# the state.py is the nervous system of the multiagent rag system 
# in langraph the state is the shared system that every agent in the pipeline read and writes to 
# so if the state is messy then agent work flow is messed up 

'''think of the system like  the medical team sititng to solve a given questions
---> example 
-------------> can  "Can Metformin reduce cancer risk in diabetic patients?
my sysytem contains multiple agents 
all these agent share memory and the shared memory is called the state 
so the state.py define the strucutre of the ssytem memory 

########### so the state.py is the database schema of your system #####
# 
# if this is bad then system becomes chaotic '''



###############################  Discussing ################## import modules #############
'''list --> means to store doc_1 doc_2 doc_3
used for doc_ids
documents  plan steps notes etc 


############ Typed _dict____#########
# if python the dictionaries doesnot have a strcuture 
# so for example user{
# 'name': 'soumen', 'age': 25}
# if some one wanted to change the age like user['age']='twenty four"
# 
# now the age becomes a string the python doesnot complain the thing break silenty
# type dicts is like a structure form 
# 
# what inputs should be there or llm should expects 
# 
# why langraph use the type dict 
# 
# because the langraph requires the state object to be dictionaries '''


from typing import List, Annotated, Optional, TypedDict, Literal, Dict, Any
import operator
from pydantic import BaseModel, Field

# =============================================================================
# Data Models (Pydantic)
# return the json format matching this schema 
#field()  provide additional description 
# =============================================================================
#these classes are inherits from the base model that instructs / forces the llm to return data in a specific JSON format.
class QAAnswerFormat(BaseModel):
    #this forces the llm to provide the chain of thought so we can see what analysis the llm made 
    analysis: str = Field(description="Your thoughts, analysis about the question and context. Think step-by-step")
    answer: str = Field(description="The answer for the question") #this provdes the final answer 

    # a simple yes or no string . This is the grounding check can the mode find the answer in the pub med abstracts
    success: str = Field(description="Binary output (Yes or No), indicate if you can answer or not")
    # this is the self reported confidence score . It can use for the model caliberation 
    rating: int = Field(default=None, description="Confidence rating 0-10. More evidence = higher score")

class PlanFormat(BaseModel):
    analysis: str = Field(description="Your analysis. Think step-by-step")

    # this is the sequence of sub question that the plananr agent generates to solve a complex query 
    step: List[str] = Field(description="Different steps to follow, should be in sorted order")

class StepTaskFormat(BaseModel):
    # this is used by the step definer . It labels task as the aggregate (summarizing what we know 
    #or question answering (needs new pub med search))
    type: str = Field(description="Type of task: 'aggregate' or 'question-answering'")
    task: str = Field(description="The detailed task to do in this step")

class PlanSummaryFormat(BaseModel): 
    output: str = Field(description="Your output summary, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confidence score")

class ClinicalIntentFormat(BaseModel):
    #categories of the query 
    intent: str = Field(description="Primary intent: 'informational', 'diagnostic', 'therapeutic', 'mechanism'")
    risk_level: str = Field(description="Risk level: 'low', 'medium', 'high'")
    requires_disclaimer: bool = Field(description="Whether a disclaimer is mandatory")
    needs_guidelines: bool = Field(description="Whether clinical guidelines are required")


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

# class ClinicalIntentState(TypedDict):
#     intent: str
#     risk_level: str
#     requires_disclaimer: bool
#     needs_guidelines: bool

class ClinicalIntentState(TypedDict):
    intent: str
    risk_level: str
    requires_disclaimer: bool
    needs_guidelines: bool


class RouterOutput(TypedDict):
    """
    Output schema for the RouterAgent.
    Defines the execution mode and resource contracts.
    """
    execution_mode: Literal[
        "direct_qa",        # Simple, single-hop, binary (Fastest)
        "disambiguation",   # Ambiguous but single-hop (Medium)
        "multihop"          # Complex, multi-step reasoning (Slowest)
    ]
    
    # Capability Flags
    requires_planning: bool
    requires_extraction: bool
    requires_evidence_grading: bool # Declarative (v1)

    # Policy
    answer_policy: Dict[str, Any]
    
    # Budget (Declarative)
    execution_budget: Optional[Dict[str, int]]

###########  the plan state and rag state are the two nested states 

#the rag stste is the nested state it is the temporary work space when one agent 
# looks for a specific looking for an answer to a single sub-question, 
# it uses this state, then throws it away once the result is moved to the GraphState.
class RagState(TypedDict):
    """
    State for RAG execution on a single query.
    """
    question: str
    documents: List[str] # Optional: Pre-fetched docs/notes
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: QAAnswerState  # Changed from Pydantic to TypedDict
    intent: str
    risk_level: str
    safety_flags: List[str]

# the plan state is used when the system is breaking big question into the small steps 
#it track docs id(pmids) so that at the end it can have we can have a full biography 
class PlanExecState(TypedDict):
    """
    State for the nested Plan Executor Graph.
    Manages the execution of a single plan (sequence of steps).
    """
    original_question: str
    plan: List[str]  # The plan to follow
    step_question: Annotated[List[StepTaskState], operator.add]  # List of sub-tasks
    #this mean dont overwrite old results append new ones 
    step_output: Annotated[List[QAAnswerState], operator.add]    # Output of each sub-task
    step_docs_ids: Annotated[List[List[str]], operator.add]      # Retrieved Doc IDs per step
    step_notes: Annotated[List[List[str]], operator.add]         # Notes per step
    plan_summary: PlanSummaryState                               # Final summary of this plan
    stop: bool  # Note: Default values not supported in TypedDict - must be set explicitly
    
    # Clinical Context passed down from GraphState
    intent: str
    risk_level: str
    needs_guidelines: bool
    requires_disclaimer: bool

class EvidencePolarity(TypedDict):
    polarity: Literal["support", "refute", "mixed", "insufficient"]
    confidence: float

# this is the global state ##################################################################
# this is the main state/ master file  of the system that remains open through out the process
class GraphState(TypedDict):
    """
    Main Global State for the MA-RAG process.
    Manages the high-level loop of Planning -> Execution -> Comparison.
    """
    original_question: str
    plan: List[str] 
    #operator.add is the memory list instead of overwriting the old thoughts it keep history 
    #his is vital for Agentic Reasoning—the system can look back and say, "Plan A didn't work, let's try Plan B."
    past_exp: Annotated[List[PlanExecState], operator.add]       # History of past plan executions
    final_answer: str
    intent: str
    risk_level: str
    safety_flags: List[str]
    requires_disclaimer: bool
    needs_guidelines: bool
    #this descides the speed of the system whether the question goes to the direct qa or complex qa
    router_output: RouterOutput # Output from RouterAgent
    evaluation_mode: bool # If True, SafetyCritic preserves unsafe answers (for benchmarking)
    #addition for a medical paper. It tracks if the evidence 
    # "supports" or "refutes" the query. Medical research is rarely 100% "yes," 
    # so having mixed or insufficient as options adds high clinical value.
    evidence_polarity: EvidencePolarity # [NEW] Directional polarity of evidence
    evidence_decision: Literal["accept", "reretrieve_diverse", "reretrieve_counter"] # [NEW] Decision on evidence quality
    retry_count: int # [NEW] Number of retries attempted
    current_documents: List[str] # [NEW] Context for re-retrieval injection
    current_doc_ids: List[str] # [NEW] PMIDs for re-retrieval injection



###########################  what the difference between the type dict and pydantic #########
''' pdantic models comtmrolls the llm ouput format 
type dict controlls system memory structure 

so one is the communication between the llm and another
 one is communication between agents 


pydantic helps us to validate the structure data it validates the data at run time 

so why pydantic helpful
because llm outputs are messy text not a structure text 
we force llm to produce output with a structure json


typr dict cannot be used to validate the system out put 


############# Type dict ###############
# defines the structure of the dictionaries used in the system
# type dict help the ide and devloper to understand the strcutre of the dictionaries
# 

###### why both used in this pydantic and the type dict #############
Stage 1 : llm produce the structure output 
example plananr produces a plan 
pydantic ensure the structure 

stage-2 : system stores results in the state : 
the result is stored in the system memory 

type dict define this meory structure 


#why not to use only pydantic or typ dic 

1) pydantic is too slow for system state 
--> langraph updates state many times per second 

if every update used the pydantic validation then system becomes slow 
type dict is a light weight 

############# why not only use the typedict #############
type dict cannot validate the llm output 

it is the type hint 
think of like 
pydantic is the quality ispection department 
and type dict is the warehouse storage system 

LLM OUTPUT
   ↓
Pydantic Models
   ↓
Validated JSON
   ↓
Converted to Dict
   ↓
Stored in TypedDict States
   ↓
LangGraph Agents read/write

pydantic is the run time validator where as type dict is the type hint 

the real power of the system is not pydantic or type dict 
it is this idea : 
shared=memory  every agetns reads and write that turns the system from simple pipeline to 
collaborative reasoning system 

'''