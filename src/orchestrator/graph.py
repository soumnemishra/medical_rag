

# grapph.py is the central nervous system of the entire code base and of the MA rag architecture 
'''
state :what work is currntly doing  (the shared data moving between the enitre process)
node : which thing is the doing the work 
edge : the path 
state graph : the entire asembly line  that connects them  '''
 
import logging  # this set ups a tracking system and print the logs in the console 
from typing import Dict, Any # this is the python type hinting ,
#it just help u to know what kind of data is moving around
##################### State Graph ###################  
'''the factory assembly line 
---> it define how your product moves from one station to the next station 
factory assembly line where a product moves from one station to 
another until it becomes a finished product'''
# the start  is the entry door for the user queries and the end is the exit door when the final anser is deliverd 
from langgraph.graph import StateGraph, START, END  # these are the building blocks of the langraph 
# the graph state hold the shsared memory 
# PlanExecState is a specialized memory specifically for the multi-step planner.
#plan state is the temporary memory used by the executor node 
from src.state.state import GraphState, PlanExecState
from src.agents.planner import planner_node
from src.agents.executor import build_executor_graph
from src.agents.clinical_intent import clinical_intent_node
from src.agents.safety_critic import safety_critic_node
from src.agents.router_agent import router_node
from src.agents.rag_node import rag_direct_node
from src.agents.evidence_polarity_agent import evidence_polarity_node
from src.agents.decision_alignment import decision_alignment_node
from src.agents.evidence_decision_agent import evidence_decision_node
from src.agents.supplemental_retrieval_node import supplemental_retrieval_node

logger = logging.getLogger(__name__)

def build_graph():
    """
    Builds the main MA-RAG Orchestration Graph.
    
    Structure:
    START -> Clinical Intent -> Router
                                  |-> Direct QA -> RagNode -> Safety Critic -> END
                                  |-> Complex   -> Planner -> Executor -> Safety Critic -> END
    
    The Executor is a nested graph that runs the iterative planning/execution loop.
    """
    
    # Pre-build the nested executor graph
    '''before u build the main graph u prebuild a executor graph .
    --> Because the complex medical queries requires multiple steps 
    
    for example retrieve drug a then retreive drug b and then comparing 
    this graph handles iteratively the task without clutering the main graph '''
    executor_graph = build_executor_graph()  # this is where the executor graph is created 
    # it is responsible for executing the task / plan step by step 
    # this fucntion runs asynchronously mean the system doesnot block while waiting for
    # executor graph 
    '''think of like u give the order to the waiter --> cook --> waiter takes order--> while
    food prep going on 
    
    graphsATE IS THE shared memory from the main graph 
    
    then fucntions returns a dictiornary '''
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
        
        # Initialize sub-graph state think like giving the executor the sheet to follow 
        input_state: PlanExecState = {
            "original_question": original_question,
            "plan": plan,
            "step_question": [], #stores sub question 
            "step_output": [], # stores answers from each sub questions 
            "step_docs_ids": [], #stores doc id like pub med ids 
            "step_notes": [], # stores notes of the each question 
            # Initialize with empty summary give the summary of the annswer with confidence score 
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
        # this create a asnchronous graph state 
        #aiinvoke meas run the graph asynchronously 
        full_output = await executor_graph.ainvoke(input_state)
        
        # Capture the result
        # The executor returns the full state state. We want to archive it in 'past_exp'
        # Bug Fix #7: Safe access for plan_summary with fallback
        plan_summary = full_output.get("plan_summary", {})
        # we use the .get instead of the [] because it doesnot crash if the key doesnot exsit 
        final_answer = plan_summary.get("answer", "No answer generated")
        
        # Bug Fix #8: Append cited PMIDs as references if not already in answer
        #this check are thr citation present and if not present then cite it 
        cited_pmids = plan_summary.get("cited_pmids", [])
        if cited_pmids and "PMID" not in final_answer and "References" not in final_answer:
            #this create the refference section  # and inlcudes 10 refference max to ignore clutter 
            refs_section = "\n\n**References:**\n" + "\n".join([f"- PMID:{pmid}" for pmid in cited_pmids[:10]])
            #now the final answer becomes the final answer with the refferences 
            final_answer += refs_section
        
        return {
            # this stores entire exhibitor results full out and the final aanswer for easy debugging 
            "past_exp": [full_output],
            "final_answer": final_answer
        }

    # Build Main Graph
    workflow = StateGraph(GraphState)
    #  below are the worker nodes 
    workflow.add_node("clinical_intent", clinical_intent_node) # this classifies the question 
    workflow.add_node("router", router_node) #router node choose between simple and complex 
    workflow.add_node("planner", planner_node) #planner creates a multistep plan 
    workflow.add_node("executor", executor_wrapper_node) # executor execute the plan 
    workflow.add_node("rag_direct", rag_direct_node)     # direct rag node for simple questions 
    workflow.add_node("evidence_polarity", evidence_polarity_node) #evidence polarity node 
    workflow.add_node("evidence_decision", evidence_decision_node) #evidence decision node 
    workflow.add_node("supplemental_retrieval", supplemental_retrieval_node) #supplemental retrieval node 
    workflow.add_node("decision_alignment", decision_alignment_node) #decision alignment node 
    workflow.add_node("safety_critic", safety_critic_node) #safety critic node 
    
    # Conditional Routing Logic
    def route_decision(state: GraphState) -> str:
        router_output = state.get("router_output", {})
        mode = router_output.get("execution_mode", "multihop") # Default to safe multihop
        # there ar 3 routes direct-qa 
        if mode == "direct_qa":
            return "rag_direct"
        elif mode == "disambiguation":
            # Stage 1: Route to Planner (Option A) to ensure plan state exist
            # Future: specialized DisambiguationNode or constrained Planner
            return "planner"
        else:
            # "multihop" or fallback
            return "planner"

    # Define Edge Flow
    workflow.add_edge(START, "clinical_intent")
    workflow.add_edge("clinical_intent", "router")
    
    # Conditional Edges from Router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag_direct": "rag_direct",
            "planner": "planner"
        }
    )
    
    # Path A: Direct QA
    workflow.add_edge("rag_direct", "evidence_polarity")
    
    # Path B/C: Complex
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evidence_polarity")

    # Convergence
    workflow.add_edge("evidence_polarity", "evidence_decision")
    
    # Conditional Edge for Decision Logic
    def check_evidence_decision(state: GraphState) -> str:
        decision = state.get("evidence_decision", "accept")
        if decision == "accept":
            return "decision_alignment"
        else:
            return "supplemental_retrieval"
            
    workflow.add_conditional_edges(
        "evidence_decision",
        check_evidence_decision,
        {
            "decision_alignment": "decision_alignment",
            "supplemental_retrieval": "supplemental_retrieval"
        }
    )
    
    # Retry Loop
    workflow.add_edge("supplemental_retrieval", "rag_direct")
    
    workflow.add_edge("decision_alignment", "safety_critic")
    
    # End
    workflow.add_edge("safety_critic", END)
    
    return workflow.compile()
