'''this file is the engine of our multiagent system 
This file is the ENGINE of your multi-agent system.

Planner → creates a plan (elsewhere)

Executor → runs that plan step by step ✅

Each step:

is defined

executed (RAG / aggregate)

stored in shared state

loop continues until stop=True

# In MA-RAG terms
This file is your Plan Executor Loop.
If Planner is the “brain”, this is the “muscle'''



#how logging been different from python 
''' print cannot separate erros vs info
cannot turned off
is bad in production

where as log allows 
log levels (info, warning, error)

file logging

filtering '''
import logging #logging is the proffesional way to print what program is doing 

# operator is a module that contains functions corresponding to Python operators.
# It is used to perform operations on variables.
import operator
from typing import Dict, Any, List 

from langgraph.graph import StateGraph, START, END #the is the state machine driven by multiagent system 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.state.state import PlanExecState, QAAnswerFormat, QAAnswerState
from src.prompts.templates import (
    AGGREGATE_SYSTEM_PROMPT,
    AGGREGATE_HUMAN_PROMPT
)
from src.core.registry import ModelRegistry
# step definer agent descides what step we are going to execute
from src.agents.step_definer import StepDefinerAgent
from src.agents.rag import RagAgent#rag agent answers the question using filtered noise 
from src.agents.extractor import ExtractorAgent # Extractor → removes noise
from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)

def build_executor_graph():
    """
    Builds the nested graph for executing a plan.
    Loop: StepDefiner -> Execution -> StepDefiner ... -> End
    """
    
    # 1. Initialize Agents
    step_definer = StepDefinerAgent()
    rag_agent = RagAgent()
    extractor_agent = ExtractorAgent()
    retriever_tool = RetrieverTool()
    
    llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
    
    # Aggregation parser
    aggregate_parser = JsonOutputParser()
    
    aggregate_prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATE_SYSTEM_PROMPT),
        ("human", AGGREGATE_HUMAN_PROMPT)
    ])

    # 2. Define Nodes
    
    def task_definer_node(state: PlanExecState) -> Dict[str, Any]:
        return step_definer.define_task(state)

    def execution_node(state: PlanExecState) -> Dict[str, Any]:
        try:
            current_task_info = state["step_question"][-1]
            task_type = current_task_info.get("type", "question-answering")
            query = current_task_info.get("task", "")
            
            logger.info(f"Executing Task: {task_type} - {query}")
            
            if task_type == "aggregate":
                # Aggregate logic with parser
                chain = aggregate_prompt | llm | aggregate_parser
                
                response = chain.invoke({
                    "question": query
                })
                
                step_doc_ids = []
                step_notes = [response.get("analysis", "")]
                answer_state = response
                
            else:
                # RAG logic with Extractor
                # 1. Retrieve
                # contexts is List[str]
                contexts, doc_ids = retriever_tool(query)
                
                # 2. Extract
                # Filter noise from contexts using ExtractorAgent
                if contexts:
                    logger.info("Extracting relevant notes from retrieved documents...")
                    extraction_result = extractor_agent.extract(query, contexts)
                    # Handle dict output if newer ExtractorAgent is used
                    if isinstance(extraction_result, dict):
                        notes_text = extraction_result.get("notes", "")
                    else:
                        notes_text = str(extraction_result)
                    # notes_text is a single string summary/notes
                else:
                    notes_text = "No documents found."
                
                # 3. QA
                # Pass extracted notes as the 'documents' context for RagAgent
                # We pass the original doc_ids for citation tracking
                
                logger.info(f"Extractor Notes: {notes_text[:200]}...")  # Log first 200 chars
                
                rag_result = rag_agent.query({
                    "question": query, 
                    "documents": [notes_text],
                    "doc_ids": doc_ids 
                })
                
                answer_state = rag_result["final_raw_answer"]
                logger.info(f"QA Answer: {answer_state.get('answer', 'No answer')[:200]}...")

                step_doc_ids = doc_ids
                step_notes = notes_text
            
            # Cast to QAAnswerState to be safe (TypedDict)
            # Assuming models return correct keys or we are lenient
            
            return {
                "step_output": [answer_state],
                "step_docs_ids": [step_doc_ids] if step_doc_ids else [],
                "step_notes": [[step_notes]] if step_notes else [] 
                # Note: step_notes in PlanExecState is List[List[str]]
                # So we wrap our single note string in a list, and then that goes into the state list
            }
            
        except Exception as e:
            logger.error(f"Execution node failed: {e}")
            fallback = {"analysis": f"Error: {e}", "answer": "Error executing step.", "success": "No", "rating": 0}
            return {"step_output": [fallback], "step_docs_ids": [], "step_notes": []}

    # 3. Define Conditional Logic
    
    def check_stop(state: PlanExecState):
        if state.get("stop"):
            return END
        return "execution_node"

    # 4. Build Graph
    workflow = StateGraph(PlanExecState)
    
    workflow.add_node("task_definer", task_definer_node)
    workflow.add_node("execution_node", execution_node)
    
    workflow.add_edge(START, "task_definer")
    
    workflow.add_conditional_edges(
        "task_definer",
        check_stop,
        {
            END: END,
            "execution_node": "execution_node"
        }
    )
    
    workflow.add_edge("execution_node", "task_definer")
    
    return workflow.compile()
