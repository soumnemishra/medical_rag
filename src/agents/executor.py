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
'''insted of writing 
step1()
step2()
step3()
 u define 
 Nodes (functions)

Edges (who runs next)

Shared state (memory)'''
from langgraph.graph import StateGraph, START, END #the is the state machine driven by multiagent system 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.state.state import PlanExecState, QAAnswerState
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
    
    # 2. Define Nodes
    
    async def task_definer_node(state: PlanExecState) -> Dict[str, Any]:
        return await step_definer.define_task(state)

    async def execution_node(state: PlanExecState) -> Dict[str, Any]:
        try:
            # Bug Fix #6: Check if step_question list is not empty before accessing
            step_questions = state.get("step_question", [])
            if not step_questions:
                logger.error("No step_question found in state")
                fallback = {"analysis": "Error: No step question", "answer": "No task to execute.", "success": "No", "rating": 0}
                return {"step_output": [fallback], "step_docs_ids": [], "step_notes": [[]]}
            
            current_task_info = step_questions[-1]
            task_type = current_task_info.get("type", "question-answering")
            query = current_task_info.get("task", "")
            
            logger.info(f"Executing Task: {task_type} - {query}")
            
            if task_type == "aggregate":
                # Aggregate logic with parser
                chain = aggregate_prompt | llm | aggregate_parser
                
                response = await chain.ainvoke({
                    "question": query
                })
                
                step_doc_ids = []
                step_notes = [response.get("analysis", "")]  # Already a list of strings
                answer_state = response
                
            else:
                # RAG logic with Extractor
                # 1. Retrieve
                # contexts is List[str]
                clinical_guidelines_needed = state.get("needs_guidelines", False)
                contexts, doc_ids = await retriever_tool(query, requires_guidelines=clinical_guidelines_needed)
                
                # 2. Extract
                # Fast-Path: Skip extraction for simple informational queries to reduce latency
                is_simple_informational = state.get("intent") == "informational" and len(state.get("plan", [])) <= 1
                
                if contexts:
                    if is_simple_informational:
                        logger.info("Fast-Path Execution: Skipping Extractor for informational query.")
                        # Join snippets directly with simple truncation
                        notes_text = "\n\n".join([f"• {c[:500]}..." for c in contexts[:3]])
                    else:
                        logger.info("Extracting relevant notes from retrieved documents...")
                        extraction_result = await extractor_agent.extract(query, contexts)
                        # Handle dict output if newer ExtractorAgent is used
                        if isinstance(extraction_result, dict):
                            notes_text = extraction_result.get("notes", "")
                        else:
                            notes_text = str(extraction_result)
                else:
                    notes_text = "No documents found."
                
                # 3. QA
                # Pass extracted notes as the 'documents' context for RagAgent
                # We pass the original doc_ids for citation tracking
                
                logger.info(f"Extractor Notes: {notes_text[:200]}...")  # Log first 200 chars
                
                rag_result = await rag_agent.query({
                    "question": query, 
                    "documents": [notes_text],
                    "doc_ids": doc_ids 
                })
                
                answer_state = rag_result["final_raw_answer"]
                logger.info(f"QA Answer: {answer_state.get('answer', 'No answer')[:200]}...")

                step_doc_ids = doc_ids
                # Bug Fix #3: step_notes should be List[str], not a single string
                step_notes = [notes_text] if notes_text else []
            
            # Cast to QAAnswerState to be safe (TypedDict)
            # Assuming models return correct keys or we are lenient
            
            return {
                "step_output": [answer_state],
                "step_docs_ids": [step_doc_ids] if step_doc_ids else [],
                # Bug Fix #3: step_notes is List[List[str]], so wrap step_notes (List[str]) in another list
                "step_notes": [step_notes] if step_notes else [[]]
            }
            
        except Exception as e:
            logger.error(f"Execution node failed: {e}")
            fallback = {"analysis": f"Error: {e}", "answer": "Error executing step.", "success": "No", "rating": 0}
            # Bug Fix #2: Return consistent structure - step_notes should be List[List[str]]
            return {"step_output": [fallback], "step_docs_ids": [], "step_notes": [[]]}

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
