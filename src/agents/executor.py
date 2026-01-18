import logging
import operator
from typing import Dict, Any, List

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.state.state import PlanExecState, QAAnswerFormat, QAAnswerState
from src.prompts.templates import (
    AGGREGATE_SYSTEM_PROMPT,
    AGGREGATE_HUMAN_PROMPT
)
from src.core.registry import ModelRegistry
from src.agents.step_definer import StepDefinerAgent
from src.agents.rag import RagAgent

logger = logging.getLogger(__name__)

def build_executor_graph():
    """
    Builds the nested graph for executing a plan.
    Loop: StepDefiner -> Execution -> StepDefiner ... -> End
    """
    
    # 1. Initialize Agents
    step_definer = StepDefinerAgent()
    rag_agent = RagAgent()
    
    llm = ModelRegistry.get_llm(temperature=0.0)
    
    # Aggregation parser
    aggregate_parser = PydanticOutputParser(pydantic_object=QAAnswerFormat)
    
    aggregate_prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATE_SYSTEM_PROMPT + "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"),
        ("human", AGGREGATE_HUMAN_PROMPT)
    ])

    # 2. Define Nodes
    
    def task_definer_node(state: PlanExecState) -> Dict[str, Any]:
        return step_definer.define_task(state)

    def execution_node(state: PlanExecState) -> Dict[str, Any]:
        try:
            current_task_info = state["step_question"][-1]
            task_type = current_task_info["type"]
            query = current_task_info["task"]
            
            logger.info(f"Executing Task: {task_type} - {query}")
            
            if task_type == "aggregate":
                # Aggregate logic with parser
                chain = aggregate_prompt | llm | aggregate_parser
                
                response = chain.invoke({
                    "question": query,
                    "format_instructions": aggregate_parser.get_format_instructions()
                })
                
                step_doc_ids = []
                step_notes = [response.analysis]
                answer_state = response.model_dump()
                
            else:
                # RAG logic
                rag_result = rag_agent.query({"question": query})
                
                answer_state = rag_result["final_raw_answer"]
                step_doc_ids = rag_result["doc_ids"]
                step_notes = rag_result["notes"]
            
            return {
                "step_output": [answer_state],
                "step_docs_ids": [step_doc_ids] if step_doc_ids else [],
                "step_notes": [step_notes] if step_notes else []
            }
            
        except Exception as e:
            logger.error(f"Execution node failed: {e}")
            fallback = QAAnswerState(analysis=f"Error: {e}", answer="Error executing step.", success="No", rating=0)
            return {"step_output": [fallback], "step_docs_ids": [], "step_notes": []}

    # 3. Define Conditional Logic
    
    def check_stop(state: PlanExecState):
        # Important: State values might be None if initialized poorly, safely get
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
