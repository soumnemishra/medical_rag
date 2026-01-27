from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import PlanExecState, StepTaskFormat, PlanSummaryFormat, PlanSummaryState, StepTaskState
from src.prompts.templates import (
    STEP_DEFINER_SYSTEM_PROMPT,
    STEP_DEFINER_HUMAN_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_HUMAN_PROMPT
)
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

class StepDefinerAgent:
    """
    Agent that decides the next task to execute in the plan or summarizes if done.
    """
    
    def __init__(self):
        self.llm = ModelRegistry.get_light_llm(temperature=0.0, json_mode=True)
        
        # Use simple JSON parser
        self.parser = JsonOutputParser()
        
        self.step_prompt = ChatPromptTemplate.from_messages([
            ("system", STEP_DEFINER_SYSTEM_PROMPT),
            ("human", STEP_DEFINER_HUMAN_PROMPT)
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", SUMMARY_HUMAN_PROMPT)
        ])
        
        # Initialize chains
        self.step_chain = self.step_prompt | self.llm | self.parser
        self.summary_chain = self.summary_prompt | self.llm | self.parser

    async def define_task(self, state: PlanExecState) -> Dict[str, Any]:
        """
        Decide next step or summarize.
        """
        try:
            plan = state.get("plan", [])
            step_output = state.get("step_output", [])
            step_question = state.get("step_question", [])
            step_docs_ids = state.get("step_docs_ids", [])  # Get doc IDs for citations
            original_question = state.get("original_question", "")
            
            should_stop = len(step_output) >= len(plan)
            # if step_output and step_output[-1].get("success", "").lower() == "no":
            #     should_stop = True
                
            if should_stop:
                return await self._summarize(original_question, plan, step_output, step_question, step_docs_ids)
            else:
                return await self._next_step(plan, step_output)
                
        except Exception as e:
            logger.error(f"Task definition failed: {e}")
            return {"stop": True, "plan_summary": {"output": "Failed", "answer": f"Error: {str(e)}", "score": 0}}

    async def _summarize(self, question: str, plan: List[str], step_outputs: List[Dict], step_questions: List[Dict], step_docs_ids: List[List[str]] = None) -> Dict[str, Any]:
        """Generate final summary with PMID citations."""
        logger.info("Plan execution complete. Summarizing...")
        
        memory = ""
        all_pmids = set()  # Collect all unique PMIDs
        
        for i, output in enumerate(step_outputs):
            task_desc = plan[i] if i < len(plan) else "Unknown Task"
            detailed_q = step_questions[i].get("task", "") if i < len(step_questions) else ""
            answer = output.get("answer", "")
            score = output.get("rating", 0)
            
            # Get PMIDs for this step
            doc_ids = step_docs_ids[i] if step_docs_ids and i < len(step_docs_ids) else []
            for pmid in doc_ids:
                all_pmids.add(pmid)
            pmids_str = ", ".join(doc_ids[:5]) if doc_ids else "None"  # Limit to 5 per step
            
            memory += f"Task: {task_desc}\nDetailed Query: {detailed_q}\nAnswer: {answer}\nConfidence: {score}\nSource PMIDs: {pmids_str}\n\n"
        
        # Debug: Log memory content
        logger.info(f"Summary Memory (first 500 chars): {memory[:500]}...")
        logger.info(f"Total unique PMIDs collected: {len(all_pmids)}")
            
        result = await self.summary_chain.ainvoke({
            "question": question,
            "plan": f"[{', '.join(plan)}]",
            "memory": memory
        })
        
        # Add collected PMIDs to result for citation in app
        if isinstance(result, dict):
            result["cited_pmids"] = list(all_pmids)
        
        # Ensure result matches PlanSummaryState
        # If parsing is loose, we trust the model produced valid dict
        return {
            "plan_summary": result, 
            "stop": True
        }

    async def _next_step(self, plan: List[str], step_outputs: List[Dict]) -> Dict[str, Any]:
        """Define next step."""
        current_step_idx = len(step_outputs)
        
        # Bug Fix #4: Bounds checking to prevent IndexError
        if not plan or current_step_idx >= len(plan):
            logger.error(f"Invalid plan state: plan length={len(plan)}, step_idx={current_step_idx}")
            return {
                "stop": True, 
                "plan_summary": {"output": "Failed", "answer": "Plan execution error: invalid step index", "score": 0}
            }
        
        cur_step_desc = plan[current_step_idx]
        
        logger.info(f"Defining next task for step {current_step_idx + 1}: {cur_step_desc}")
        
        memory = ""
        for i, output in enumerate(step_outputs):
            memory += f"Task: {plan[i]}\nAnswer: {output.get('answer', '')}\n\n"
            
        try:
            result = await self.step_chain.ainvoke({
                "plan": f"[{', '.join(plan)}]",
                "cur_step": cur_step_desc,
                "memory": memory
            })
            
            # Bug Fix #15: ensure task is not missing
            task_text = result.get("task")
            if not task_text:
                logger.warning(f"StepDefiner returned empty task for step: {cur_step_desc}. Using default.")
                task_text = cur_step_desc
                result["task"] = task_text
                
        except Exception as e:
            # Fallback on JSON parse error or other LLM failure
            logger.warning(f"StepDefiner LLM/Parse failed: {e}. Falling back to default task.")
            result = {
                "type": "question-answering",
                "task": cur_step_desc
            }
            
        logger.info(f"Defined task: {result.get('type')} - {result.get('task')}")
        return {"step_question": [result]}

async def step_node(state: PlanExecState) -> Dict[str, Any]:
    agent = StepDefinerAgent()
    return await agent.define_task(state)
