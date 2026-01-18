from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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
        self.llm = ModelRegistry.get_llm(temperature=0.3)
        
        self.step_parser = PydanticOutputParser(pydantic_object=StepTaskFormat)
        self.summary_parser = PydanticOutputParser(pydantic_object=PlanSummaryFormat)
        
        step_sys = STEP_DEFINER_SYSTEM_PROMPT + "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"
        self.step_prompt = ChatPromptTemplate.from_messages([
            ("system", step_sys),
            ("human", STEP_DEFINER_HUMAN_PROMPT)
        ])
        
        summary_sys = SUMMARY_SYSTEM_PROMPT + "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", summary_sys),
            ("human", SUMMARY_HUMAN_PROMPT)
        ])

    def define_task(self, state: PlanExecState) -> Dict[str, Any]:
        """
        Decide next step or summarize.
        """
        try:
            plan = state.get("plan", [])
            step_output = state.get("step_output", [])
            step_question = state.get("step_question", [])
            original_question = state.get("original_question", "")
            
            # Check stopping conditions:
            # 1. All steps in plan executed
            # 2. Last step explicitly reported "No" success
            should_stop = len(step_output) >= len(plan)
            if step_output and step_output[-1].get("success", "").lower() == "no":
                should_stop = True
                
            if should_stop:
                return self._summarize(original_question, plan, step_output, step_question)
            else:
                return self._next_step(plan, step_output)
                
        except Exception as e:
            logger.error(f"Task definition failed: {e}")
            # Fallback to stop and summarize with error
            return {"stop": True, "plan_summary": {"output": "Failed", "answer": f"Error: {str(e)}", "score": 0}}

    def _summarize(self, question: str, plan: List[str], step_outputs: List[Dict], step_questions: List[Dict]) -> Dict[str, Any]:
        """Generate final summary."""
        logger.info("Plan execution complete. Summarizing...")
        
        memory = ""
        for i, output in enumerate(step_outputs):
            task_desc = plan[i] if i < len(plan) else "Unknown Task"
            detailed_q = step_questions[i].get("task", "") if i < len(step_questions) else ""
            answer = output.get("answer", "")
            score = output.get("rating", 0)
            
            memory += f"Task: {task_desc}\nDetailed Query: {detailed_q}\nAnswer: {answer}\nConfidence: {score}\n\n"
            
        chain = self.summary_prompt | self.llm | self.summary_parser
        
        result = chain.invoke({
            "question": question,
            "plan": f"[{', '.join(plan)}]",
            "memory": memory,
            "format_instructions": self.summary_parser.get_format_instructions()
        })
        
        return {
            "plan_summary": result.model_dump(), 
            "stop": True
        }

    def _next_step(self, plan: List[str], step_outputs: List[Dict]) -> Dict[str, Any]:
        """Define next step."""
        current_step_idx = len(step_outputs)
        cur_step_desc = plan[current_step_idx]
        
        logger.info(f"Defining next task for step {current_step_idx + 1}: {cur_step_desc}")
        
        memory = ""
        for i, output in enumerate(step_outputs):
            memory += f"Task: {plan[i]}\nAnswer: {output.get('answer', '')}\n\n"
            
        chain = self.step_prompt | self.llm | self.step_parser
        
        result = chain.invoke({
            "plan": f"[{', '.join(plan)}]",
            "cur_step": cur_step_desc,
            "memory": memory,
            "format_instructions": self.step_parser.get_format_instructions()
        })
        
        logger.info(f"Defined task: {result.type} - {result.task}")
        # Need to return dict that matches StepTaskState
        return {"step_question": [result.model_dump()]}

def step_node(state: PlanExecState) -> Dict[str, Any]:
    agent = StepDefinerAgent()
    return agent.define_task(state)
