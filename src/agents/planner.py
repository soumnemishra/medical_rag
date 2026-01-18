from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.state.state import GraphState, PlanFormat
from src.prompts.templates import (
    PLANNING_SYSTEM_PROMPT,
    PLANNING_HUMAN_PROMPT
)
from src.core.registry import ModelRegistry
import logging

from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    Agent responsible for breaking down a complex query into a structured plan.
    """
    
    def __init__(self):
        self.llm = ModelRegistry.get_llm(temperature=0.3)
        self.parser = PydanticOutputParser(pydantic_object=PlanFormat)
        
        # Inject format instructions into system prompt
        system_prompt = PLANNING_SYSTEM_PROMPT + "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", PLANNING_HUMAN_PROMPT)
        ])
        
    def _format_memory(self, past_exp: List[Dict[str, Any]]) -> str:
        """Format past experiences into a string for the prompt."""
        if not past_exp:
            return "No past experience."
            
        memory_str = ""
        for i, exp in enumerate(past_exp):
            plan_str = ", ".join(exp.get("plan", []))
            summary = exp.get("plan_summary", {})
            status = summary.get("output", "Unknown")
            score = summary.get("score", 0)
            
            memory_str += f"Trial {i+1}:\n"
            memory_str += f"Plan: [{plan_str}]\n"
            memory_str += f"Status: {status} Score: {score}\n\n"
            
        return memory_str

    def plan(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a plan based on the original question and past experiences.
        """
        try:
            question = state["original_question"]
            past_exp = state.get("past_exp", [])
            memory = self._format_memory(past_exp)
            
            logger.info(f"PLANNING for: {question}")
            
            # Using the parser in the chain
            chain = self.prompt | self.llm | self.parser
            
            result = chain.invoke({
                "question": question,
                "memory": memory,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            logger.info(f"Generated Plan: {result.step}")
            return {"plan": result.step}
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback for robustness
            return {"plan": [f"Answer the question: {question}"]}

# Node entry point
def planner_node(state: GraphState) -> Dict[str, Any]:
    agent = PlannerAgent()
    return agent.plan(state)
