from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import GraphState, PlanFormat
from src.prompts.templates import PLANNING_SYSTEM_PROMPT, PLANNING_HUMAN_PROMPT
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    Agent responsible for breaking down a complex query into a structured plan.
    """
    
    def __init__(self):
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNING_SYSTEM_PROMPT),
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

    async def plan(self, state: GraphState) -> Dict[str, Any]:
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
            
            result = await chain.ainvoke({
                "question": question,
                "memory": memory
            })
            
            raw_steps = result.get("step", [])
            
            # Normalize steps to List[str] - handle both string and dict formats
            steps = []
            for step in raw_steps:
                if isinstance(step, str):
                    steps.append(step)
                elif isinstance(step, dict):
                    # LLM sometimes returns {'type': 'Search', 'question': '...'} format
                    step_text = step.get("question") or step.get("task") or step.get("step") or str(step)
                    steps.append(step_text)
                else:
                    steps.append(str(step))
            
            logger.info(f"Generated Plan: {steps}")
            return {"plan": steps}
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback
            return {"plan": [f"Answer the question: {state['original_question']}"]}


from src.agents.registry import AgentRegistry

async def planner_node(state: GraphState) -> Dict[str, Any]:
    agent = AgentRegistry.get_instance().planner
    return await agent.plan(state)

