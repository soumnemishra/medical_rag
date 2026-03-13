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
        self.parser = JsonOutputParser(pydantic_object=PlanFormat)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNING_SYSTEM_PROMPT),
            ("human", PLANNING_HUMAN_PROMPT)
        ])
        
    def _format_memory(self, past_exp: List[Dict[str, Any]]) -> str:
        """Format past experiences into a string for the prompt."""
        if not past_exp:
            return "No past experience."
            
        lines = []
        for i, exp in enumerate(past_exp):
            lines.append(f"Attempt {i+1}:")
            for j, step in enumerate(exp.get("plan", [])):
                # We need to extract results accurately if step outputs differ in schema
                # Falling back to empty object checks
                result = exp.get("step_output", [])[j] if j < len(exp.get("step_output", [])) else {}
                status = result.get("success", "unknown") if isinstance(result, dict) else "unknown"
                docs = exp.get("step_docs_ids", [])[j] if j < len(exp.get("step_docs_ids", [])) else []
                doc_count = len(docs)
                step_val = step.get("question", step) if isinstance(step, dict) else step
                lines.append(f"  Step {j+1}: '{step_val}' → {status}, {doc_count} docs found")
            
            score = exp.get("plan_summary", {}).get("score", 0) if isinstance(exp.get("plan_summary"), dict) else 0
            lines.append(f"  Overall score: {score}/1.0")
            lines.append(f"  Lesson: {exp.get('lesson', 'none recorded')}")
        return "\n".join(lines)

    async def plan(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a plan based on the original question and past experiences.
        """
        try:
            question = state["original_question"]
            past_exp = state.get("past_exp", [])
            intent = state.get("intent", "informational")
            risk_level = state.get("risk_level", "low")
            needs_guidelines = state.get("needs_guidelines", False)
            
            memory = self._format_memory(past_exp)
            
            logger.info(f"PLANNING for: {question} (Intent: {intent}, Risk: {risk_level})")
            
            # Using the parser in the chain
            chain = self.prompt | self.llm | self.parser
            
            result = await chain.ainvoke({
                "question": question,
                "memory": memory,
                "intent": intent,
                "risk_level": risk_level,
                "needs_guidelines": needs_guidelines
            })
            
            raw_steps = result.get("plan", [])
            
            steps = []
            for step in raw_steps:
                if isinstance(step, dict):
                    # Keep valid structure assuming LLM mapped output to format
                    if "id" not in step or "question" not in step:
                        steps.append({"id": len(steps)+1, "question": step.get("question", str(step)), "depends_on": []})
                    else:
                        steps.append(step)
                elif isinstance(step, str):
                    steps.append({"id": len(steps)+1, "question": step, "depends_on": []})
                else:
                    steps.append({"id": len(steps)+1, "question": str(step), "depends_on": []})
            
            if not steps:
                logger.error("Planner returned empty steps — LLM may have returned wrong key")
                return {
                    "plan": [],
                    "safety_flags": state.get("safety_flags", []) + ["empty_plan"]
                }
            
            logger.info(f"Generated Plan ({len(steps)} steps): {[s.get('question', '') for s in steps]}")
            return {
                "plan": steps,
                "plan_complexity": result.get("complexity", "moderate")
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback that propagates errors correctly
            return {
                "plan": [],
                "plan_error": str(e),
                "safety_flags": state.get("safety_flags", []) + ["planning_failed"]
            }


from src.agents.registry import AgentRegistry

async def planner_node(state: GraphState) -> Dict[str, Any]:
    agent = AgentRegistry.get_instance().planner
    return await agent.plan(state)

