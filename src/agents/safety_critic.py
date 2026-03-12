from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import GraphState
from src.prompts.templates import SAFETY_CRITIC_SYSTEM_PROMPT, SAFETY_CRITIC_HUMAN_PROMPT
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

class ClinicalSafetyCriticAgent:
    """
    Agent responsible for auditing the final answer for safety compliance.
    """
    
    def __init__(self):
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SAFETY_CRITIC_SYSTEM_PROMPT),
            ("human", SAFETY_CRITIC_HUMAN_PROMPT)
        ])

    async def critique(self, state: GraphState) -> Dict[str, Any]:
        """
        Audit the final answer.
        Returns dictionary with updates to GraphState (final_answer, safety_flags).
        """
        try:
            answer = state.get("final_answer", "")
            intent = state.get("intent", "unknown")
            risk = state.get("risk_level", "low")
            
            # Check for evaluation mode (New Flag)
            evaluation_mode = state.get("evaluation_mode", False)
            
            # Skip empty answers or error states
            if not answer or "Error" in answer:
                return {"safety_flags": ["skipped_empty"]}
                
            logger.info(f"Auditing answer for safety (Risk: {risk})...")
            
            chain = self.prompt | self.llm | self.parser
            
            result = await chain.ainvoke({
                "answer": answer,
                "intent": intent,
                "risk_level": risk
            })
            
            is_safe = result.get("is_safe", True)
            issues = result.get("issues", [])
            refined = result.get("refined_answer")
            
            if not is_safe:
                logger.warning(f"Safety Check FAILED. Issues: {issues}")
                
                # EVALUATION MODE LOGIC
                if evaluation_mode:
                    logger.info("EVALUATION MODE: Preserving original answer despite safety flags.")
                    # Return original answer, but keep flags for metrics
                    return {
                        "final_answer": answer,
                        "safety_flags": issues
                    }
                
                # STANDARD MODE LOGIC (Overwrite/Append)
                if refined:
                    logger.info("Applying refined safe answer.")
                    return {
                        "final_answer": refined,
                        "safety_flags": issues
                    }
                else:
                    # If unsafe but no refinement, append warning
                    warning = "\n\n**SAFETY WARNING**: This content has been flagged for review: " + "; ".join(issues)
                    return {
                        "final_answer": answer + warning,
                        "safety_flags": issues
                    }
            
            logger.info("Safety Check PASSED.")
            return {"safety_flags": []}

        except Exception as e:
            logger.error(f"Safety audit failed: {e}")
            return {"safety_flags": ["audit_error"]}


from src.agents.registry import AgentRegistry

async def safety_critic_node(state: GraphState) -> Dict[str, Any]:
    agent = AgentRegistry.get_instance().safety_critic
    return await agent.critique(state)

