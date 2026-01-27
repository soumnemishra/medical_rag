from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import GraphState, ClinicalIntentFormat
from src.prompts.templates import CLINICAL_INTENT_SYSTEM_PROMPT, CLINICAL_INTENT_HUMAN_PROMPT
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

class ClinicalIntentAgent:
    """
    Agent responsible for classifying medical intent and risk level of queries.
    This is the first gate in the Clinical-Grade pipeline.
    """
    
    def __init__(self):
        # Use a fast model for intent classification to minimize latency
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser(pydantic_object=ClinicalIntentFormat)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_INTENT_SYSTEM_PROMPT),
            ("human", CLINICAL_INTENT_HUMAN_PROMPT)
        ])
        
    async def classify(self, state: GraphState) -> Dict[str, Any]:
        """
        Classify the intent of the question.
        Returns a dictionary with keys matching the GraphState fields to update.
        """
        try:
            question = state["original_question"]
            logger.info(f"Classifying intent for: {question}")
            
            chain = self.prompt | self.llm | self.parser
            
            result = await chain.ainvoke({
                "question": question
            })
            
            logger.info(f"Intent classification complete: {result}")
            
            # Return update for the state
            return {
                "intent": result.get("intent", "informational"),
                "risk_level": result.get("risk_level", "low"),
                "requires_disclaimer": result.get("requires_disclaimer", False),
                "needs_guidelines": result.get("needs_guidelines", False),
                "safety_flags": [] # Initialize empty flags
            }
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Fallback to safe defaults
            return {
                "intent": "informational", 
                "risk_level": "low",
                "requires_disclaimer": False,
                "needs_guidelines": False,
                "safety_flags": ["error_classification_failed"]
            }

async def clinical_intent_node(state: GraphState) -> Dict[str, Any]:
    agent = ClinicalIntentAgent()
    return await agent.classify(state)
