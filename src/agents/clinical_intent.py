# ClinicalIntentAgent decides what kind of medical question this is 
# and how risky it is, before any retrieval or generation happens.
#This is not an answer-generation agent.
#It is a control + safety + routing agent.

'''2️ Why this agent exists (architectural reason)

In medical RAG systems, you cannot treat all queries equally.

Example:

“What is diabetes?” → safe, informational

 “Can I stop taking insulin?” → high risk, clinical

 “I have chest pain right now” → emergency

 So this agent answers:

Is this safe to answer normally?

Do we need disclaimers?

Do we need clinical guidelines?

Should downstream agents be restricted?

That’s why your docstring says:

“This is the first gate in the Clinical-Grade pipeline.”'''
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
#the graph state is the shared note book 
from src.state.state import GraphState, ClinicalIntentFormat #this is the shared memory accross the agents 
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
         #this is just like ticking the boxes on form 
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)#we are assigning here a small model 
        #because intent classification is a simple task and does not require a large model
        #the temperature is set to 0.0 to make the model more deterministic and less creative
        #this json_mode=true forces the brain to speak in structured format not in setence 
        self.parser = JsonOutputParser(pydantic_object=ClinicalIntentFormat)#this just instruct the llm to fll out the form 
        #it just fill and pass to the next agent so that it doesnot breaks the shared state doesnot breaks 
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_INTENT_SYSTEM_PROMPT),
            ("human", CLINICAL_INTENT_HUMAN_PROMPT)
        ])
        #async create agent reusability si that agent is initialized once and not all the time this saves time and computation
    async def classify(self, state: GraphState) -> Dict[str, Any]:
        """
        Classify the intent of the question.
        Returns a dictionary with keys matching the GraphState fields to update.
        ainvoke is the "Start" button. It pushes the question into the first station.
        await is the "Waiting Room." Because the LLM "Brain" takes a few seconds to think, await tells the rest of your computer: 
        "Hey, go do other chores while we wait for this brain to finish. Don't just sit here frozen.
        """
        try:
            question = state["original_question"]
            logger.info(f"Classifying intent for: {question}")
            
            chain = self.prompt | self.llm | self.parser #instruction-->thinking--> structuredoutput
            
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


from src.agents.registry import AgentRegistry
#The Integration (clinical_intent_node)
#This is a "wrapper" function. 
# #It’s what your StateGraph actually calls. 
# It pulls the agent from a Registry (a warehouse for your agents) 
# so you don't keep recreating the same agent over and over, saving memory.
async def clinical_intent_node(state: GraphState) -> Dict[str, Any]:
    agent = AgentRegistry.get_instance().clinical_intent
    return await agent.classify(state)

#“Each agent in my system works like a medical specialist.
#They all read from a shared state and only add their own findings, which makes the system safe, debuggable, and modular.”