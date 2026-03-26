###################################### tools that we will be using ############################
from typing import Dict, Any  #python way of saying this is dict 
from langchain_core.prompts import ChatPromptTemplate # a template for talking to ai 
from langchain_core.output_parsers import JsonOutputParser #makes ai to give clean  structure answer 
#the graph state is the shared note book 
from src.state.state import GraphState, ClinicalIntentFormat #this is the shared memory accross the agents  and each agent read from this file 
from src.prompts.templates import CLINICAL_INTENT_SYSTEM_PROMPT, CLINICAL_INTENT_HUMAN_PROMPT #this contain actual instrcutions 
from src.core.registry import ModelRegistry #the warehouse that holds the model 
import logging #for keeping records of what happened 
from tenacity import retry, stop_after_attempt, wait_fixed  # this basically solves api timeout etc 

logger = logging.getLogger(__name__)



############# thinnk of like this is the job description of the agent ############# 
class ClinicalIntentAgent:
    """
    Agent responsible for classifying medical intent and risk level of queries.
    This is the first gate in the Clinical-Grade pipeline.
    """
    #this is setuping the agent desk 
    def __init__(self):
        # Use a fast model for intent classification to minimize latency
         #this is just like ticking the boxes on form 
         #get the llm brain ready selm.llm get ai brain ready 
         #model registry says go to the warehouse and get me the flash llm (small fast model)
         #temperature is set to 0.0 to make the model more deterministic and less creative
         #json_mode=true forces the brain to speak in structured format not in setence 
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)#we are assigning here a small model 
        if self.llm is None:
            raise RuntimeError("ClinicalIntentAgent: Flash LLM failed to load. Check ModelRegistry.")
        #because intent classification is a simple task and does not require a large model
        #the temperature is set to 0.0 to make the model more deterministic and less creative
        #this json_mode=true forces the brain to speak in structured format not in setence 

        # it takes the  json output of the llm and compare it with clinical itent format 
        #so that it doesnot breaks the shared state doesnot breaks 
        self.parser = JsonOutputParser(pydantic_object=ClinicalIntentFormat)#this just instruct the llm to fll out the form 
        #it just fill and pass to the next agent so that it doesnot breaks the shared state doesnot breaks 

        #####  below think of like the instructions for the agent 
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_INTENT_SYSTEM_PROMPT),
            ("human", CLINICAL_INTENT_HUMAN_PROMPT)
        ])

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _classify_with_retry(self, question: str) -> dict:
        chain = self.prompt | self.llm | self.parser
        return await chain.ainvoke({"question": question})
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
            
            result = await self._classify_with_retry(question)
            
            logger.info(f"Intent classification complete: {result}")
            
            # Return update for the state
            return {
                "intent": result.get("intent", "informational"),
                "risk_level": result.get("risk_level", "high"),
                "requires_disclaimer": result.get("requires_disclaimer", True),
                "needs_guidelines": result.get("needs_guidelines", True),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "safety_flags": []
            }
            
        except Exception as e:
            logger.error(f"Failed after retries: {e}")
            # Fallback to safe defaults
            return {
                "intent": "informational", 
                "risk_level": "high",
                "requires_disclaimer": True,
                "needs_guidelines": True,
                "safety_flags": ["classification_error_safe_fallback"]
            }


from src.agents.registry import AgentRegistry
#The Integration (clinical_intent_node)
#This is a "wrapper" function. 
# #It’s what your StateGraph actually calls. 
# It pulls the agent from a Registry (a warehouse for your agents) 
# so you don't keep recreating the same agent over and over, saving memory.
async def clinical_intent_node(state: GraphState) -> Dict[str, Any]:
    try:
        agent = AgentRegistry.get_instance().clinical_intent
        return await agent.classify(state)
    except Exception as e:
        logger.critical(f"clinical_intent_node failed: {e}")
        return {
            "intent": "informational",
            "risk_level": "high",
            "requires_disclaimer": True,
            "needs_guidelines": True,
            "safety_flags": ["node_level_error"]
        }

#“Each agent in my system works like a medical specialist.
#They all read from a shared state and only add their own findings, which makes the system safe, debuggable, and modular.”