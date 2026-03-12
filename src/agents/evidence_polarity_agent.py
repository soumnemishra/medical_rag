from typing import Dict, Any, List  # this are the type hints we tell python that we are going to work on the dict and list and any other data types
from langchain_core.prompts import ChatPromptTemplate # this is the template for the prompt
from langchain_core.output_parsers import JsonOutputParser   # this is the parser for the output
from src.state.state import GraphState, EvidencePolarity # this is the state of the graph
from src.prompts.templates import EVIDENCE_POLARITY_SYSTEM_PROMPT, EVIDENCE_POLARITY_HUMAN_PROMPT # this is the prompt for the evidence polarity
from src.core.registry import ModelRegistry # this is the registry for the models
import logging # this is the logger for the logs

logger = logging.getLogger(__name__)

class EvidencePolarityAgent:
    """
    Agent responsible for detecting the directional polarity of retrieved evidence 
    (support/refute/mixed/insufficient) without altering the final answer.
    """
    
    def __init__(self):
        # Use Flash model for low latency
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True) # this is the model that we are going to use for the evidence polarity
        self.parser = JsonOutputParser(pydantic_object=EvidencePolarity) # we tell the llm to fill out this form for the next agent 
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_POLARITY_SYSTEM_PROMPT), # this is the system prompt for the evidence polarity
            ("human", EVIDENCE_POLARITY_HUMAN_PROMPT) # this is the human prompt for the evidence polarity
        ])
        
    def _format_evidence(self, past_exp: List[Dict[str, Any]]) -> str:
        """
        if search didnt find any evidence it will return "No evidence retrieved."
        """
        if not past_exp:
            return "No evidence retrieved."
        # this creates a empty string 
        # begins looking through the past_exp (past search results). It grabs the text (step_notes) and the IDs (PMIDs  
        evidence_str = ""
        count = 1
        # past exp--> past search results 
        for exp in past_exp:
            # Check for direct RAG notes/docs
            step_notes = exp.get("step_notes", [])
            step_ids = exp.get("step_docs_ids", [])
            
            # Handle nested list structure of step_notes/step_docs_ids
            #It loops through every note, attaches its ID (PMID), and adds it to the list. 
            # It cuts off any text longer than 1,000 characters to keep it from getting too long
            for i, notes_list in enumerate(step_notes):
                # Flatten if it's a list of lists
                if isinstance(notes_list, list):
                    current_notes = notes_list
                else:
                    current_notes = [str(notes_list)]
                
                # Get corresponding IDs if available
                current_ids = []
                if i < len(step_ids):
                    chunk_ids = step_ids[i]
                    if isinstance(chunk_ids, list):
                        current_ids = chunk_ids
                    
                # Format each note
                for note in current_notes:
                    # Skip empty notes
                    if not note or str(note).strip() == "":
                        continue
                        
                    ids_str = f" (PMID: {', '.join(current_ids)})" if current_ids else ""
                    evidence_str += f"Item {count}{ids_str}:\n{note[:1000]}\n\n" # Truncate very long notes
                    count += 1
                    
        if not evidence_str:
            return "No relevant evidence text found in execution history."
            
        return evidence_str

    async def analyze(self, state: GraphState) -> Dict[str, Any]:
        """
        open the shared notebook and read the evidence and decide the polarity.
        """
        try:
            question = state["original_question"]
            past_exp = state.get("past_exp", [])
            
            logger.info(f"Analyzing evidence polarity for: {question}")
            #this is the evidence that we are going to use for the evidence polarity
            evidence_text = self._format_evidence(past_exp)
            
            chain = self.prompt | self.llm | self.parser
            #It hits the "Start" button on the assembly line. The AI reads the question and the evidence, then fills out the form.
            result = await chain.ainvoke({
                "question": question,
                "evidence": evidence_text
            })
            
            logger.info(f"Evidence Polarity: {result}")
            #It returns the results back to the state. It tells the system: "The research says Support/Refute/Mixed, and I am X% sure.
            return {
                "evidence_polarity": {
                    "polarity": result.get("polarity", "insufficient"),
                    "confidence": float(result.get("confidence", 0.0))
                }
            }
            
        except Exception as e:
            logger.error(f"Evidence polarity analysis failed: {e}")
            # Non-blocking failure
            return {
                "evidence_polarity": {
                    "polarity": "insufficient",
                    "confidence": 0.0
                }
            }

#This is the function that the "Graph" (the Orchestrator) actually calls. It grabs the agent from the "Registry" and tells it to start the analysis.
from src.agents.registry import AgentRegistry

async def evidence_polarity_node(state: GraphState) -> Dict[str, Any]:
    agent = AgentRegistry.get_instance().evidence_polarity
    return await agent.analyze(state)
