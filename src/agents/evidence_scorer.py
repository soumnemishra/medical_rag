
# this is the medical medical researcher and reviewer
# Input	Raw text notes from papers.	To turn messy text into data.
# Batching	Process 5 notes at a time.	To prevent LLM "brain fog" and errors.
# Grading	Assign A, B, or C.	To tell the clinician how much to trust the fact.
# Fallback	Default to Grade C if error occurs.	To keep the system running no matter what. 



from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.prompts.templates import EVIDENCE_SCORER_SYSTEM_PROMPT, EVIDENCE_SCORER_HUMAN_PROMPT
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

# Maximum facts to score per LLM call to prevent hanging
BATCH_SIZE = 5
#so that it doesnot causes the llm overload and it goes into batches 
class EvidenceScorerAgent:
    """
    Agent responsible for grading the quality of extracted evidence.
    Assigns study types and grades (A/B/C) to facts.
    """
    
    def __init__(self):
        # Use a fast model for scoring
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)  
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVIDENCE_SCORER_SYSTEM_PROMPT),
            ("human", EVIDENCE_SCORER_HUMAN_PROMPT)
        ])

    async def _score_batch(self, notes: List[str]) -> List[Dict[str, Any]]:
        """Score a single batch of notes."""
        if not notes:
            return []
            
        notes_str = "\n".join([f"- {note}" for note in notes]) 
        chain = self.prompt | self.llm | self.parser #instructions--> reasoning--> answer
        
        try:
            result = await chain.ainvoke({"notes": notes_str})
            scored_data = result.get("scored_notes", [])
            
            final_scored = []
            for item in scored_data:
                final_scored.append({
                    "fact": item.get("fact", "Unknown Fact"),
                    "study_type": item.get("study_type", "Unspecified"),
                    "grade": item.get("grade", "C"),
                    "confidence": item.get("confidence", 0.0)
                })
            return final_scored
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            # Fallback for failed batch
            return [{"fact": note, "grade": "C", "study_type": "Unknown"} for note in notes]

    async def score_notes(self, notes: List[str]) -> List[Dict[str, Any]]: #
        """
        Score a list of extracted notes/facts in batches.
        Args:
            notes: List of fact strings.
        Returns:
            List of dicts with fact, study_type, grade, confidence.
        """
        if not notes:
            return []
        
        logger.info(f"Scoring {len(notes)} facts for evidence quality in batches of {BATCH_SIZE}...")
        
        all_scored = []
        
        # Process in batches to prevent LLM from hanging
        for i in range(0, len(notes), BATCH_SIZE):
            batch = notes[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(notes) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"Scoring batch {batch_num}/{total_batches} ({len(batch)} facts)...")
            
            scored_batch = await self._score_batch(batch)
            all_scored.extend(scored_batch)
        
        logger.info(f"Evidence scoring complete. {len(all_scored)} facts scored.")
        return all_scored



#The EvidenceScorerAgent evaluates the strength of extracted medical facts by classifying 
# study types and grading evidence quality in a deterministic, batched, and fail-safe manner.