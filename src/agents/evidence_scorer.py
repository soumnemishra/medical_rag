from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.prompts.templates import EVIDENCE_SCORER_SYSTEM_PROMPT, EVIDENCE_SCORER_HUMAN_PROMPT
from src.core.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

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

    async def score_notes(self, notes: List[str]) -> List[Dict[str, Any]]:
        """
        Score a list of extracted notes/facts.
        Args:
            notes: List of fact strings.
        Returns:
            List of dicts with fact, study_type, grade, confidence.
        """
        if not notes:
            return []
            
        try:
            # Format notes for prompt
            notes_str = "\n".join([f"- {note}" for note in notes])
            
            chain = self.prompt | self.llm | self.parser
            
            logger.info(f"Scoring {len(notes)} facts for evidence quality...")
            result = await chain.ainvoke({"notes": notes_str})
            
            scored_data = result.get("scored_notes", [])
            
            # Sanitize and ensure all input facts are returned (or best effort)
            final_scored = []
            for item in scored_data:
                final_scored.append({
                    "fact": item.get("fact", "Unknown Fact"),
                    "study_type": item.get("study_type", "Unspecified"),
                    "grade": item.get("grade", "C"), # Default to Low
                    "confidence": item.get("confidence", 0.0)
                })
            
            logger.info(f"Evidence scoring complete. {len(final_scored)} facts scored.")
            return final_scored

        except Exception as e:
            logger.error(f"Evidence scoring failed: {e}")
            # Fallback: return unscored dicts
            return [{"fact": note, "grade": "C", "study_type": "Unknown"} for note in notes]
