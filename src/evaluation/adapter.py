from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging

from src.orchestrator.graph import build_graph
from src.state.state import GraphState

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    answer: str
    sources: List[str] = field(default_factory=list)

class MaRagAdapter:
    """
    Adapter to make MA-RAG pipeline compatible with PubMedQAEvaluator.
    """
    
    def __init__(self):
        self.graph = build_graph()
        
    async def answer_query(self, prompt: str) -> AgentResult:
        """
        Run the pipeline and return result in expected format.
        """
        # The prompt from PubMedQA includes instructions. 
        # We might want to extract just the question if the planner gets confused,
        # but the planner is robust. We'll feed the whole prompt or just the question part?
        # PubMedQAQuestion.to_prompt() formats it with "Question: ... Options: ...".
        # This is fine for the Planner.
        
        initial_state: GraphState = {
            "original_question": prompt,
            "plan": [],
            "past_exp": [],
            "final_answer": ""
        }
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            final_answer = result.get("final_answer", "")
            if not final_answer:
                # Fallback if final_answer not set (e.g. error)
                past_exp = result.get("past_exp", [])
                if past_exp:
                    last_exp = past_exp[-1]
                    summary = last_exp.get("plan_summary", {})
                    final_answer = summary.get("answer", "No answer generated.")
                else:
                    final_answer = "No answer generated."

            print(f"\n[DEBUG] Raw Model Output: {final_answer[:500]}...\n")
            
            # Collect sources (PMIDs or URLs from document objects)
            # MA-RAG stores doc IDs in step_docs_ids
            all_sources = set()
            past_exp = result.get("past_exp", [])
            for exp in past_exp:
                step_ids = exp.get("step_docs_ids", [])
                for ids_list in step_ids:
                    for doc_id in ids_list:
                        all_sources.add(doc_id)
            
            return AgentResult(
                answer=final_answer,
                sources=list(all_sources)
            )
            
        except Exception as e:
            logger.error(f"Adapter failed: {e}")
            return AgentResult(answer=f"Error: {e}", sources=[])
