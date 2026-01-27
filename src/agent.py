# FILE: src/agent.py
"""
MedicalAgent - High-level interface for the MA-RAG pipeline.

This module provides a simple interface for the Streamlit app to interact
with the multi-agent RAG system.

Example:
    agent = MedicalAgent()
    response, query_log, reasoning = await agent.chat("What is the treatment for diabetes?")
"""

import logging
from typing import List, Dict, Any, Tuple

from src.orchestrator.graph import build_graph
from src.state.state import GraphState

logger = logging.getLogger(__name__)


class MedicalAgent:
    """
    High-level interface for medical question answering using MA-RAG.
    
    Wraps the LangGraph orchestration pipeline and provides a simple
    async interface for chat-based interactions.
    """
    
    def __init__(self):
        """Initialize the MedicalAgent with the MA-RAG graph."""
        # Lazy initialization
        logger.info("Initializing MedicalAgent (Lazy Load)...")
        self.graph = None
        # self.graph = build_graph() # Defer build to first call
    
    async def chat(
        self, 
        query: str, 
        history: List[Dict[str, str]] = None
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a medical question and return the answer.
        
        Returns:
            Tuple of (answer, query_log, reasoning_steps, risk_metadata)
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Lazy init
            if self.graph is None:
                logger.info("Initializing Graph lazily...")
                self.graph = build_graph()
            
            # Initialize the graph state
            initial_state: GraphState = {
                "original_question": query,
                "plan": [],
                "past_exp": [],
                "final_answer": "",
                "intent": "informational",
                "risk_level": "low",
                "safety_flags": [],
                "needs_guidelines": False,
                "requires_disclaimer": False
            }
            
            # Run the graph asynchronously
            result = await self.graph.ainvoke(initial_state)
            
            # Extract the final answer
            final_answer = result.get("final_answer", "I was unable to generate an answer.")
            
            # Build query log from past experiences
            query_log = []
            reasoning_steps = []
            
            # Extract reasoning from past experiences
            past_exp = result.get("past_exp", [])
            if past_exp:
                exp = past_exp[0]  # Get the first (and usually only) execution
                
                # Add planning phase
                plan = exp.get("plan", [])
                if plan:
                    reasoning_steps.append({
                        "phase": "PLANNING",
                        "thought": f"Breaking down the question into {len(plan)} steps",
                        "details": ", ".join(plan[:3]) + ("..." if len(plan) > 3 else "")
                    })
                
                # Add execution phases
                step_questions = exp.get("step_question", [])
                step_outputs = exp.get("step_output", [])
                
                for i, (sq, so) in enumerate(zip(step_questions, step_outputs)):
                    task_type = sq.get("type", "question-answering")
                    task = sq.get("task", "")
                    answer = so.get("answer", "")[:200]
                    
                    phase = "SEARCHING" if task_type == "question-answering" else "SYNTHESIZING"
                    reasoning_steps.append({
                        "phase": phase,
                        "thought": task[:100],
                        "details": answer[:150] + ("..." if len(answer) > 150 else "")
                    })
                
                # Add final synthesis phase
                reasoning_steps.append({
                    "phase": "SYNTHESIZING",
                    "thought": "Combining all findings into final answer",
                    "details": ""
                })
            
            logger.info(f"Query processed successfully. Answer length: {len(final_answer)}")
            
            # Extract risk metadata for UI display
            risk_metadata = {
                "risk_level": result.get("risk_level", "low"),
                "intent": result.get("intent", "informational"),
                "requires_disclaimer": result.get("requires_disclaimer", False),
                "needs_guidelines": result.get("needs_guidelines", False)
            }
            
            return final_answer, query_log, reasoning_steps, risk_metadata
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I encountered an error while processing your question: {str(e)}"
            return error_response, [], [], {"risk_level": "unknown", "intent": "error"}
    
    async def answer_query(self, query: str) -> "AgentResult":
        """
        Simplified interface for evaluation.
        """
        response, _, _ = await self.chat(query)
        return AgentResult(answer=response, sources=[])


class AgentResult:
    """Result from agent query for evaluation compatibility."""
    
    def __init__(self, answer: str, sources: List[str]):
        self.answer = answer
        self.sources = sources
