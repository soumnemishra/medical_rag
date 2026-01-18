# FILE: src/agents/synthesizer.py
"""
Synthesizer Agent for the Agentic RAG system.

Generates final answers in medical SOAP format based on extracted evidence.
Includes confidence scoring for backtracking decisions.

Example Usage:
    >>> from src.agents.synthesizer import SynthesizerAgent
    >>> synthesizer = SynthesizerAgent()
    >>> state = await synthesizer.synthesize(state)
    >>> print(state["final_answer"])
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.orchestrator.state import AgentStep, GraphState, QAAnswer

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

SYNTHESIZER_SYSTEM_PROMPT = """You are a medical research synthesizer providing evidence-based answers for healthcare professionals.

Your task is to synthesize information from extracted research findings into a clear, actionable answer.

Guidelines:
1. **Analyze Evidence**: Review all extracted notes carefully
2. **Assess Consistency**: Note if findings agree or conflict
3. **Provide Answer**: Give a direct answer to the question
4. **Cite Sources**: Reference PMIDs for key claims
5. **Rate Confidence**: Score 0-10 based on evidence quality

Confidence Scoring:
- 0-3: Insufficient or conflicting evidence
- 4-6: Moderate evidence, some uncertainty
- 7-9: Strong evidence, high confidence
- 10: Definitive evidence from multiple high-quality sources

Answer Format:
- Be direct and actionable
- Use medical terminology appropriate for clinicians
- Include key statistics when available
- Acknowledge limitations or uncertainties

IMPORTANT: Only answer based on the provided evidence. If evidence is insufficient, say so and rate confidence appropriately."""

SYNTHESIZER_HUMAN_PROMPT = """Original Question: {question}

Extracted Evidence:
{evidence}

Provide your synthesized answer with confidence rating:"""


class SynthesizerAgent:
    """
    Agent for synthesizing final answers from extracted evidence.
    
    Generates answers in medical format with confidence scoring
    for TC-RAG style backtracking decisions.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        use_local: bool = False,  # Default to Gemini for synthesis
    ):
        """
        Initialize the Synthesizer Agent.
        
        Args:
            llm: Optional LangChain LLM
            use_local: If True, use local Ollama (not recommended for synthesis)
        """
        self.llm = llm
        self.use_local = use_local
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIZER_SYSTEM_PROMPT),
            ("human", SYNTHESIZER_HUMAN_PROMPT),
        ])
        
        logger.info(f"SynthesizerAgent initialized (use_local={use_local})")
    
    def _get_llm(self) -> Any:
        """Get or create the LLM instance with Ollama primary, Gemini fallback."""
        if self.llm is not None:
            return self.llm
        
        # Try Ollama first (local)
        if self.use_local:
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(
                    model="llama3.2:3b",  # Better for synthesis
                    base_url="http://localhost:11434",
                    temperature=0.3,
                )
                logger.info("Synthesizer using local Ollama (llama3.2:3b)")
            except Exception as e:
                logger.warning(f"Ollama failed: {e}, falling back to Gemini")
                self.use_local = False
                return self._get_llm()
        else:
            # Fallback to Gemini
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                logger.info("Synthesizer using Gemini")
            except Exception as e:
                logger.error(f"Gemini also failed: {e}")
                raise
        
        return self.llm
    
    def _format_evidence(self, state: GraphState) -> str:
        """Format extracted notes as evidence for the prompt."""
        notes = state.get("extracted_notes", [])
        
        if not notes:
            # Fallback to raw documents
            docs = state.get("retrieved_docs", [])
            if not docs:
                return "No evidence available."
            
            parts = []
            for doc in docs[:5]:
                parts.append(f"[PMID:{doc['pmid']}] {doc['title']}\n{doc['abstract'][:500]}...")
            return "\n\n".join(parts)
        
        return "\n\n".join(notes)
    
    def _extract_confidence(self, response: str) -> int:
        """Extract confidence score from LLM response."""
        import re
        
        # Look for patterns like "Confidence: 7/10", "confidence score: 8", etc.
        patterns = [
            r'[Cc]onfidence[:\s]+(\d+)\s*/\s*10',
            r'[Cc]onfidence[:\s]+(\d+)',
            r'(\d+)/10',
            r'score[:\s]+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 10)  # Clamp to 0-10
        
        # Default confidence based on response length and presence of PMIDs
        if 'PMID' in response and len(response) > 200:
            return 7
        elif len(response) > 100:
            return 5
        return 3
    
    def _clean_answer(self, response: str) -> str:
        """Clean the LLM response for display."""
        import re
        
        # Remove confidence scoring lines at the end
        response = re.sub(r'\n*[Cc]onfidence[:\s]+\d+.*$', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n*\d+/10\s*$', '', response, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        response = response.strip()
        
        return response
    
    async def synthesize(self, state: GraphState) -> Dict[str, Any]:
        """
        Synthesize final answer from extracted evidence.
        
        Args:
            state: Current graph state with extracted_notes
        
        Returns:
            State updates with final_answer and confidence
        """
        from langchain_core.output_parsers import StrOutputParser
        import re
        
        question = state["original_question"]
        evidence = self._format_evidence(state)
        iteration = state.get("iteration_count", 0)
        sources = state.get("sources", [])
        
        logger.info(f"[SYNTHESIZER] Generating answer (iteration {iteration})")
        
        try:
            llm = self._get_llm()
            
            # Use StrOutputParser for Ollama compatibility
            chain = self.prompt | llm | StrOutputParser()
            
            raw_response = await chain.ainvoke({
                "question": question,
                "evidence": evidence,
            })
            
            logger.info(f"[SYNTHESIZER] Got response, parsing...")
            
            # Parse confidence from response
            confidence = self._extract_confidence(raw_response)
            answer = self._clean_answer(raw_response)
            
            logger.info(f"[SYNTHESIZER] Answer generated, confidence={confidence}/10")
            
            # Build formatted answer with sources
            formatted_answer = f"{answer}\n\n"
            if sources:
                formatted_answer += f"**Sources:** {', '.join([f'PMID:{s}' for s in sources[:5]])}\n"
            formatted_answer += f"\n*Confidence: {confidence}/10*"
            
            return {
                "final_answer": formatted_answer,
                "final_confidence": confidence,
                "current_step": AgentStep.DONE,
                "reasoning_trace": [{
                    "phase": "SYNTHESIZE",
                    "thought": f"Generated answer with confidence {confidence}/10",
                    "details": answer[:200],
                    "iteration": iteration,
                }],
            }
            
        except Exception as e:
            logger.error(f"[SYNTHESIZER] Failed: {e}")
            
            # Fallback answer
            fallback = "Unable to generate a complete answer based on the available evidence."
            
            return {
                "final_answer": fallback,
                "final_confidence": 3,  # Low confidence triggers backtrack
                "current_step": AgentStep.DONE,
                "reasoning_trace": [{
                    "phase": "SYNTHESIZE",
                    "thought": f"Synthesis failed: {str(e)[:100]}",
                    "iteration": iteration,
                }],
            }
    
    def synthesize_sync(self, state: GraphState) -> Dict[str, Any]:
        """Synchronous version."""
        import asyncio
        return asyncio.run(self.synthesize(state))


# =============================================================================
# Node Function
# =============================================================================

_synthesizer_agent: Optional[SynthesizerAgent] = None


def get_synthesizer_agent() -> SynthesizerAgent:
    """Get or create the global synthesizer agent."""
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = SynthesizerAgent(use_local=True)  # Try Ollama first
    return _synthesizer_agent


async def synthesizer_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node function for synthesis."""
    agent = get_synthesizer_agent()
    return await agent.synthesize(state)


def synthesizer_node_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous version."""
    agent = get_synthesizer_agent()
    return agent.synthesize_sync(state)
