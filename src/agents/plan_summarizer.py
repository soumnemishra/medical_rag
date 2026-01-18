# FILE: src/agents/plan_summarizer.py
"""
Plan Summarizer Agent for MA-RAG style plan evaluation.

Evaluates completed plan execution and produces final answer
with aggregate confidence score.

Example Usage:
    >>> from src.agents.plan_summarizer import PlanSummarizerAgent
    >>> summarizer = PlanSummarizerAgent()
    >>> result = summarizer.summarize(state)
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.orchestrator.state import GraphState, PlanExecutorState, PlanSummary
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a medical research summarizer. Your task is to evaluate
a plan's execution and produce a final answer to the original question.

Input:
- Original medical question
- The plan (sequence of sub-tasks)
- Output of each step in the plan

Your task:
1. Review all step outputs for relevance and quality
2. Synthesize a comprehensive final answer
3. Rate your confidence based on evidence quality
4. Note if any steps failed or produced insufficient results

Output Format:
STATUS: Successful or Unsuccessful
FINAL_ANSWER: <comprehensive answer based on all evidence>
CONFIDENCE: <0-10 based on evidence quality>
REASONING: <brief explanation of your synthesis>
"""

SUMMARY_HUMAN_PROMPT = """Original Question: {question}

Plan: {plan}

Step Outputs:
{memory}

Based on the above plan execution, provide your final synthesis.
Original Question: {question}
"""


class PlanSummarizerAgent:
    """
    Agent for summarizing plan execution and producing final answer.
    
    MA-RAG style evaluation with confidence aggregation.
    """
    
    def __init__(self, llm: Optional[Any] = None, use_local: bool = True):
        self.llm = llm
        self.use_local = use_local
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", SUMMARY_HUMAN_PROMPT),
        ])
        
        logger.info("PlanSummarizerAgent initialized")
    
    def _get_llm(self) -> Any:
        """Get LLM with Ollama primary, Gemini fallback."""
        if self.llm is not None:
            return self.llm
        
        if self.use_local:
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(
                    model="llama3.2:3b",  # Better for synthesis
                    base_url="http://localhost:11434",
                    temperature=0.3,
                )
                logger.info("PlanSummarizer using Ollama (llama3.2:3b)")
            except Exception as e:
                logger.warning(f"Ollama failed: {e}, using Gemini")
                self.use_local = False
                return self._get_llm()
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                logger.info("PlanSummarizer using Gemini")
            except Exception as e:
                logger.error(f"Gemini also failed: {e}")
                raise
        
        return self.llm
    
    def _format_step_memory(self, step_outputs: List[Dict]) -> str:
        """Format step outputs for the prompt."""
        if not step_outputs:
            return "No step outputs available."
        
        parts = []
        for i, step in enumerate(step_outputs):
            step_name = step.get("step_name", f"Step {i+1}")
            task = step.get("task", "Unknown task")
            answer = step.get("answer", {})
            
            if isinstance(answer, dict):
                answer_text = answer.get("answer", "No answer")
                confidence = answer.get("confidence", 5)
                success = answer.get("success", "unknown")
            else:
                answer_text = str(answer)
                confidence = 5
                success = "yes"
            
            parts.append(f"""
{step_name}:
- Task: {task}
- Answer: {answer_text}
- Confidence: {confidence}/10
- Success: {success}
""")
        
        return "\n".join(parts)
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse summary response from LLM."""
        result = {
            "output": "Unsuccessful",
            "answer": "",
            "confidence": 5,
            "reasoning": "",
        }
        
        # Extract STATUS
        status_match = re.search(r'STATUS:\s*(Successful|Unsuccessful)', response, re.IGNORECASE)
        if status_match:
            result["output"] = status_match.group(1)
        
        # Extract FINAL_ANSWER
        answer_match = re.search(r'FINAL_ANSWER:\s*(.+?)(?=CONFIDENCE:|REASONING:|$)', response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        if conf_match:
            result["confidence"] = min(max(int(conf_match.group(1)), 0), 10)
        
        # Extract REASONING
        reason_match = re.search(r'REASONING:\s*(.+?)$', response, re.DOTALL)
        if reason_match:
            result["reasoning"] = reason_match.group(1).strip()
        
        # Fallback: if no structured output, use whole response as answer
        if not result["answer"]:
            result["answer"] = response.strip()
            # Estimate confidence based on response quality
            if len(response) > 200 and 'PMID' in response:
                result["confidence"] = 7
            elif len(response) > 100:
                result["confidence"] = 5
        
        return result
    
    async def summarize(self, state: GraphState) -> Dict[str, Any]:
        """
        Summarize plan execution and generate final answer.
        
        Args:
            state: Current graph state with step_outputs
        
        Returns:
            State updates with final_answer, final_confidence
        """
        question = state["original_question"]
        plan = state.get("plan", [])
        step_outputs = state.get("step_outputs", [])
        sources = state.get("sources", [])
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"[PLAN_SUMMARIZER] Summarizing {len(step_outputs)} step outputs")
        
        try:
            llm = self._get_llm()
            chain = self.prompt | llm | StrOutputParser()
            
            memory = self._format_step_memory(step_outputs)
            
            response = await chain.ainvoke({
                "question": question,
                "plan": str(plan),
                "memory": memory,
            })
            
            parsed = self._parse_summary_response(response)
            
            logger.info(f"[PLAN_SUMMARIZER] Confidence: {parsed['confidence']}/10")
            
            # Format final answer with sources
            final_answer = parsed["answer"]
            if sources:
                final_answer += f"\n\n**Sources:** {', '.join([f'PMID:{s}' for s in sources[:5]])}"
            final_answer += f"\n\n*Confidence: {parsed['confidence']}/10*"
            
            return {
                "final_answer": final_answer,
                "final_confidence": parsed["confidence"],
                "reasoning_trace": [{
                    "phase": "SUMMARIZE",
                    "thought": f"Plan {'succeeded' if parsed['output'] == 'Successful' else 'failed'} with confidence {parsed['confidence']}/10",
                    "details": parsed["reasoning"][:200] if parsed["reasoning"] else "",
                    "iteration": iteration,
                }],
            }
            
        except Exception as e:
            logger.error(f"[PLAN_SUMMARIZER] Failed: {e}")
            
            return {
                "final_answer": f"Unable to synthesize answer: {str(e)[:100]}",
                "final_confidence": 3,
                "reasoning_trace": [{
                    "phase": "SUMMARIZE",
                    "thought": f"Summarization failed: {str(e)[:50]}",
                    "iteration": iteration,
                }],
            }
    
    def summarize_sync(self, state: GraphState) -> Dict[str, Any]:
        """Synchronous version."""
        import asyncio
        return asyncio.run(self.summarize(state))


# =============================================================================
# Node Functions
# =============================================================================

_summarizer_agent: Optional[PlanSummarizerAgent] = None


def get_summarizer_agent() -> PlanSummarizerAgent:
    """Get or create the global summarizer agent."""
    global _summarizer_agent
    if _summarizer_agent is None:
        _summarizer_agent = PlanSummarizerAgent(use_local=True)
    return _summarizer_agent


async def summarizer_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node function for plan summarization."""
    agent = get_summarizer_agent()
    return await agent.summarize(state)


def summarizer_node_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous version."""
    agent = get_summarizer_agent()
    return agent.summarize_sync(state)
