# FILE: src/agents/planner.py
"""
Planner Agent for the Agentic RAG system.

Decomposes medical queries into PICO components and creates a search plan.
Combines MA-RAG planning patterns with medical domain expertise.

Example Usage:
    >>> from src.agents.planner import PlannerAgent
    >>> planner = PlannerAgent()
    >>> state = await planner.plan(state)
    >>> print(state["plan"])
    ['Search for melanoma treatment', 'Extract stage II specific data', ...]
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.orchestrator.state import AgentStep, GraphState, PlanOutput
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a medical research planning assistant specializing in evidence-based medicine.
Your task is to decompose complex medical questions into structured search plans using PICO methodology.

PICO Components:
- **P**opulation/Problem: Patient population, disease, condition
- **I**ntervention: Treatment, diagnostic test, exposure
- **C**omparison: Alternative treatment or placebo (if applicable)
- **O**utcome: Clinical outcomes, endpoints, results

For each question:
1. **Analyze** the question to identify key medical concepts
2. **Extract PICO** components from the question
3. **Create a plan** with ordered steps to answer the question

Guidelines:
- Keep plans concise (2-5 steps typically)
- Each step should be a searchable sub-question
- Consider: diagnosis, treatment, prognosis, etiology as question types
- Use medical terminology appropriate for PubMed search

If you have past failed attempts (in memory), adjust your strategy:
- Try different search terms or synonyms
- Break down into smaller sub-questions
- Consider alternative PICO framings
"""

PLANNER_HUMAN_PROMPT = """Question: {question}

Past attempts (if any):
{memory}

Create a search plan with PICO decomposition."""


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent:
    """
    Agent for decomposing medical queries into search plans.
    
    Uses LLM to extract PICO components and create ordered sub-tasks.
    Integrates with existing query_builder for PubMed optimization.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        use_local: bool = True,
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            llm: Optional LangChain LLM. If None, uses default based on use_local.
            use_local: If True, use local Ollama. If False, use Gemini.
        """
        self.llm = llm
        self.use_local = use_local
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", PLANNER_HUMAN_PROMPT),
        ])
        
        logger.info(f"PlannerAgent initialized (use_local={use_local})")
    
    def _get_llm(self) -> Any:
        """Get or create the LLM instance."""
        if self.llm is not None:
            return self.llm
        
        # Lazy import to avoid circular dependencies
        if self.use_local:
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model="phi4-mini",
                    base_url="http://localhost:11434",
                    temperature=0.3,
                )
                logger.info("Using local Ollama (phi4-mini)")
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}, falling back to Gemini")
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
                logger.info(f"Using Gemini ({settings.GEMINI_MODEL})")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                raise
        
        return self.llm
    
    def _format_memory(self, state: GraphState) -> str:
        """Format past attempts for the prompt."""
        memory_stack = state.get("memory_stack", [])
        past_attempts = state.get("past_attempts", [])
        
        if not memory_stack and not past_attempts:
            return "No previous attempts."
        
        parts = []
        
        # Add memory stack items
        for i, item in enumerate(memory_stack[-3:]):  # Last 3 items
            parts.append(f"- Attempt {i+1}: {item}")
        
        # Add reasoning from past attempts
        for attempt in past_attempts[-2:]:  # Last 2 attempts
            if "plan" in attempt:
                parts.append(f"- Previous plan: {attempt['plan'][:3]}...")
        
        return "\n".join(parts) if parts else "No previous attempts."
    
    def _extract_pico_from_plan(self, plan_output: PlanOutput) -> Dict[str, List[str]]:
        """Extract PICO query from plan output."""
        return {
            "population": plan_output.pico_population,
            "intervention": plan_output.pico_intervention,
            "modifiers": plan_output.pico_modifiers,
        }
    
    async def plan(self, state: GraphState) -> Dict[str, Any]:
        """
        Create a search plan for the given query.
        
        Args:
            state: Current graph state with original_question
        
        Returns:
            Updates to graph state including plan and pico_query
        """
        question = state["original_question"]
        memory = self._format_memory(state)
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"[PLANNER] Processing: {question[:50]}... (iteration {iteration})")
        
        try:
            llm = self._get_llm()
            
            # Create chain with structured output
            chain = self.prompt | llm.with_structured_output(PlanOutput)
            
            # Invoke chain
            result: PlanOutput = await chain.ainvoke({
                "question": question,
                "memory": memory,
            })
            
            logger.info(f"[PLANNER] Created plan with {len(result.steps)} steps")
            
            return {
                "plan": result.steps,
                "pico_query": self._extract_pico_from_plan(result),
                "current_step": AgentStep.RETRIEVE,
                "reasoning_trace": [{
                    "phase": "PLAN",
                    "thought": f"Decomposed into {len(result.steps)} steps",
                    "details": result.analysis[:200],
                    "iteration": iteration,
                }],
            }
            
        except Exception as e:
            logger.error(f"[PLANNER] Failed: {e}")
            
            # Fallback: simple plan based on question
            fallback_plan = self._create_fallback_plan(question)
            
            return {
                "plan": fallback_plan,
                "pico_query": {"population": [], "intervention": [], "modifiers": []},
                "current_step": AgentStep.RETRIEVE,
                "reasoning_trace": [{
                    "phase": "PLAN",
                    "thought": f"Fallback plan used due to error: {str(e)[:100]}",
                    "iteration": iteration,
                }],
            }
    
    def _create_fallback_plan(self, question: str) -> List[str]:
        """Create a simple fallback plan when LLM fails."""
        return [
            f"Search PubMed for: {question}",
            "Extract relevant findings from literature",
            "Synthesize answer based on evidence",
        ]
    
    def plan_sync(self, state: GraphState) -> Dict[str, Any]:
        """Synchronous version of plan for non-async contexts."""
        import asyncio
        return asyncio.run(self.plan(state))


# =============================================================================
# Node Function for LangGraph
# =============================================================================

# Global agent instance (lazy initialized)
_planner_agent: Optional[PlannerAgent] = None


def get_planner_agent() -> PlannerAgent:
    """Get or create the global planner agent."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent(use_local=True)
    return _planner_agent


async def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node function for planning.
    
    Args:
        state: Current graph state
    
    Returns:
        State updates from planner
    """
    agent = get_planner_agent()
    return await agent.plan(state)


def planner_node_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous version for non-async graphs."""
    agent = get_planner_agent()
    return agent.plan_sync(state)
