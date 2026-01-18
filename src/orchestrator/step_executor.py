# FILE: src/orchestrator/step_executor.py
"""
Step Executor Subgraph for MA-RAG style plan execution.

Executes plan steps one at a time, tracking per-step confidence
and accumulating results. Supports two task types:
- search: RAG retrieval + extraction
- aggregate: Combine previous step outputs

Example Usage:
    >>> from src.orchestrator.step_executor import build_step_executor
    >>> executor = build_step_executor(retriever, extractor)
    >>> result = executor.invoke({"plan": [...], "original_question": "..."})
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

# Fix nested event loops
import nest_asyncio
nest_asyncio.apply()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from src.orchestrator.state import (
    PlanExecutorState, 
    StepOutput, 
    StepTask,
    QAAnswer,
    RetrievedDoc,
)
from src.pubmed_client import PubMedClient
from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates (MA-RAG style)
# =============================================================================

STEP_DEFINER_SYSTEM = """Given a plan and current step, decide the task type and specific query.

Task Types:
- "search": Query requires retrieving new information from literature
- "aggregate": Query requires combining/comparing previous step results

Rules:
- Be specific in the query - include all relevant details
- For aggregate: include key findings from previous steps in the query
- Never put "based on previous results" - be explicit
"""

STEP_DEFINER_HUMAN = """
Plan: {plan}
Current step: {cur_step}
Results from previous steps:
{memory}

Determine the task type (search/aggregate) and write a specific query for this step.
Format: TYPE: <type>
QUERY: <detailed query>
"""

AGGREGATE_SYSTEM = """You are a medical research assistant. Combine and synthesize information
from previous research steps to answer the current question.

Be concise and evidence-based. Cite sources (PMIDs) when available.
Rate your confidence from 0-10 based on evidence quality.
"""

AGGREGATE_HUMAN = """Question: {question}

Previous findings:
{context}

Synthesize an answer. Format:
ANSWER: <your answer>
CONFIDENCE: <0-10>
ANALYSIS: <brief reasoning>
"""


# =============================================================================
# Step Executor Agent
# =============================================================================

class StepExecutor:
    """
    Executes plan steps one at a time, MA-RAG style.
    
    Handles both search (RAG) and aggregate (synthesis) tasks.
    Tracks per-step confidence for backtracking decisions.
    """
    
    def __init__(
        self,
        pubmed_client: Optional[PubMedClient] = None,
        max_docs_per_step: int = 5,
    ):
        self.pubmed_client = pubmed_client or PubMedClient()
        self.max_docs_per_step = max_docs_per_step
        self._llm = None
        
        # Prompts
        self.step_definer_prompt = ChatPromptTemplate.from_messages([
            ("system", STEP_DEFINER_SYSTEM),
            ("human", STEP_DEFINER_HUMAN),
        ])
        
        self.aggregate_prompt = ChatPromptTemplate.from_messages([
            ("system", AGGREGATE_SYSTEM),
            ("human", AGGREGATE_HUMAN),
        ])
        
        logger.info("StepExecutor initialized")
    
    def _get_llm(self):
        """Get LLM with Ollama primary, Gemini fallback."""
        if self._llm is not None:
            return self._llm
        
        try:
            from langchain_ollama import OllamaLLM
            self._llm = OllamaLLM(
                model="phi4-mini",
                base_url="http://localhost:11434",
                temperature=0.2,
            )
            logger.info("StepExecutor using Ollama (phi4-mini)")
        except Exception as e:
            logger.warning(f"Ollama failed: {e}, using Gemini")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        
        return self._llm
    
    def _format_memory(self, step_outputs: List[StepOutput]) -> str:
        """Format previous step outputs for context."""
        if not step_outputs:
            return "No previous results yet."
        
        parts = []
        for i, step in enumerate(step_outputs):
            answer = step.get("answer", {})
            answer_text = answer.get("answer", "No answer") if isinstance(answer, dict) else str(answer)
            confidence = answer.get("confidence", 5) if isinstance(answer, dict) else 5
            parts.append(f"Step {i+1}: {step.get('task', '')}\nAnswer: {answer_text}\nConfidence: {confidence}/10")
        
        return "\n\n".join(parts)
    
    def _parse_step_task(self, response: str) -> Dict[str, str]:
        """Parse task type and query from LLM response."""
        import re
        
        task_type = "search"  # Default
        query = response.strip()
        
        # Extract TYPE
        type_match = re.search(r'TYPE:\s*(search|aggregate)', response, re.IGNORECASE)
        if type_match:
            task_type = type_match.group(1).lower()
        
        # Extract QUERY
        query_match = re.search(r'QUERY:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
        
        return {"type": task_type, "query": query}
    
    def _parse_aggregate_response(self, response: str) -> Dict[str, Any]:
        """Parse aggregate response."""
        import re
        
        answer = response.strip()
        confidence = 5
        analysis = ""
        
        # Extract ANSWER
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=CONFIDENCE:|ANALYSIS:|$)', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        if conf_match:
            confidence = min(max(int(conf_match.group(1)), 0), 10)
        
        # Extract ANALYSIS
        analysis_match = re.search(r'ANALYSIS:\s*(.+?)$', response, re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        return {
            "answer": answer,
            "confidence": confidence,
            "analysis": analysis,
            "success": "yes" if confidence >= 5 else "no",
        }
    
    async def define_step_task(self, state: PlanExecutorState) -> Dict[str, Any]:
        """
        Define what task to execute for the current step.
        
        Checks if we should stop or continue, and determines task type.
        """
        idx = state.get("current_step_idx", 0)
        plan = state.get("plan", [])
        step_outputs = state.get("step_outputs", [])
        
        logger.info(f"[STEP_EXECUTOR] Defining task for step {idx+1}/{len(plan)}")
        
        # Check stopping conditions
        if idx >= len(plan):
            logger.info("[STEP_EXECUTOR] All steps complete")
            return {"stop": True}
        
        # Check for previous failure
        if step_outputs and isinstance(step_outputs[-1], dict):
            last_answer = step_outputs[-1].get("answer", {})
            if isinstance(last_answer, dict) and last_answer.get("success", "yes").lower() == "no":
                logger.info("[STEP_EXECUTOR] Previous step failed, stopping")
                return {"stop": True}
        
        # Define task for current step
        cur_step = plan[idx]
        memory = self._format_memory(step_outputs)
        
        try:
            llm = self._get_llm()
            chain = self.step_definer_prompt | llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "plan": str(plan),
                "cur_step": cur_step,
                "memory": memory,
            })
            
            task_info = self._parse_step_task(response)
            
            step_task = StepTask(
                task_type=task_info["type"],
                task_query=task_info["query"],
            )
            
            logger.info(f"[STEP_EXECUTOR] Task defined: {task_info['type']} - {task_info['query'][:50]}...")
            
            return {
                "step_tasks": [step_task],
                "stop": False,
            }
            
        except Exception as e:
            logger.error(f"[STEP_EXECUTOR] Task definition failed: {e}")
            # Fallback: use step directly as search query
            return {
                "step_tasks": [StepTask(task_type="search", task_query=cur_step)],
                "stop": False,
            }
    
    async def execute_step(self, state: PlanExecutorState) -> Dict[str, Any]:
        """
        Execute the current step task.
        
        Routes to search or aggregate based on task type.
        """
        step_tasks = state.get("step_tasks", [])
        if not step_tasks:
            return {"stop": True}
        
        current_task = step_tasks[-1]
        idx = state.get("current_step_idx", 0)
        
        if isinstance(current_task, dict):
            task_type = current_task.get("task_type", "search")
            task_query = current_task.get("task_query", "")
        else:
            task_type = current_task.task_type
            task_query = current_task.task_query
        
        logger.info(f"[STEP_EXECUTOR] Executing step {idx+1}: {task_type}")
        
        if task_type == "aggregate":
            result = await self._execute_aggregate(state, task_query)
        else:
            result = await self._execute_search(state, task_query)
        
        # Increment step index
        result["current_step_idx"] = idx + 1
        
        return result
    
    async def _execute_search(self, state: PlanExecutorState, query: str) -> Dict[str, Any]:
        """Execute a search (RAG) task."""
        logger.info(f"[STEP_EXECUTOR] Searching: {query[:50]}...")
        
        try:
            # Retrieve documents
            docs = self.pubmed_client.search(query)[:self.max_docs_per_step]
            
            # Convert to RetrievedDoc format
            retrieved_docs = []
            doc_ids = []
            for doc in docs:
                retrieved_docs.append(RetrievedDoc(
                    pmid=doc.pmid,
                    title=doc.title,
                    abstract=doc.abstract,
                    year=doc.year,
                    relevance_score=1.0,
                ))
                doc_ids.append(doc.pmid)
            
            logger.info(f"[STEP_EXECUTOR] Found {len(docs)} documents")
            
            # Extract relevant info (simple for now)
            notes = []
            for doc in docs[:3]:
                notes.append(f"[PMID:{doc.pmid}] {doc.title[:100]}")
            
            # Create step output
            step_output = StepOutput(
                step_name=f"Step {state.get('current_step_idx', 0) + 1}",
                task=query,
                answer={
                    "answer": f"Found {len(docs)} relevant articles",
                    "analysis": "; ".join(notes),
                    "success": "yes" if docs else "no",
                    "confidence": 7 if docs else 3,
                    "sources": doc_ids,
                },
                docs_used=doc_ids,
            )
            
            return {
                "step_outputs": [step_output],
                "retrieved_docs": retrieved_docs,
                "extracted_notes": notes,
            }
            
        except Exception as e:
            logger.error(f"[STEP_EXECUTOR] Search failed: {e}")
            return {
                "step_outputs": [StepOutput(
                    step_name=f"Step {state.get('current_step_idx', 0) + 1}",
                    task=query,
                    answer={
                        "answer": f"Search failed: {str(e)[:50]}",
                        "success": "no",
                        "confidence": 0,
                    },
                    docs_used=[],
                )],
            }
    
    async def _execute_aggregate(self, state: PlanExecutorState, query: str) -> Dict[str, Any]:
        """Execute an aggregate task (combine previous results)."""
        logger.info(f"[STEP_EXECUTOR] Aggregating: {query[:50]}...")
        
        try:
            # Build context from previous steps
            step_outputs = state.get("step_outputs", [])
            context = self._format_memory(step_outputs)
            
            llm = self._get_llm()
            chain = self.aggregate_prompt | llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "question": query,
                "context": context,
            })
            
            parsed = self._parse_aggregate_response(response)
            
            step_output = StepOutput(
                step_name=f"Step {state.get('current_step_idx', 0) + 1} (Aggregate)",
                task=query,
                answer=parsed,
                docs_used=[],
            )
            
            return {"step_outputs": [step_output]}
            
        except Exception as e:
            logger.error(f"[STEP_EXECUTOR] Aggregate failed: {e}")
            return {
                "step_outputs": [StepOutput(
                    step_name=f"Step {state.get('current_step_idx', 0) + 1}",
                    task=query,
                    answer={"answer": str(e), "success": "no", "confidence": 0},
                    docs_used=[],
                )],
            }


# =============================================================================
# Build Step Executor Subgraph
# =============================================================================

def build_step_executor(pubmed_client: Optional[PubMedClient] = None) -> StateGraph:
    """
    Build the step executor subgraph.
    
    Flow: task_definer → single_task_execute → (loop or END)
    """
    executor = StepExecutor(pubmed_client=pubmed_client)
    
    # Sync wrappers for LangGraph
    def define_task(state: PlanExecutorState) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(executor.define_step_task(state))
    
    def execute_task(state: PlanExecutorState) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(executor.execute_step(state))
    
    def should_continue(state: PlanExecutorState) -> str:
        if state.get("stop", False):
            return END
        return "execute_step"
    
    # Build graph
    graph = StateGraph(PlanExecutorState)
    
    graph.add_node("define_task", define_task)
    graph.add_node("execute_step", execute_task)
    
    graph.add_edge(START, "define_task")
    graph.add_conditional_edges("define_task", should_continue)
    graph.add_edge("execute_step", "define_task")  # Loop back
    
    return graph.compile()


def create_executor_initial_state(
    question: str, 
    plan: List[str]
) -> PlanExecutorState:
    """Create initial state for step executor."""
    return PlanExecutorState(
        original_question=question,
        plan=plan,
        current_step_idx=0,
        step_tasks=[],
        step_outputs=[],
        retrieved_docs=[],
        extracted_notes=[],
        stop=False,
    )
