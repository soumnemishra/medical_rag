# FILE: src/agents/retriever.py
"""
Retriever Agent for the Agentic RAG system.

Performs hybrid retrieval combining:
- Live PubMed search (via existing pubmed_client)
- PICO-optimized queries (via existing query_builder)

Example Usage:
    >>> from src.agents.retriever import RetrieverAgent
    >>> retriever = RetrieverAgent()
    >>> state = await retriever.retrieve(state)
    >>> print(len(state["retrieved_docs"]))
    10
"""

import logging
from typing import Any, Dict, List, Optional

from src.orchestrator.state import AgentStep, GraphState, RetrievedDoc
from src.pubmed_client import PubMedClient, RetrievedDocument
from src.query_builder import PubMedQueryBuilder, PICOQuery

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Agent for retrieving medical literature from PubMed.
    
    Integrates with existing PubMedClient and QueryBuilder
    for domain-optimized retrieval.
    """
    
    def __init__(
        self,
        pubmed_client: Optional[PubMedClient] = None,
        query_builder: Optional[PubMedQueryBuilder] = None,
        max_results: int = 10,
    ):
        """
        Initialize the Retriever Agent.
        
        Args:
            pubmed_client: Optional PubMed client instance
            query_builder: Optional PICO query builder instance
            max_results: Maximum documents to retrieve
        """
        self.pubmed_client = pubmed_client or PubMedClient()
        self.query_builder = query_builder or PubMedQueryBuilder()
        self.max_results = max_results
        
        logger.info(f"RetrieverAgent initialized (max_results={max_results})")
    
    def _build_pico_query(self, state: GraphState) -> str:
        """
        Build a PubMed query.
        
        For now, uses the original question directly for reliability.
        PICO optimization can be re-enabled once planner outputs are validated.
        """
        question = state["original_question"]
        logger.info(f"Using simple search with question: {question[:50]}...")
        return question
    
    def _convert_to_retrieved_doc(self, doc: RetrievedDocument) -> RetrievedDoc:
        """Convert PubMed document to state-compatible format."""
        return RetrievedDoc(
            pmid=doc.pmid,
            title=doc.title,
            abstract=doc.abstract,
            year=doc.year,
            relevance_score=1.0,  # PubMed doesn't provide scores directly
        )
    
    async def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        Retrieve relevant documents from PubMed.
        
        Args:
            state: Current graph state with plan and pico_query
        
        Returns:
            State updates including retrieved_docs
        """
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"[RETRIEVER] Starting retrieval (iteration {iteration})")
        
        try:
            # Build query from PICO or fallback
            query = self._build_pico_query(state)
            
            # Also try plan steps if available
            plan_queries = []
            for step in state.get("plan", [])[:2]:  # First 2 steps
                if "search" in step.lower():
                    plan_queries.append(step)
            
            # Perform primary search
            documents = self.pubmed_client.search(query)
            
            logger.info(f"[RETRIEVER] Found {len(documents)} documents")
            
            # Convert to state format
            retrieved_docs = [
                self._convert_to_retrieved_doc(doc) 
                for doc in documents[:self.max_results]
            ]
            
            # Extract PMIDs for sources
            sources = [doc.pmid for doc in documents[:self.max_results]]
            
            return {
                "retrieved_docs": retrieved_docs,
                "sources": sources,
                "current_step": AgentStep.EXTRACT,
                "reasoning_trace": [{
                    "phase": "RETRIEVE",
                    "thought": f"Retrieved {len(retrieved_docs)} documents from PubMed",
                    "details": f"Query: {query[:150]}...",
                    "iteration": iteration,
                }],
            }
            
        except Exception as e:
            logger.error(f"[RETRIEVER] Failed: {e}")
            
            return {
                "retrieved_docs": [],
                "sources": [],
                "current_step": AgentStep.EXTRACT,
                "reasoning_trace": [{
                    "phase": "RETRIEVE",
                    "thought": f"Retrieval failed: {str(e)[:100]}",
                    "iteration": iteration,
                }],
            }
    
    def retrieve_sync(self, state: GraphState) -> Dict[str, Any]:
        """Synchronous version for non-async contexts."""
        import asyncio
        return asyncio.run(self.retrieve(state))


# =============================================================================
# Node Function for LangGraph
# =============================================================================

_retriever_agent: Optional[RetrieverAgent] = None


def get_retriever_agent() -> RetrieverAgent:
    """Get or create the global retriever agent."""
    global _retriever_agent
    if _retriever_agent is None:
        _retriever_agent = RetrieverAgent()
    return _retriever_agent


async def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node function for retrieval.
    
    Args:
        state: Current graph state
    
    Returns:
        State updates with retrieved documents
    """
    agent = get_retriever_agent()
    return await agent.retrieve(state)


def retriever_node_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous version for non-async graphs."""
    agent = get_retriever_agent()
    return agent.retrieve_sync(state)
