# FILE: src/agents/extractor.py
"""
Extractor Agent for the Agentic RAG system.

Filters noise from retrieved documents, extracting only relevant information.
Based on MA-RAG's extract pattern for improved answer quality.

Example Usage:
    >>> from src.agents.extractor import ExtractorAgent
    >>> extractor = ExtractorAgent()
    >>> state = await extractor.extract(state)
    >>> print(state["extracted_notes"])
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.orchestrator.state import AgentStep, GraphState, RetrievedDoc

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

EXTRACTOR_SYSTEM_PROMPT = """You are a medical research assistant specializing in extracting relevant information from scientific abstracts.

Your task is to:
1. Read the passage carefully
2. Identify information relevant to the question
3. Extract key findings, statistics, conclusions
4. Remove irrelevant background or methodology details

Output Format:
- List the relevant findings as bullet points
- Include specific data (percentages, p-values, outcomes) when present
- Note the study type if mentioned (RCT, meta-analysis, cohort)
- If no relevant information found, state: "No relevant information in this document."

Be concise but complete. Focus on answering the clinical question."""

EXTRACTOR_HUMAN_PROMPT = """Question: {question}

Passage (PMID:{pmid}):
{passage}

Extract relevant information:"""


class ExtractorAgent:
    """
    Agent for extracting relevant information from documents.
    
    Filters noise and extracts key findings to improve answer quality.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        use_local: bool = True,
        max_docs_to_extract: int = 5,
    ):
        """
        Initialize the Extractor Agent.
        
        Args:
            llm: Optional LangChain LLM
            use_local: If True, use local Ollama
            max_docs_to_extract: Maximum documents to process
        """
        self.llm = llm
        self.use_local = use_local
        self.max_docs_to_extract = max_docs_to_extract
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTOR_SYSTEM_PROMPT),
            ("human", EXTRACTOR_HUMAN_PROMPT),
        ])
        
        logger.info(f"ExtractorAgent initialized (max_docs={max_docs_to_extract})")
    
    def _get_llm(self) -> Any:
        """Get or create the LLM instance with Ollama primary, Gemini fallback."""
        if self.llm is not None:
            return self.llm
        
        if self.use_local:
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(
                    model="phi4-mini",
                    base_url="http://localhost:11434",
                    temperature=0.1,  # Low temp for extraction
                )
                logger.info("Extractor using local Ollama (phi4-mini)")
            except Exception as e:
                logger.warning(f"Ollama failed: {e}, using Gemini")
                self.use_local = False
                return self._get_llm()
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                logger.info("Extractor using Gemini")
            except Exception as e:
                logger.error(f"Gemini also failed: {e}")
                raise
        
        return self.llm
    
    async def _extract_from_doc(
        self,
        doc: RetrievedDoc,
        question: str,
    ) -> str:
        """Extract relevant information from a single document."""
        try:
            llm = self._get_llm()
            chain = self.prompt | llm | StrOutputParser()
            
            result = await chain.ainvoke({
                "question": question,
                "pmid": doc["pmid"],
                "passage": doc["abstract"][:2000],  # Limit length
            })
            
            return f"[PMID:{doc['pmid']}] {result}"
            
        except Exception as e:
            logger.error(f"Extraction failed for PMID:{doc['pmid']}: {e}")
            return f"[PMID:{doc['pmid']}] Extraction failed"
    
    async def extract(self, state: GraphState) -> Dict[str, Any]:
        """
        Extract relevant information from retrieved documents.
        
        Args:
            state: Current graph state with retrieved_docs
        
        Returns:
            State updates with extracted_notes
        """
        docs = state.get("retrieved_docs", [])
        question = state["original_question"]
        iteration = state.get("iteration_count", 0)
        
        logger.info(f"[EXTRACTOR] Processing {len(docs)} documents")
        
        if not docs:
            return {
                "extracted_notes": [],
                "current_step": AgentStep.SYNTHESIZE,
                "reasoning_trace": [{
                    "phase": "EXTRACT",
                    "thought": "No documents to extract from",
                    "iteration": iteration,
                }],
            }
        
        # Extract from top N documents
        extracted_notes = []
        for doc in docs[:self.max_docs_to_extract]:
            note = await self._extract_from_doc(doc, question)
            extracted_notes.append(note)
        
        logger.info(f"[EXTRACTOR] Extracted {len(extracted_notes)} notes")
        
        return {
            "extracted_notes": extracted_notes,
            "current_step": AgentStep.SYNTHESIZE,
            "reasoning_trace": [{
                "phase": "EXTRACT",
                "thought": f"Extracted relevant info from {len(extracted_notes)} documents",
                "iteration": iteration,
            }],
        }
    
    def extract_sync(self, state: GraphState) -> Dict[str, Any]:
        """Synchronous version."""
        import asyncio
        return asyncio.run(self.extract(state))


# =============================================================================
# Node Function
# =============================================================================

_extractor_agent: Optional[ExtractorAgent] = None


def get_extractor_agent() -> ExtractorAgent:
    """Get or create the global extractor agent."""
    global _extractor_agent
    if _extractor_agent is None:
        _extractor_agent = ExtractorAgent()
    return _extractor_agent


async def extractor_node(state: GraphState) -> Dict[str, Any]:
    """LangGraph node function for extraction."""
    agent = get_extractor_agent()
    return await agent.extract(state)


def extractor_node_sync(state: GraphState) -> Dict[str, Any]:
    """Synchronous version."""
    agent = get_extractor_agent()
    return agent.extract_sync(state)
