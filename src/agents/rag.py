from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state.state import RagState, QAAnswerFormat
from src.prompts.templates import QA_SYSTEM_PROMPT, QA_HUMAN_PROMPT
from src.core.registry import ModelRegistry
from src.tools.retriever import RetrieverTool
import logging

logger = logging.getLogger(__name__)

class RagAgent:
    """
    Agent responsible for executing RAG on a single query.
    """
    
    def __init__(self, retriever_tool: RetrieverTool = None):
        self.retriever = retriever_tool or RetrieverTool()
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", QA_HUMAN_PROMPT)
        ])

    async def query(self, state: RagState) -> Dict[str, Any]:
        """
        Execute RAG pipeline for the given question.
        """
        question = state["question"]
        
        # 1. Retrieval
        # Check if documents are already provided (e.g., from Extractor)
        if state.get("documents"):
            contexts = state["documents"]
            doc_ids = state.get("doc_ids", [])
            logger.info("Using provided documents (skipping retrieval)")
        else:
            try:
                contexts, doc_ids = await self.retriever(question)
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                contexts, doc_ids = [], []
            
        if not contexts:
            context_text = "No relevant documents found."
            logger.warning(f"No documents found for query: {question}")
        else:
            context_text = "\n\n".join(contexts)

        # 2. Generation using LLM
        try:
            chain = self.prompt | self.llm | self.parser
            
            response = await chain.ainvoke({
                "context": context_text,
                "question": question
            })
            
            # response is a dict from JsonOutputParser
            # Ensure it has basic fields
            analysis = response.get("analysis", "No analysis")
            
            return {
                "question": question,
                "documents": contexts,
                "doc_ids": doc_ids,
                "notes": [analysis],
                "final_raw_answer": response
            }
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            fallback = {
                "analysis": f"Error: {e}",
                "answer": "Failed to generate answer due to error.",
                "success": "No",
                "rating": 0
            }
            return {
                "question": question,
                "documents": [],
                "doc_ids": [],
                "notes": ["Error during generation"],
                "final_raw_answer": fallback
            }

async def rag_node(state: RagState) -> Dict[str, Any]:
    agent = RagAgent()
    return await agent.query(state)
