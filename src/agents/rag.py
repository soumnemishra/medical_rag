from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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
        self.llm = ModelRegistry.get_llm(temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=QAAnswerFormat)
        
        system_prompt = QA_SYSTEM_PROMPT + "\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", QA_HUMAN_PROMPT)
        ])

    def query(self, state: RagState) -> Dict[str, Any]:
        """
        Execute RAG pipeline for the given question.
        """
        question = state["question"]
        
        # 1. Retrieval
        try:
            contexts, doc_ids = self.retriever(question)
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
            
            response = chain.invoke({
                "context": context_text,
                "question": question,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Format output for Graph
            return {
                "question": question,
                "documents": contexts,
                "doc_ids": doc_ids,
                "notes": [response.analysis],
                "final_raw_answer": response.model_dump()
            }
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            # Fallback
            fallback = QAAnswerFormat(
                analysis=f"Error: {e}",
                answer="Failed to generate answer due to error.",
                success="No",
                rating=0
            )
            return {
                "question": question,
                "documents": [],
                "doc_ids": [],
                "notes": ["Error during generation"],
                "final_raw_answer": fallback.model_dump()
            }

def rag_node(state: RagState) -> Dict[str, Any]:
    agent = RagAgent()
    return agent.query(state)
