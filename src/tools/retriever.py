from typing import List, Tuple
from src.pubmed_client import PubMedClient
import logging

logger = logging.getLogger(__name__)

class RetrieverTool:
    """
    Adapter for PubMedClient to be used in MA-RAG pipeline.
    """
    
    def __init__(self, top_k: int = 5):
        self.client = PubMedClient(max_results=top_k)
        self.top_k = top_k
        
    def __call__(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Search PubMed and return text content and IDs.
        
        Returns:
            Tuple[List[str], List[str]]: (List of context strings, List of PMIDs)
        """
        try:
            # Clean query for better search
            cleaned_query = query.strip()
            
            docs = self.client.search(cleaned_query, max_results=self.top_k)
            
            list_docs = []
            list_doc_ids = []
            
            for doc in docs:
                # Format context clearly for the LLM
                context = f"Title: {doc.title}\nAbstract: {doc.abstract}\nDate: {doc.year}"
                list_docs.append(context)
                list_doc_ids.append(doc.pmid)
                
            return list_docs, list_doc_ids
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return [], []
