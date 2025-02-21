from litellm import rerank
import os

class Reranker:
    def __init__(self, model: str = "cohere/rerank-english-v3.0") -> None:
        self.model = model
        
    def rerank(self, query_text: str, query_results: list[dict], top_n: int = 3) -> list[dict]:
        """
        Rerank the query results using the Cohere reranking model.
        
        Args:
            query_text: The search query text
            query_results: List of dictionaries containing documents to rerank
            top_n: Number of top results to return
            
        Returns:
            List of reranked documents sorted by relevance score
        """
        documents = [doc.get('document', '') for doc in query_results]
        
        results = rerank(
            model=self.model,
            query=query_text,
            documents=documents,
            top_n=top_n
        )
        
        # Map scores back to original documents
        reranked = []
        for idx, score in enumerate(results['results']):
            doc = query_results[score['index']].copy()
            doc['relevance_score'] = score['relevance_score'] 
            reranked.append(doc)
            
        return reranked