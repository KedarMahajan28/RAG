from typing import List, Dict, Any
from utils.config import COHERE_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

class Reranker:
    def __init__(self):
        self.cohere_client = None
        self.local_model = None
        self._init_reranker()

    def _init_reranker(self):
        # 1. Try initializing Cohere if configured
        if COHERE_API_KEY:
            try:
                import cohere
                self.cohere_client = cohere.Client(api_key=COHERE_API_KEY)
                logger.info("Cohere Reranker initialized successfully.")
                return
            except ImportError:
                logger.warning("Cohere package not installed. Trying local cross-encoder.")
        
        # 2. Try initializing local Cross-Encoder using sentence-transformers
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
            self.local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            logger.info("Local Cross-Encoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load local Cross-Encoder: {e}. Falling back to token-overlap scoring.")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Reranks documents using Cohere API, Local Cross-Encoder, or simple overlap fallback."""
        if not documents:
            return []

        # Route to Cohere if available
        if self.cohere_client:
            try:
                doc_contents = [doc["content"] for doc in documents]
                response = self.cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=doc_contents,
                    top_n=top_n
                )
                
                reranked_docs = []
                for result in response.results:
                    idx = result.index
                    doc = documents[idx].copy()
                    doc["rerank_score"] = result.relevance_score
                    reranked_docs.append(doc)
                return reranked_docs
            except Exception as e:
                logger.error(f"Cohere Reranking failed: {e}. Falling back...")

        # Route to local Cross-Encoder model if available
        if self.local_model:
            try:
                pairs = [[query, doc["content"]] for doc in documents]
                scores = self.local_model.predict(pairs)
                
                scored_docs = []
                for idx, score in enumerate(scores):
                    doc = documents[idx].copy()
                    doc["rerank_score"] = float(score)
                    scored_docs.append(doc)
                
                # Sort descending
                scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_docs[:top_n]
            except Exception as e:
                logger.error(f"Local Cross-Encoder reranking failed: {e}. Falling back to token overlap.")

        # Fallback keyword overlap heuristic
        logger.info("Executing simple token-overlap reranking...")
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in documents:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(overlap)
            scored_docs.append(doc_copy)
            
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_n]
