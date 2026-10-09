from typing import List, Dict, Any
from rag.vectordb import VectorDB
from utils.logger import get_logger

logger = get_logger(__name__)

def search_vector_store(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search vector DB for documents match the query.
    """
    logger.info(f"Vector search tool invoked for query: '{query}'")
    try:
        db = VectorDB()
        results = db.similarity_search(query, k=limit)
        return results
    except Exception as e:
        logger.error(f"Error in vector search tool: {e}")
        return []
