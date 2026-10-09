import os
from typing import List
from utils.logger import get_logger



logger = get_logger(__name__)

_embeddings_instance = None

class EmbeddingsManager:
    def __init__(self):
        self.embeddings = None
        self._init_embeddings()

    def _init_embeddings(self):
        """Initializes the local SentenceTransformers embedding model."""
        global _embeddings_instance
        if _embeddings_instance is not None:
            self.embeddings = _embeddings_instance
            return

        logger.info("Initializing local SentenceTransformers Embeddings (all-MiniLM-L6-v2)...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from src.utils.config import HF_TOKEN
            
            model_kwargs = {}
            if HF_TOKEN:
                model_kwargs["token"] = HF_TOKEN

            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs=model_kwargs
            )
            _embeddings_instance = self.embeddings
        except Exception as e:
            logger.error(f"Failed to load local embeddings: {e}. Creating fallback mock embeddings.")
            # Simple dummy class for testing without dependencies installed
            class MockEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.1] * 384 for _ in texts]
                def embed_query(self, text: str) -> List[float]:
                    return [0.1] * 384
            self.embeddings = MockEmbeddings()
            _embeddings_instance = self.embeddings

    def get_embeddings(self):
        return self.embeddings
