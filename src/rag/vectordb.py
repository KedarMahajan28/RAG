import chromadb
from typing import List, Dict, Any
from utils.config import CHROMA_DB_DIR
from utils.logger import get_logger
from rag.embeddings import EmbeddingsManager
from chromadb.config import Settings

logger = get_logger(__name__)

class VectorDB:
    def __init__(self, collection_name: str = "rag_documents"):
        self.chroma_dir = CHROMA_DB_DIR
        self.collection_name = collection_name
        self.embeddings_manager = EmbeddingsManager()
        self.embeddings = self.embeddings_manager.get_embeddings()
        self.client = chromadb.PersistentClient( path=self.chroma_dir, settings=Settings( anonymized_telemetry=False ) )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.info(f"Connected to ChromaDB at: {self.chroma_dir}, collection: {self.collection_name}")

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Adds documents to the ChromaDB collection."""
        if not texts:
            logger.warning("Empty document list provided to add_documents.")
            return
        
        logger.info(f"Generating embeddings and adding {len(texts)} chunks to ChromaDB...")
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Successfully added documents.")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs a similarity search."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted_results = []
        if results and results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0] if results['metadatas'] else [None] * len(docs)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(docs)
            ids = results['ids'][0]
            
            for i in range(len(docs)):
                formatted_results.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "score": distances[i]
                })
        return formatted_results
