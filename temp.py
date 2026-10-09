from src.rag.vectordb import VectorDB
from src.rag.ingest import IngestionPipeline
from src.agents.graph import RAGGraph

print('VectorDB...'); v = VectorDB()
print('Pipeline...'); p = IngestionPipeline()
print('RAGGraph...'); g = RAGGraph()
print('All ok')