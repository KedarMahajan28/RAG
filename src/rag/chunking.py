from typing import List, Dict, Any

class ChunkingStrategy:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of specified chunk_size with overlap.
        Implements a simple character-level recursive split-like logic if Langchain is unavailable.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            return splitter.split_text(text)
        except ImportError:
            # Fallback character-based sliding window chunking
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += (self.chunk_size - self.chunk_overlap)
            return chunks

    def chunk_document(self, doc_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunks a document and appends incremental IDs and metadata."""
        chunks = self.split_text(doc_text)
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunked_docs.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        return chunked_docs
