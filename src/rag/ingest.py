import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from utils.logger import get_logger
from rag.chunking import ChunkingStrategy
from rag.vectordb import VectorDB

logger = get_logger(__name__)


@dataclass
class FileIngestionResult:
    file_name: str
    success: bool
    chunks_ingested: int = 0
    error: Optional[str] = None


@dataclass
class IngestionSummary:
    results: List[FileIngestionResult] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> List[FileIngestionResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[FileIngestionResult]:
        return [r for r in self.results if not r.success]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks_ingested for r in self.results)


class IngestionPipeline:
    def __init__(self):
        self.chunker = ChunkingStrategy()
        self.db = VectorDB()

    def parse_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is not installed. Run: pip install pypdf")

        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def parse_text_or_md(self, file_path: Path) -> str:
        """Read plain text or markdown files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def ingest_file(self, file_path: Path) -> FileIngestionResult:
        """Ingest a single file into the vector store."""
        file_name = file_path.name
        suffix = file_path.suffix.lower()
        logger.info(f"Ingesting file: {file_name} (type: {suffix})")

        if suffix == ".pdf":
            content = self.parse_pdf(file_path)
        elif suffix in [".txt", ".md"]:
            content = self.parse_text_or_md(file_path)
        else:
            msg = f"Unsupported file type: {suffix}"
            logger.warning(f"{msg} — skipping {file_name}")
            return FileIngestionResult(file_name=file_name, success=False, error=msg)

        if not content.strip():
            msg = "Extracted content is empty"
            logger.warning(f"{msg} — skipping {file_name}")
            return FileIngestionResult(file_name=file_name, success=False, error=msg)

        metadata = {
            "source": file_name,
            "file_path": str(file_path),
            "file_type": suffix,
        }

        chunks = self.chunker.chunk_document(content, metadata)
        logger.info(f"Split {file_name} into {len(chunks)} chunks.")

        if not chunks:
            msg = "Chunking produced no chunks"
            return FileIngestionResult(file_name=file_name, success=False, error=msg)

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{file_name}_chunk_{c['metadata']['chunk_index']}" for c in chunks]

        self.db.add_documents(texts, metadatas, ids)

        processed_path = DATA_PROCESSED_DIR / file_name
        with open(processed_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(content)
        logger.info(f"Saved processed backup to {processed_path}")

        return FileIngestionResult(file_name=file_name, success=True, chunks_ingested=len(chunks))

    def clear(self):
        """Reset the vector store. Useful for re-ingestion from the UI."""
        self.db.reset()
        logger.info("Vector store cleared.")

    def run(self) -> IngestionSummary:
        """Run the ingestion pipeline for all files in DATA_RAW_DIR."""
        logger.info(f"Scanning: {DATA_RAW_DIR}")
        summary = IngestionSummary()

        raw_files = [
            f for f in Path(DATA_RAW_DIR).rglob("*")
            if f.is_file() and f.name != ".gitkeep"
        ]

        if not raw_files:
            logger.info("No raw documents found.")
            return summary

        logger.info(f"Found {len(raw_files)} file(s).")

        for file_path in raw_files:
            result = self.ingest_file(file_path)
            summary.results.append(result)
            if not result.success:
                logger.error(f"Failed to ingest {result.file_name}: {result.error}")

        logger.info(
            f"Ingestion complete — {len(summary.successful)}/{summary.total_files} files, "
            f"{summary.total_chunks} total chunks."
        )
        return summary


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    summary = pipeline.run()
    for r in summary.failed:
        print(f"FAILED: {r.file_name} — {r.error}")