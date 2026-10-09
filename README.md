# Agentic RAG (Retrieval-Augmented Generation) Pipeline

A modular, production-ready implementation of an Agentic RAG system built on top of LangGraph, ChromaDB, Tavily, Ragas, and Streamlit.

## Directory Structure

```
RAG/
├── src/
│   ├── agents/
│   │   ├── graph.py           # LangGraph workflow & routing logic
│   │   ├── retriever.py       # Retrieval agent
│   │   ├── planner.py         # Query planning agent
│   │   ├── researcher.py      # Web search/research agent
│   │   └── responder.py       # Final answer generation agent
│   │
│   ├── tools/
│   │   ├── vector_search.py   # Chroma retrieval tool
│   │   ├── web_search.py      # Tavily search tool
│   │   └── reranker.py        # Cohere/Cross-encoder reranking tool
│   │
│   ├── rag/
│   │   ├── ingest.py          # Document ingestion pipeline
│   │   ├── chunking.py        # Text chunking strategies
│   │   ├── embeddings.py      # Embedding generation
│   │   └── vectordb.py        # Vector database setup
│   │
│   ├── prompts/
│   │   ├── planner.txt        # System prompt for planning agent
│   │   ├── retriever.txt      # System prompt for retrieval agent
│   │   └── responder.txt      # System prompt for responder agent
│   │
│   ├── eval/
│   │   └── evaluator.py       # RAGAS evaluation framework
│   │
│   ├── utils/
│   │   ├── config.py          # Environment configs & settings
│   │   ├── logger.py          # Loguru unified logging setup
│   │   └── helpers.py         # Utility functions
│   │
│   ├── app.py                 # Streamlit UI dashboard
│   └── main.py                # Main execution entry point (CLI test)
│
├── data/
│   ├── raw/                   # Original source documents
│   └── processed/             # Chunked / prepared documents
│
├── vectorstore/               # Persistent ChromaDB storage directory
│
├── requirements.txt           # Project dependencies
├── .env                       # Environment configuration (API keys)
└── README.md                  # Project documentation (this file)
```

## Setup & Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   COHERE_API_KEY=your_cohere_key
   ```

3. Place raw documents in the `data/raw/` directory.

4. Run document ingestion:
   ```bash
   python -m src.rag.ingest
   ```

5. Launch the Streamlit dashboard:
   ```bash
   streamlit run src/app.py
   ```

6. Run evaluation:
   ```bash
   python -m src.eval.evaluator
   ```
