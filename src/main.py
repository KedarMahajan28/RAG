import sys
from pathlib import Path
# Add project root to sys.path to resolve module import issues when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from src.agents.graph import RAGGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_query(query: str):
    """Executes the Agentic RAG workflow for a given query."""
    logger.info(f"Starting execution of Agentic RAG workflow...")
    logger.info(f"User Query: '{query}'")
    
    graph = RAGGraph()
    state = graph.execute_compiled_graph(query)
    
    print("\n" + "="*50)
    print("FINAL AGENTIC RESPONSE:")
    print("="*50)
    print(state.get("response", "No response generated."))
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG CLI Test Runner")
    parser.add_argument("query", type=str, nargs="?", help="The user query to search/respond to")
    args = parser.parse_args()

    # Default query if none provided
    query = args.query
    if not query:
        query = "Explain how LangGraph controls agent state in RAG pipelines."

    run_query(query)

if __name__ == "__main__":
    main()
