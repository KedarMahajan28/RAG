import os
from typing import List, Dict, Any
from utils.config import TAVILY_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API. Falls back to simple search if not configured.
    """
    logger.info(f"Web search tool invoked for query: '{query}'")
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY is not set. Returning placeholder mock web search results.")
        return [
            {
                "title": "No api key",
                "content": f"This is a simulated web search result for: {query}. To get real web search results, please configure TAVILY_API_KEY in your .env file.",
                "url": "https://example.com/mock-search-1"
            }
        ]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        
        results = []
        for res in response.get("results", []):
            results.append({
                "title": res.get("title", ""),
                "content": res.get("content", ""),
                "url": res.get("url", "")
            })
        return results
    except Exception as e:
        logger.error(f"Error executing Tavily web search: {e}")
        return []
