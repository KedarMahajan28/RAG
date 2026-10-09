from typing import List, Dict, Any
from tools.web_search import search_web
from utils.logger import get_logger

logger = get_logger(__name__)

class ResearcherAgent:
    def __init__(self):
        pass

    def research(self, task_description: str) -> List[Dict[str, Any]]:
        """Performs web search to gather external information."""
        logger.info(f"Researcher agent running search for task: '{task_description}'")
        results = search_web(task_description, max_results=3)
        logger.info(f"Researcher agent retrieved {len(results)} search results.")
        return results
        
    def evaluate_source_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filters search results based on basic keyword matching or heuristic."""
        # Simple placeholder for evaluation logic
        filtered_results = []
        for result in results:
            content_lower = result.get("content", "").lower()
            query_terms = [t for t in query.lower().split() if len(t) > 3]
            score = sum(1 for term in query_terms if term in content_lower)
            
            # Simple metadata tagging
            result_copy = result.copy()
            result_copy["relevance_score"] = score
            filtered_results.append(result_copy)
            
        return filtered_results
