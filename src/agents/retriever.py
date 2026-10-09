from typing import List, Dict, Any
from pathlib import Path
from tools.vector_search import search_vector_store
from tools.reranker import Reranker
from utils.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE
from utils.logger import get_logger
from utils.helpers import format_prompt

logger = get_logger(__name__)

class RetrieverAgent:
    def __init__(self):
        self.reranker = Reranker()
        self.model_name = LLM_MODEL
        self.temperature = TEMPERATURE
        self.prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "retriever.txt"

    def refine_query(self, task_description: str, original_query: str) -> str:
        """Uses LLM (Groq) to construct a database query based on the task description and context."""
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            logger.info("GROQ_API_KEY not configured. Using task description directly as retrieval query.")
            return task_description
        
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatGroq(model=self.model_name, temperature=self.temperature, groq_api_key=GROQ_API_KEY)
            
            user_prompt_content = format_prompt(
                str(self.prompt_path),
                task=task_description,
                context=original_query
            )
            
            system_prompt = (
                "You are an expert Retrieval Agent. Output ONLY the optimized, single-line keyword search query. "
                "Do not include any introductions, quotes, or explanations."
            )
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt_content)
            ])
            
            refined = response.content.strip().strip('"').strip("'")
            logger.info(f"Refined search query: '{refined}' (original task: '{task_description}')")
            return refined
        except Exception as e:
            logger.error(f"Error refining search query with LLM: {e}. Falling back to task description.")
            return task_description

    def retrieve(self, task_description: str, original_query: str) -> List[Dict[str, Any]]:
        """Retrieves and reranks documents from the vector database."""
        logger.info(f"Retrieval agent running for task: '{task_description}'")
        
        # 1. Refine the query
        search_query = self.refine_query(task_description, original_query)
        
        # 2. Fetch candidates from vector database
        candidates = search_vector_store(search_query, limit=10)
        if not candidates and search_query != original_query:
            logger.info("Search query yielded no results. Retrying with original query.")
            candidates = search_vector_store(original_query, limit=10)

        # 3. Rerank the candidates
        logger.info(f"Retrieved {len(candidates)} candidates. Reranking...")
        reranked = self.reranker.rerank(query=original_query, documents=candidates, top_n=3)
        
        return reranked
