from typing import List, Dict, Any
from pathlib import Path
from utils.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE
from utils.logger import get_logger
from utils.helpers import format_prompt

logger = get_logger(__name__)

class PlannerAgent:
    def __init__(self):
        self.model_name = LLM_MODEL
        self.temperature = TEMPERATURE
        self.prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"

    def plan(self, query: str) -> Dict[str, Any]:
        """Generates a query breakdown plan using the system prompt file and ChatGroq."""
        logger.info(f"Generating plan for: '{query}'")
        
        # Format the user prompt content
        user_prompt_content = format_prompt(str(self.prompt_path), query=query)
        
        # Rule-based fallback plan
        mock_plan = {
            "query": query,
            "steps": [
                {"step_id": 1, "task": f"Search vector database for concepts: {query}", "tool": "vector_search"},
                {"step_id": 2, "task": f"Search web resources for recent developments in: {query}", "tool": "web_search"},
                {"step_id": 3, "task": "Synthesize context and write the final response", "tool": "generator"}
            ]
        }

        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not configured. Returning fallback mock plan.")
            return mock_plan

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            import json

            llm = ChatGroq(model=self.model_name, temperature=self.temperature, groq_api_key=GROQ_API_KEY)
            
            system_prompt = (
                "You are an expert Query Planner Agent. Decompose the user query into sub-tasks. "
                "Output your response strictly as a JSON object with this key structure: "
                '{"query": "<original query>", "steps": [{"step_id": 1, "task": "<action description>", "tool": "<vector_search|web_search>"}]}'
            )
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt_content)
            ])
            
            try:
                # Remove possible markdown wrapper
                cleaned_content = response.content.strip().replace("```json", "").replace("```", "")
                plan_data = json.loads(cleaned_content)
                return plan_data
            except Exception as json_err:
                logger.error(f"Failed to parse LLM planner output as JSON: {response.content}. Error: {json_err}")
                return mock_plan
                
        except Exception as e:
            logger.error(f"Error invoking LLM PlannerAgent: {e}")
            return mock_plan
