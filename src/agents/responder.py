from typing import List, Dict, Any
from pathlib import Path
from utils.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE
from utils.logger import get_logger
from utils.helpers import format_prompt

logger = get_logger(__name__)

class ResponderAgent:
    def __init__(self):
        self.model_name = LLM_MODEL
        self.temperature = TEMPERATURE
        self.prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "responder.txt"

    def generate(self, query: str, vector_context: List[Dict[str, Any]], web_context: List[Dict[str, Any]]) -> str:
        """Synthesizes the final answer using retrieved contexts, prompt file, and ChatGroq."""
        logger.info(f"Generating final response for query: '{query}'")
        
        # Format contexts into readable strings
        vector_str = "\n\n".join([
            f"[Source: {doc.get('metadata', {}).get('source', 'Unknown')} (Chunk {doc.get('metadata', {}).get('chunk_index', 0)})]\n{doc.get('content', '')}"
            for doc in vector_context
        ])
        
        web_str = "\n\n".join([
            f"[Web Source: {res.get('title', 'Unknown')} ({res.get('url', '')})]\n{res.get('content', '')}"
            for res in web_context
        ])

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY is not configured. Generating static/fallback response.")
            ans = f"Mocked synthesis for query: '{query}'\n\n"
            if vector_context:
                ans += f"Key VectorDB findings:\n" + "\n".join([f"- {c['content'][:150]}... [{c.get('metadata', {}).get('source', 'Unknown')}]" for c in vector_context]) + "\n\n"
            if web_context:
                ans += f"Key Web findings:\n" + "\n".join([f"- {w['content'][:150]}... ({w.get('url', '')})" for w in web_context]) + "\n\n"
            ans += "To enable full generative answers, please set GROQ_API_KEY in your .env."
            return ans

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatGroq(model=self.model_name, temperature=self.temperature, groq_api_key=GROQ_API_KEY)
            
            system_prompt = (
                "You are an expert Responder Agent. Synthesize the final answer to the user query based "
                "strictly on the provided local retrieved facts and web search context. "
                "Always cite sources by their name (e.g. [Source: doc.txt] or [Web Source: Title])."
            )
            
            user_prompt_content = format_prompt(
                str(self.prompt_path),
                query=query,
                context=vector_str if vector_str else "No relevant documents found.",
                web_context=web_str if web_str else "No relevant web resources found."
            )
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt_content)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM ResponderAgent: {e}")
            return f"Error synthesizing response. Source highlights:\nVector Store: {len(vector_context)} docs, Web Search: {len(web_context)} results."
