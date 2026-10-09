from typing import TypedDict, List, Dict, Any, Union
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.researcher import ResearcherAgent
from agents.responder import ResponderAgent
from utils.logger import get_logger

logger = get_logger(__name__)

# State definition
class AgentState(TypedDict):
    query: str
    plan: Dict[str, Any]
    vector_context: List[Dict[str, Any]]
    web_context: List[Dict[str, Any]]
    response: str
    next_step: str

class RAGGraph:
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.researcher = ResearcherAgent()
        self.responder = ResponderAgent()

    def plan_query(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Plan the query breakdown."""
        logger.info("--- NODE: Plan Query ---")
        plan_data = self.planner.plan(state["query"])
        return {
            "plan": plan_data,
            "next_step": "retrieve_or_search"
        }

    def gather_information(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Gather information from vector database and web search."""
        logger.info("--- NODE: Gather Information ---")
        plan = state["plan"]
        vector_results = []
        web_results = []

        for step in plan.get("steps", []):
            tool = step.get("tool")
            task = step.get("task")
            if tool == "vector_search":
                res = self.retriever.retrieve(task, state["query"])
                vector_results.extend(res)
            elif tool == "web_search":
                res = self.researcher.research(task)
                # Filter/evaluate relevance
                filtered = self.researcher.evaluate_source_relevance(res, state["query"])
                web_results.extend(filtered)

        return {
            "vector_context": vector_results,
            "web_context": web_results,
            "next_step": "generate_response"
        }

    def generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Generate the final response."""
        logger.info("--- NODE: Generate Response ---")
        response = self.responder.generate(
            query=state["query"],
            vector_context=state["vector_context"],
            web_context=state["web_context"]
        )
        return {
            "response": response,
            "next_step": "end"
        }

    def execute_compiled_graph(self, query: str) -> Dict[str, Any]:
        """
        Builds the LangGraph and executes it.
        If LangGraph import fails, it executes as a fallback sequential Python workflow.
        """
        initial_state: AgentState = {
            "query": query,
            "plan": {},
            "vector_context": [],
            "web_context": [],
            "response": "",
            "next_step": "plan_query"
        }

        try:
            from langgraph.graph import StateGraph, END
            
            # Initialize Graph
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("plan_query", self.plan_query)
            workflow.add_node("gather_information", self.gather_information)
            workflow.add_node("generate_response", self.generate_response)
            
            # Setup Entry point
            workflow.set_entry_point("plan_query")
            
            # Setup Conditional / Direct edges
            workflow.add_edge("plan_query", "gather_information")
            workflow.add_edge("gather_information", "generate_response")
            workflow.add_edge("generate_response", END)
            
            # Compile graph
            app = workflow.compile()
            logger.info("Successfully compiled LangGraph workflow.")
            
            # Run graph
            final_state = app.invoke(initial_state)
            return final_state

        except Exception as e:
            logger.warning(f"LangGraph compile failed or package not installed ({e}). Running custom fallback runner.")
            # Fallback workflow execution engine
            state = initial_state.copy()
            
            # Run plan_query node
            updates = self.plan_query(state)
            state.update(updates)
            
            # Run gather_information node
            updates = self.gather_information(state)
            state.update(updates)
            
            # Run generate_response node
            updates = self.generate_response(state)
            state.update(updates)
            
            return state

if __name__ == "__main__":
    graph = RAGGraph()
    res = graph.execute_compiled_graph("What is agentic RAG and why use ChromaDB?")
    print("\nResult:")
    print(res["response"])
