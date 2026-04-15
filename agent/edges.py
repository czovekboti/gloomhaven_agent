from langgraph.graph import END
from agent.state import AppState
from agent.utils import cfg

max_attempts = cfg["agent"]["max_generation_attempts"] 
def route_relevance(state: AppState) -> str:
    if state.get("web_search_needed"):
        return "web_search_retriever"
    return "final_response_generator"

def route_evaluation(state: AppState) -> str:
    attempts = state.get("generation_attempts", 0)
    if state.get("answer_is_good", True) or attempts >= max_attempts:
        return END
    if attempts == 1:
        return "query_rewriter"
    return "web_search_retriever"

def route_final_response(state: AppState) -> str:
    # If max attempts reached, end the graph execution. Otherwise, route to evaluation.
    if state.get("generation_attempts", max_attempts) >= max_attempts or max_attempts < 1:
        return END
    else:
        return "answer_evaluator"