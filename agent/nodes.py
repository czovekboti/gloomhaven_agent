import json
from agent.state import AppState, AgentResponse
from agent import utils
from agent import prompts

def query_rewriter(state: AppState, llm=None) -> AppState:
    """Entry node: rewrites user query for better web and rag retrival"""
    chat_history = utils._format_recent_messages(state.get("messages", []))
    prompt = prompts.REWRITE_PROMPT.format(question=state["current_input"], chat_history=chat_history)
    fallback = {"optimized_query": state["current_input"]}
    parsed = utils._invoke_and_parse_json(llm, prompt, fallback)
    state["search_query"] = parsed.get("optimized_query", state["current_input"])
    return state

def rag_retriever(state: AppState, rag_service=None) -> AppState:
    query = state.get("search_query", state["current_input"])
    chunks = rag_service.retrieve_chunks(query, n_results=utils.cfg["rag"]["n_results"])
    state["retrieved_chunks"] = chunks 
    return state

def web_search_retriever(state: AppState, search_tool=None) -> AppState:
    try:
        # Use 'or' to concisely handle missing keys or empty strings
        query = state.get("search_query") or f"{state['current_input']} Gloomhaven board game rules"
        results = search_tool.invoke(query)
        state["web_search_results"], state["urls"] = utils._normalize_search_results(results=results)
    except Exception as e:
        print(f"  Web search error: {e}")
        state["web_search_results"] = []
        state["urls"] = []
    return state

def final_response_generator(state: AppState, llm=None) -> AppState:
    prompt = prompts.GENERATE_PROMPT.format(
        chat_history=utils._format_recent_messages(state.get("messages", [])),
        question=state["current_input"],
        context=_get_context(state),
    )
    fallback = {
        "explanation": "I'm sorry, I encountered an error and couldn't process the rules properly.",
        "correct": True,
        "category": "Scenario"
    }
    parsed = utils._invoke_and_parse_json(llm, prompt, fallback)
    parsed["sources"] = _get_sources(state)
    if state["generation_attempts"] >= utils.MAX_ATTEMPTS:
        # only append if we're on the last attempt
        parsed["explanation"] += " (Note: The agent had multiple attempts to generate a good answer and may be unreliable.)"
    final_response = AgentResponse(**parsed)
    state["final_response"] = final_response
    return state

def relevance_prompt_generator(state: AppState, llm=None) -> AppState:
    prompt = prompts.RELEVANCE_PROMPT.format(
        question=state["current_input"],
        chunks=utils._format_chunks(state["retrieved_chunks"])
    )
    fallback = {"web_search_needed": True, "search_query": f"{state['current_input']} Gloomhaven board game rules"}
    parsed = utils._invoke_and_parse_json(llm, prompt, fallback)
    if not state["retrieved_chunks"]: # empty chunks should always trigger web search regardless of llm response
        state["web_search_needed"] = True
    else:
        state["web_search_needed"] = parsed.get("web_search_needed", False)
    return state


def answer_evaluator(state: AppState, llm=None) -> AppState:
    """Evaluates response quality. If not good triggers new rag retrieval with web search fallback on next step."""
    attempts = state.get("generation_attempts", 0) + 1
    state["generation_attempts"] = attempts
    if state.get("final_response"):
        resp = state["final_response"]
        answer_text = f"Explanation: {resp.explanation}\nCorrectness: {resp.correct}\nCategory: {resp.category}"
    else:
        answer_text = "No answer generated."
    prompt = prompts.EVALUATE_ANSWER_PROMPT.format(
        question=state["current_input"],
        answer=answer_text
    )
    fallback = {"is_good": True}
    parsed = utils._invoke_and_parse_json(llm, prompt, fallback)
    state["answer_is_good"] = parsed.get("is_good", True)
    if attempts == 2 and not state["answer_is_good"]: # from 2nd attempt trigger web search
        state["web_search_needed"] = True
    return state

# Helpers:
def _get_sources(state: AppState):
        sources = []
        if state.get("web_search_needed") and state.get("urls"):
            sources = state["urls"]
        elif state.get("retrieved_chunks"):
            sources = sorted(set(chunk.get("source") for chunk in state["retrieved_chunks"] if chunk.get("source")))
        return sources

def _get_context(state: AppState):
    context = []
    if state.get("web_search_needed") and state.get("web_search_results"):
        context = utils._format_web_results(state["web_search_results"])
    else:
        context = utils._format_chunks(state["retrieved_chunks"])
    return context