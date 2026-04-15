from langgraph.graph import END, StateGraph
from agent import edges
from agent.state import AppState
from agent import nodes
from functools import partial
from agent.utils import cfg

def build_graph(rag_service=None, search_tool=None, llm=None) -> StateGraph:
    graph = StateGraph(AppState)
    
    # Add nodes
    graph.add_node("query_rewriter", partial(nodes.query_rewriter, llm=llm))
    graph.add_node("rag_retriever", partial(nodes.rag_retriever, rag_service=rag_service))
    graph.add_node("relevance_prompt_generator", partial(nodes.relevance_prompt_generator, llm=llm))
    graph.add_node("web_search_retriever", partial(nodes.web_search_retriever, search_tool=search_tool))
    graph.add_node("final_response_generator", partial(nodes.final_response_generator, llm=llm))
    graph.add_node("answer_evaluator", partial(nodes.answer_evaluator, llm=llm))
    
    # Define edges
    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "rag_retriever")
    graph.add_edge("rag_retriever", "relevance_prompt_generator")
    graph.add_edge("web_search_retriever", "final_response_generator")
    
    # Define conditional edges
    graph.add_conditional_edges("relevance_prompt_generator", edges.route_relevance, {
        "web_search_retriever": "web_search_retriever",
        "final_response_generator": "final_response_generator"
    })
    graph.add_conditional_edges("answer_evaluator", edges.route_evaluation, {
        "web_search_retriever": "web_search_retriever",
        END: END,
        "query_rewriter": "query_rewriter"
    })
    graph.add_conditional_edges("final_response_generator", edges.route_final_response, {
        END: END,
        "answer_evaluator": "answer_evaluator"
    })
    
    return graph.compile(debug=cfg["agent"]["debug"])