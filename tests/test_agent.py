import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent.nodes import (
    rag_retriever,
    relevance_prompt_generator,
    final_response_generator,
    answer_evaluator,
    web_search_retriever,
)
from agent.graph import build_graph
from agent.state import AgentResponse
def create_mock_llm(response_content):
    mock = MagicMock()
    mock.invoke.return_value = type('obj', (object,), {'content': response_content})()
    return mock


def create_mock_rag(chunks):
    mock = MagicMock()
    mock.retrieve_chunks.return_value = chunks
    return mock


def create_mock_search(results):
    mock = MagicMock()
    mock.invoke.return_value = results
    return mock


class TestGraph:
    def test_graph_builds(self):
        graph = build_graph(
            rag_service=create_mock_rag([]),
            search_tool=None
        )
        assert graph is not None


class TestRetryAndFallback:
    def test_bad_answer_triggers_retry(self):
        state = {
            "current_input": "test",
            "generation_attempts": 0,
            "web_search_needed": False,
            "final_response": AgentResponse(explanation="I don't know", correct=True, category="Combat", sources=[]),
        }
        mock_llm = create_mock_llm('{"is_good": false}')
        result = answer_evaluator(state, llm=mock_llm)
        assert result["generation_attempts"] == 1
        assert result["answer_is_good"] is False

    def test_malformed_json_uses_fallback(self):
        mock_llm = create_mock_llm("this is not json at all")
        state = {
            "current_input": "test",
            "retrieved_chunks": [{"text": "rules", "source": "doc.pdf"}],
            "web_search_needed": False,
            "generation_attempts": 0,
            "messages": [],
        }
        result = final_response_generator(state, llm=mock_llm)
        assert result["final_response"].explanation == "I'm sorry, I encountered an error and couldn't process the rules properly."
        assert result["final_response"].correct is True
        assert result["final_response"].category == "Scenario"


class TestWebSearch:
    def test_empty_chunks_trigger_web_search(self):
        mock_llm = create_mock_llm('{"web_search_needed": false, "search_query": ""}')
        state = {"current_input": "test", "retrieved_chunks": []}
        result = relevance_prompt_generator(state, llm=mock_llm)
        assert result["web_search_needed"] is True

    def test_chunks_present_no_web_search(self):
        mock_llm = create_mock_llm('{"web_search_needed": false, "search_query": ""}')
        state = {"current_input": "test", "retrieved_chunks": [{"text": "rules", "source": "doc.pdf"}]}
        result = relevance_prompt_generator(state, llm=mock_llm)
        assert result["web_search_needed"] is False

    def test_successful_web_search(self):
        mock_search = create_mock_search([{"content": "result 1", "url": "http://test.com"}])
        state = {"current_input": "test", "search_query": "test query"}
        result = web_search_retriever(state, search_tool=mock_search)
        assert len(result["web_search_results"]) == 1
        assert result["urls"] == ["http://test.com"]

    def test_retry_triggers_web_search(self):
        mock_llm = create_mock_llm('{"is_good": false}')
        state = {
            "current_input": "test",
            "generation_attempts": 1,
            "final_response": AgentResponse(explanation="bad answer", correct=True, category="Combat", sources=[]),
        }
        result = answer_evaluator(state, llm=mock_llm)
        assert result["generation_attempts"] == 2
        assert result["web_search_needed"] is True


class TestRAG:
    def test_rag_uses_optimized_query(self):
        mock_rag = create_mock_rag([{"text": "rules", "source": "doc.pdf"}])
        state = {
            "current_input": "what are hexes",
            "search_query": "Gloomhaven hex types movement",
            "retrieved_chunks": []
        }
        rag_retriever(state, rag_service=mock_rag)
        call_args = mock_rag.retrieve_chunks.call_args
        assert call_args[0][0] == "Gloomhaven hex types movement"