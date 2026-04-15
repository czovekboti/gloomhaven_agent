from langgraph.graph import add_messages
from pydantic import BaseModel
from typing import Annotated, Literal, Optional, TypedDict

class AgentResponse(BaseModel):
    explanation: str
    correct: bool
    category: Literal["BoardGameSetup", "Combat", "Scenario", "Character"]
    sources: list[str]

class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    current_input: str
    retrieved_chunks: list[dict]
    web_search_needed: bool
    search_query: str
    web_search_results: list[dict]
    urls: list[str]
    final_response: AgentResponse
    answer_is_good: bool
    generation_attempts: int