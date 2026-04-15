import json
import re
import yaml
from pathlib import Path

# Load config once when module is imported
with open(Path(__file__).resolve().parent.parent / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MAX_ATTEMPTS = cfg["agent"]["max_generation_attempts"]

def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant chunks retrieved."
    return "\n\n".join(chunk["text"] for chunk in chunks)


def _format_web_results(results: list[dict]) -> str:
    if not results:
        return "No relevant web results found."
    return "\n\n".join(r.get("content", "") for r in results)


def _extract_urls(results: list[dict]) -> list[str]:
    return [r["url"] for r in results if r.get("url")]


def _format_recent_messages(messages: list) -> str:
    n = cfg["agent"].get("chat_history_messages", 1)
    history = messages[:-1] if messages else []
    if not history:
        return "No previous conversation."
    # return last n pairs of messages (user + ai)
    recent = history[-n*2:]
    formatted = []
    for msg in recent:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = getattr(msg, "content", str(msg))
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def _invoke_and_parse_json(llm, prompt: str, fallback: dict = None) -> dict:
    """Invokes llm and parses response as json."""
    if llm is None:
        raise ValueError("LLM instance is None.")
    current_prompt = prompt
    for attempt in range(cfg["agent"]["json_generation_attempts"]):
        response = llm.invoke(current_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        clean = re.sub(r"```json|```", "", content).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            current_prompt = prompt + f"\nYour previous response failed JSON parsing with error: {e}\n Respond ONLY with valid JSON."
    return fallback.copy() if fallback else {}

def _normalize_search_results(results: any) -> tuple[list[dict], list[str]]:
    """"Handle different types of search results"""
    if not results:
        return [], []
        
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(results, dict) and "results" in results:
        results =  results["results"]
    elif not isinstance(results, list):
        results = [{"content": str(results), "url": ""}]
    
    return results, _extract_urls(results)
