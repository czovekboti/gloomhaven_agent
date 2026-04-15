import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from services.rag_service import RagService
from agent.graph import build_graph
from agent.utils import _invoke_and_parse_json
from agent.llm_factory import create_llm
from tests.dataset_generator import prepare_dataset
from agent.utils import cfg

# Load environment variables before initializing the LLM
load_dotenv()
llm = create_llm()

EVALUATOR_PROMPT = """You are an impartial judge evaluating the accuracy of a Gloomhaven AI Agent.
You will be provided with a Question, the Expected 'Correct' status (whether the play was actually correct), and the Agent's Prediction (whether it thought the play was correct), along with its Explanation.

Question: {question}
Expected Correct: {expected}
Expected Category: {category}
Agent Prediction: {prediction}
Agent Category: {agent_category}
Agent Explanation: {explanation}

Did the agent correctly predict whether the player's action was right or wrong based on the rules?
Answer true if the Agent Prediction matches the Expected Correct, otherwise false.
Also provide a brief reasoning for your evaluation.

You MUST respond with a valid JSON object in this exact format:
{{
  "match": <true or false>,
  "reasoning": "<brief reasoning>"
}}
"""

def evaluate_agent():
    search_tool = TavilySearch(
        max_results=cfg["web_search"]["max_results"],
        api_key=os.getenv("TAVILY_API_KEY")
    )

    embedding_model = SentenceTransformer(cfg["embedding"]["model"])
    db_path = Path(__file__).resolve().parent.parent / "chroma_db"
    docs_path = Path(__file__).resolve().parent.parent / "docs"
    chroma_client = chromadb.PersistentClient(
        path=str(db_path)
    )

    rag_service = RagService(
        embedding_model=embedding_model,
        chroma_client=chroma_client,
        docs_dir=docs_path
    )

    agent_graph = build_graph(
        rag_service=rag_service,
        search_tool=search_tool,
        llm=llm
    )

    dataset = prepare_dataset()
    if not dataset or len(dataset) < 1:
        print("Empty dataset. Aborting evaluation.")
        return
    print(f"Successfully prepared {len(dataset)} examples for evaluation.")
    print("Running evaluation...")

    report = {
        "total_evaluated": 0,
        "llm_judge_matches": 0,
        "exact_correct_matches": 0,
        "exact_category_matches": 0,
        "web_searches_triggered": 0,
        "llm_accuracy_percentage": 0.0,
        "exact_correct_accuracy_percentage": 0.0,
        "exact_category_accuracy_percentage": 0.0,
        "category_breakdown": {},
        "failed_evaluations": [],
        "successful_evaluations": [],
        "errors": []
    }

    for idx, item in enumerate(dataset):
        question = item.get("question", "")
        expected = item.get("expected_correct", False)
        expected_category = item.get("category", "Scenario")
        
        print(f"\nEvaluating {idx+1}/{len(dataset)}: {question[:50]}...")
        
        state = {
            "answer_is_good": False,
            "search_query": "",
            "current_input": question,
            "messages": [],
            "retrieved_chunks": [],
            "web_search_needed": False,
            "web_search_results": [],
            "urls": [],
            "generation_attempts": 0
        }

        try:
            final_state = agent_graph.invoke(state)
            agent_response = final_state.get("final_response")
            if not agent_response:
                raise ValueError("Agent returned no response.")
            agent_correct = agent_response.correct
            agent_explanation = agent_response.explanation
            agent_category = agent_response.category
        except Exception as e:
            print(f"Agent error evaluating question {idx+1}: {e}")
            report["errors"].append({
                "type": "agent_error",
                "question": question,
                "error": str(e)
            })
            continue
            
        used_web_search = final_state.get("web_search_needed", False)

        try:
            # Use LLM as judge
            eval_prompt = EVALUATOR_PROMPT.format(
                question=question,
                expected=expected,
                prediction=agent_correct,
                agent_category = agent_category,
                category=expected_category,
                explanation=agent_explanation
            )
            
            judge_parsed = _invoke_and_parse_json(llm, eval_prompt, {"match": False, "reasoning": ""})
            match = judge_parsed.get("match", False)
            reasoning = judge_parsed.get("reasoning", "")
        except Exception as e:
            print(f"Judge error evaluating question {idx+1}: {e}")
            report["errors"].append({
                "type": "judge_error",
                "question": question,
                "error": str(e)
            })
            continue
            
        exact_correct_match = (agent_correct == expected)
        exact_category_match = (agent_category == expected_category)

        # Update metrics
        report["total_evaluated"] += 1
        if match:
            report["llm_judge_matches"] += 1
        if exact_correct_match:
            report["exact_correct_matches"] += 1
        if exact_category_match:
            report["exact_category_matches"] += 1
            
        if used_web_search:
            report["web_searches_triggered"] += 1
            
        # Category tracking
        if expected_category not in report["category_breakdown"]:
            report["category_breakdown"][expected_category] = {
                "total": 0, "llm_correct": 0, "exact_correct": 0, "exact_category": 0
            }
        report["category_breakdown"][expected_category]["total"] += 1
        if match:
            report["category_breakdown"][expected_category]["llm_correct"] += 1
        if exact_correct_match:
            report["category_breakdown"][expected_category]["exact_correct"] += 1
        if exact_category_match:
            report["category_breakdown"][expected_category]["exact_category"] += 1
            
        if not match or not exact_correct_match or not exact_category_match:
            report["failed_evaluations"].append({
                "question": question,
                "expected_correct": expected,
                "expected_category": expected_category,
                "agent_prediction": agent_correct,
                "agent_category": agent_category,
                "agent_explanation": agent_explanation,
                "used_web_search": used_web_search,
                "llm_judge_match": match,
                "exact_correct_match": exact_correct_match,
                "exact_category_match": exact_category_match,
                "judge_reasoning": reasoning
            })
        else:
            report["successful_evaluations"].append({
                "question": question,
                "expected_correct": expected,
                "expected_category": expected_category,
                "agent_prediction": agent_correct,
                "agent_category": agent_category,
                "agent_explanation": agent_explanation,
                "used_web_search": used_web_search,
                "llm_judge_match": match,
                "exact_correct_match": exact_correct_match,
                "exact_category_match": exact_category_match,
                "judge_reasoning": reasoning
            })

    # Finalize report
    if report["total_evaluated"] > 0:
        total = report["total_evaluated"]
        report["llm_accuracy_percentage"] = (report["llm_judge_matches"] / total) * 100
        report["exact_correct_accuracy_percentage"] = (report["exact_correct_matches"] / total) * 100
        report["exact_category_accuracy_percentage"] = (report["exact_category_matches"] / total) * 100
        report["web_search_rate_percentage"] = (report["web_searches_triggered"] / total) * 100
        
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(json.dumps(report, indent=2))
    report_path = Path(__file__).parent / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    
    return report

if __name__ == "__main__":
    evaluate_agent()