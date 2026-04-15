import json
from pathlib import Path
from dotenv import load_dotenv

from agent.utils import _invoke_and_parse_json
from agent.llm_factory import create_llm

load_dotenv()

DATA_FILE = Path(__file__).parent / "evaluation_data.json"

SYNTHETIC_DATA_PROMPT = """You are an expert at Gloomhaven rules.
Based on the provided examples, generate exactly 12 new, diverse question-answer pairs to evaluate an AI agent's understanding of Gloomhaven mechanics.

Examples:
{examples}

Instructions:
1. Create exactly 12 unique scenarios distinct from the examples above.
2. Make them moderately complex, testing edge cases or common rule misunderstandings.
3. Distribute them across the following categories: Combat, BoardGameSetup, Scenario, Character.

You MUST respond ONLY with a valid JSON object in this exact format:
{{
  "dataset": [
    {{
      "question": "<scenario question>",
      "expected_correct": <true or false>,
      "category": "<Combat, BoardGameSetup, Scenario, or Character>"
    }}
  ]
}}
"""

def load_evaluation_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"manual_examples": [], "generated_examples": []}

def save_evaluation_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def generate_synthetic_dataset(llm, manual_examples, max_retries=3):
    print("Generating synthetic dataset of 12 examples...")
    examples_text = "\n".join([
        f'{i+1}. Question: "{ex["question"]}"\n   Expected Correct: {ex["expected_correct"]}\n   Category: {ex["category"]}'
        for i, ex in enumerate(manual_examples)
    ])
    prompt = SYNTHETIC_DATA_PROMPT.format(examples=examples_text)
    parsed = _invoke_and_parse_json(llm, prompt, fallback={"dataset": []})
    return parsed.get("dataset", [])

def prepare_dataset():
    data = load_evaluation_data()
    manual = data.get("manual_examples", [])
    generated = data.get("generated_examples", [])
    
    if generated:
        print(f"Loading {len(generated)} generated examples from file")
        return manual + generated
        
    if not manual:
        print("No manual examples found. Please add to evaluation_data.json")
        return []
        
    llm = create_llm()
    new_generated = generate_synthetic_dataset(llm, manual)
    if new_generated:
        data["generated_examples"] = new_generated
        save_evaluation_data(data)
        print(f"Generated and saved {len(new_generated)} examples")
        
    return manual + new_generated

if __name__ == "__main__":
    prepare_dataset()