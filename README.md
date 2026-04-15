# Gloomhaven Rules AI Agent

An AI agent that answers questions about Gloomhaven board game rules using RAG over the official rulebook, with a web search fallback.

## How to Use

Open `notebook.ipynb` and run the cells — works both locally and in Google Colab.

You'll need to set up a `.env` file with your API keys first (see below) or you can set them while running the notebook.

## Example Questions

- "Can I attack an adjacent enemy with a ranged attack?"
- "How do I set up a scenario?"
- "What happens when I draw a null card?"

## How it Works

![Agent Flow](graph.png)

The agent uses a LangGraph pipeline with up to 3 attempts to find a good answer:

1. **First attempt** — retrieves relevant chunks from the rulebook via RAG
2. **Second attempt** — rewrites the query and searches the rulebook again
3. **Third attempt** — falls back to web search via Tavily

The agent may also fall back to web search after the first attempt if retrieval quality is low.

## Evaluation

- 3 manual question-answer pairs are used as seeds
- 12 synthetic examples are generated from these using the LLM
- The agent answers each question
- Results are checked for correctness and an LLM-as-judge reviews the explanations

## Configuration

Edit `config.yaml` to change the LLM model, temperature, number of retrieved chunks, and max retry attempts.

## Project Structure
## Evaluation:
1. Generate 15 question-answer pairs based on 3 examples
2. Input questions into agenti ai
3. Check manually and also use llm as judge 
4. Evaluate results

## Project Structure

```
├── config.yaml          # Configuration
├── agent/
│   ├── agent.py        # Main agent class
│   ├── edges.py        # Routing logic
│   ├── graph.py        # LangGraph setup
│   ├── llm_factory.py  # LLM creation
│   ├── nodes.py        # Agent steps
│   ├── prompts.py      # Prompt templates
│   ├── state.py       # State definitions
│   └── utils.py       # Utility functions
├── services/
│   └── rag_service.py  # Rulebook search
├── tests/
│   ├── dataset_generator.py  # Test data generation
│   ├── evaluate_agent.py     # Evaluation
│   └── test_agent.py         # Unit tests
└── notebook.ipynb            # Run evaluation and chat here
```

## API Keys Needed

- **GROQ_API_KEY**: Get from [groq.com](https://groq.com)
- **TAVILY_API_KEY**: Get from [tavily.com](https://tavily.com)

The agent uses Groq for LLM calls and Tavily for web search.