# Gloomhaven AI Agent

An AI agent that answers questions about Gloomhaven board game rules. Ask a question about how to play, and it searches the rulebook to find the answer.

## What it does

- Takes a question about Gloomhaven rules
- Searches the rulebook for relevant information
- If needed, searches the web for extra help
- Gives you an answer with explanation

## How to use:

Run neccessary cells in jupyter notebook.  

## Example Questions

- "Can I attack an adjacent enemy with a ranged attack?"
- "How do I set up a scenario?"
- "What happens when I draw a null card?"

## How it Works

![Agent Flow](graph.png)

The agent tries up to 3 times to get a good answer:
1. First try: searches the rulebook
2. Second try: rewrites the query and searches again
3. Third try: falls back to web search

## Configuration

Edit `config.yaml` to change:
- LLM model and temperature
- Number of rulebook chunks to retrieve
- Maximum retry attempts

## Testing

Run unit tests:
```bash
python -m pytest tests/test_agent.py -v
```

Run evaluation:
```bash
python tests/evaluate_agent.py
```

## Project Structure

```
├── main.py              # Entry point
├── config.yaml          # Configuration
├── llm_factory.py       # LLM creation
├── agent/
│   ├── nodes.py        # Agent steps
│   ├── edges.py        # Routing logic
│   ├── graph.py       # LangGraph setup
│   └── state.py       # State definitions
├── services/
│   └── rag_service.py # Rulebook search
├── tests/
│    ├── test_agent.py   # Unit tests
│    └── evaluate_agent.py  # Evaluation
└── notebook.ipynb # run evaluation and chat here
```

## API Keys Needed

- **GROQ_API_KEY**: Get from [groq.com](https://groq.com)
- **TAVILY_API_KEY**: Get from [tavily.com](https://tavily.com)

The agent uses Groq for LLM calls and Tavily for web search.
