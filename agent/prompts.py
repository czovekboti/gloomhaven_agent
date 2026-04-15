GENERATE_PROMPT = """You are a Gloomhaven board game rules expert. Answer the user's question based on the provided context.

Recent conversation:
{chat_history}

Question: {question}
Context:
{context}
 
You MUST respond with a valid JSON object in this exact format:
{{
  "explanation": "<detailed explanation referencing the rules>",
  "correct": <true or false — whether the user played the situation correctly. Use true if not applicable>,
  "category": "<one of: BoardGameSetup, Combat, Scenario, Character>"
}}"""

REWRITE_PROMPT = """You are an expert at optimizing search queries for a vector database.
The user is asking a question about the Gloomhaven board game.
Rewrite the question to be a concise, keyword-rich search query that will effectively retrieve relevant rulebook sections.
Do not answer the question, just provide the optimized search query.

Recent conversation:
{chat_history}

User Question: {question}

Respond ONLY with a valid JSON object in this exact format:
{{
  "optimized_query": "<the optimized search query string>"
}}"""

RELEVANCE_PROMPT = """You are evaluating whether retrieved rulebook chunks are sufficient to answer a Gloomhaven question.

Question: {question}

Retrieved chunks:
{chunks}

Are these chunks sufficient to answer the question accurately?
IMPORTANT: If the user explicitly asks to search the web, check online, or something similar in their question, you MUST return true for web_search_needed.

Respond with valid JSON:
{{
  "web_search_needed": <true if chunks are insufficient or web search was explicitly requested, false if sufficient>
}}"""

EVALUATE_ANSWER_PROMPT = """You are evaluating an AI agent's answer to a Gloomhaven board game question.
Determine if the answer is confident, accurate, and completely resolves the user's question based on the rules.
If the answer states that it lacks information, cannot answer, or is unsure, it is NOT good.

Question: {question}
Agent Answer: {answer}

Respond ONLY with a valid JSON object in this exact format:
{{
  "is_good": <true or false>
}}"""