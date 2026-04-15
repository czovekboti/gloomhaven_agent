import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from agent.llm_factory import create_llm

from agent.utils import cfg
from services.rag_service import RagService
from agent.graph import build_graph

class GloomhavenAgent:
    def __init__(self):
        load_dotenv()
        self.cfg = cfg  # Saved to self so chat_loop can access it

        self.llm = create_llm()
        self.search_tool = TavilySearch(
            max_results=self.cfg["web_search"]["max_results"],
            api_key=os.getenv("TAVILY_API_KEY")
        )

        embedding_model = SentenceTransformer(self.cfg["embedding"]["model"])
        base_dir = Path(__file__).resolve().parent.parent
        db_path = base_dir / "chroma_db"
        docs_path = base_dir / "docs"
        chroma_client = chromadb.PersistentClient(path=str(db_path))

        self.rag_service = RagService(
            embedding_model=embedding_model,
            chroma_client=chroma_client,
            docs_dir=docs_path
        )

        self.app = build_graph(
            rag_service=self.rag_service,
            search_tool=self.search_tool,
            llm=self.llm
        )
    

    def chat_loop(self):
        print("Please enter your question (Type 'quit' to stop)")
        print("-" * 60)
        
        chat_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
                
            if not user_input:
                continue
                
            chat_history.append(HumanMessage(content=user_input))
            
            max_messages = self.cfg["agent"].get("chat_history_messages", 1) * 2 + 1
            if len(chat_history) > max_messages:
                chat_history = chat_history[-max_messages:]
                
            state = {
                "current_input": user_input,
                "messages": chat_history,
                "retrieved_chunks": [],
                "web_search_needed": False,
                "search_query": "",
                "web_search_results": [],
                "urls": [],
                "generation_attempts": 0,
            }
            
            try:
                final_state = self.app.invoke(state)
                
                response = final_state.get("final_response")
                if response:
                    chat_history.append(AIMessage(content=response.explanation))
                    print(f"\nAgent: {response.explanation}")
                    print(f"Correct: {response.correct} | Category: {response.category}")
                    if response.sources:
                        print("\nSources used:")
                        for source in response.sources:
                            print(f"- {source}")
                else:
                    print("\nAgent: I'm sorry, I couldn't generate a proper response.")
            except Exception as e:
                print(f"\nAn error occurred: {e}")