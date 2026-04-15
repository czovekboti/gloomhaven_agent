from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline
from agent.utils import cfg

def create_llm():    
    try:
        llm = ChatGroq(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
        )
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"Groq unavailable ({e}), falling back to local model")
        return _create_local_llm(cfg)

def _create_local_llm(cfg):
    return HuggingFacePipeline.from_model_id(
        model_id=cfg["llm"]["local_model"],
        task="text-generation",
    )
