from pathlib import Path
import hashlib
import json
import re
import fitz
from agent.utils import cfg
import nltk


class RagService:
    def __init__(self, 
                 chroma_client,
                 embedding_model, 
                 docs_dir: Path
    ):
        self.model = embedding_model
        self.docs_dir = Path(docs_dir)
        self.chroma_client = chroma_client
        self.hash_file = self.docs_dir.parent / "chroma_db" / "file_hashes.json"
        self.cfg = cfg
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        # Automatically check for new files and create the collection on startup
        self.index_all_documents()

    # Use hashes to track file changes and avoid re-indexing unchanged files
    def load_hashes(self) -> dict:
        if self.hash_file.exists():
            with open(self.hash_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}   

    def save_hashes(self, hashes: dict):
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)

    def file_hash(self, path: Path) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            text = re.sub(r'(?:\b[a-zA-Z0-9]\b[\s\r\n]+){4,}', ' ', text)
            text = text.replace('Ǳ', ' ')
            
            return text.strip()
        except Exception as e:
            raise ValueError(f"PDF read error ({pdf_path.name}): {e}")
        
    def chunk_text(self, text: str) -> list[str]:
        chunk_size = self.cfg["rag"]["chunk_size"] 
        overlap = self.cfg["rag"]["chunk_overlap"] 
        # Semantic sentence-based chunking using NLTK
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Implement overlap by keeping the last few sentences
                overlap_chunk = []
                overlap_words = 0
                for s in reversed(current_chunk):
                    if overlap_words + len(s.split()) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_words += len(s.split())
                    else:
                        break
                current_chunk = overlap_chunk
                current_word_count = overlap_words
                
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        return self.model.encode(chunks, show_progress_bar=False).tolist()

    def index_all_documents(self):
        if not self.docs_dir.exists():
            print(f"Docs directory not found: {self.docs_dir.resolve()}")
            return
        hashes = self.load_hashes()
        total_new = 0
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            return
        collection_name = self.cfg["rag"]["collection_name"]
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        new_files = 0
        for pdf_path in pdf_files:
            current_hash = self.file_hash(pdf_path)
            hash_key = str(pdf_path)
            if hashes.get(hash_key) == current_hash:
                continue
            try:
                text = self.extract_text_from_pdf(pdf_path)
            except ValueError as e:
                print(f"{e}")
                continue

            if not text:
                continue
            chunks = self.chunk_text(text)
            embeddings = self.embed_chunks(chunks)
            ids = [f"{collection_name}__{pdf_path.stem}__{i}" for i in range(len(chunks))]

            try:
                existing_ids = collection.get(where={"source": pdf_path.name})["ids"]
                if existing_ids:
                    collection.delete(ids=existing_ids)
            except Exception as e:
                print(f"Warning: Could not clean up old chunks for {pdf_path.name}: {e}")

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{
                    "source": pdf_path.name,
                    "chunk_index": i
                } for i in range(len(chunks))]
            )

            hashes[hash_key] = current_hash
            new_files += 1
            total_new += 1

        self.save_hashes(hashes)

    def retrieve_chunks(self, query: str, n_results: int = 5) -> list[dict]:
        try:
            collection = self.chroma_client.get_collection(name=self.cfg["rag"]["collection_name"])
        except Exception:
            print(f"  [RAG] Collection not found. Skipping retrieval.")
            return []
            
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()[0]
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )        
        except Exception as e:
            print(f"  [RAG] Query error: {e}")
            return []
            
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            
            for doc, meta in zip(docs, metas):
                chunk_data = {"text": doc}
                chunk_data.update(meta)
                formatted_results.append(chunk_data)
                
        return formatted_results
