import faiss
import joblib
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import MODELS_DIR, ANTHROPIC_API_KEY, RAG_TOP_K
from src.utils.logger import get_logger

log = get_logger(__name__)

def load_index(index_name: str = "rag"):
    index  = faiss.read_index(str(MODELS_DIR / f"{index_name}.index"))
    chunks = joblib.load(MODELS_DIR / f"{index_name}_chunks.pkl")
    return index, chunks

def retrieve_context(query: str, index, chunks, k: int = RAG_TOP_K) -> str:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = embedder.encode([query]).astype("float32")
    _, idx = index.search(q_vec, k)
    return "\n\n".join([chunks[i] for i in idx[0]])

def ask_claude(question: str, context: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content":
            f"You are a clinical research assistant.\n\n"
            f"Context from study protocol:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely and cite page context where possible."
        }]
    )
    return response.content[0].text
