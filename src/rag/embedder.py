import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"

def build_faiss_index(chunks: list, index_name: str = "rag"):
    embedder = SentenceTransformer(MODEL_NAME)
    vectors = embedder.encode(chunks, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(MODELS_DIR / f"{index_name}.index"))
    joblib.dump(chunks, MODELS_DIR / f"{index_name}_chunks.pkl")
    log.info(f"FAISS index built with {index.ntotal} vectors")
    return index, chunks
