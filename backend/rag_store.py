import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ── Local Embedder ────────────────────────────────────────────────────────────
class LocalEmbedder:
    """
    Local sentence-transformers embedder.
    all-MiniLM-L6-v2: fast, lightweight, good for English clinical text.
    """
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        print(f"  Loading embedding model: {model} ...")
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


# ── Vector Store ──────────────────────────────────────────────────────────────
class ProtocolVectorStore:
    """
    ChromaDB-backed store. Persisted to disk.
    One collection for all protocols (filtered by 'source' metadata).
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "protocols",
        embedder: Optional[LocalEmbedder] = None,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder or LocalEmbedder()

    def build_index(self, chunks: list[dict], batch_log: bool = True) -> int:
        """
        chunks: list of {"text": str, "metadata": dict}
        Idempotent: upsert won't duplicate existing chunks.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids = [
            f"{m.get('source','doc')}_{m.get('chunk_index', i)}"
            for i, m in enumerate(metas)
        ]

        if batch_log:
            print(f"  Embedding {len(texts)} chunks locally ...")
        embeddings = self.embedder.embed_documents(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        if batch_log:
            print(f"  Indexed {len(texts)} chunks → '{self.collection.name}'")
        return len(texts)

    def query(
        self,
        question: str,
        top_k: int = 4,
        section: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve top_k chunks. Optional metadata filters: section, source.
        """
        query_emb = self.embedder.embed_query(question)
        if section and source:
            where_clause = {
                "$and": [
                    {"section": section},
                    {"source": source}
                ]
            }
        elif section:
            where_clause = {"section": section}
        elif source:
            where_clause = {"source": source}
        else:
            where_clause = None
        
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,
        )

        hits = []
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self.collection.count()

    def is_indexed(self, source: str) -> bool:
        """Check if a protocol is already indexed (avoids re-embedding)."""
        res = self.collection.get(where={"source": source}, limit=1)
        return len(res["ids"]) > 0