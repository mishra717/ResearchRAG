"""
Vector Store Module (FAISS)
"""

import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STORE_DIR  = Path("./data/processed")
INDEX_FILE = STORE_DIR / "faiss.index"
META_FILE  = STORE_DIR / "metadata.pkl"
DEFAULT_TOP_K = 5


class VectorStoreManager:
    def __init__(self, store_dir: str = str(STORE_DIR)):
        self._store_dir = Path(store_dir)
        self._index     = None
        self._metadata: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")

        from embeddings import embed_texts, EMBEDDING_DIM

        texts   = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(vectors)

        self._index    = index
        self._metadata = [c["metadata"] | {"text": c["text"]} for c in chunks]
        self.save()

    def save(self) -> None:
        import faiss
        self._store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info(f"💾 Vector store saved → {self._store_dir}")

    def load(self) -> None:
        import faiss
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("No store found. Run /ingest first.")
        self._index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "rb") as f:
            self._metadata = pickle.load(f)
        logger.info(f"📂 Loaded {self._index.ntotal} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        if self._index is None:
            raise RuntimeError("Store not initialised. Call build() or load() first.")
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append({
                "text":             meta["text"],
                "paper_name":       meta["paper_name"],
                "page":             meta["page"],
                "chunk_index":      meta["chunk_index"],
                "similarity_score": float(np.clip(score, 0.0, 1.0)),
            })
        return results

    def status(self) -> Dict[str, Any]:
        if self._index is None:
            return {"indexed": False, "total_chunks": 0, "papers": [],
                    "embedding_model": "all-mpnet-base-v2", "vector_db": "FAISS"}
        papers = list({m["paper_name"] for m in self._metadata})
        return {
            "indexed":       True,
            "total_chunks":  self._index.ntotal,
            "papers":        sorted(papers),
            "embedding_model": "all-mpnet-base-v2",
            "vector_db":     "FAISS IndexFlatIP",
        }


_store: Optional[VectorStoreManager] = None

def get_vector_store() -> VectorStoreManager:
    global _store
    if _store is None:
        _store = VectorStoreManager()
        try:
            _store.load()
        except FileNotFoundError:
            pass
    return _store
