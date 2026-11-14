"""
Embedding Module
────────────────
Responsibilities:
  1. Load Sentence Transformers model (all-mpnet-base-v2)
  2. Encode text chunks → 768-dimensional vectors
  3. Provide a singleton EmbeddingModel to avoid repeated loading
"""

import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

# ─── Model Configuration ──────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 32   # encode in batches to avoid OOM on large corpora


# ─── Singleton ────────────────────────────────────────────────────────────────

_model_instance = None


def _load_model():
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL_NAME}")
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("✅ Embedding model loaded.")
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _model_instance


# ─── Public API ───────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised 768-d vectors.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    np.ndarray of shape (N, 768), dtype float32
    """
    model = _load_model()
    logger.info(f"🔢 Embedding {len(texts)} text(s)…")

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=len(texts) > BATCH_SIZE,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string.

    Returns
    -------
    np.ndarray of shape (1, 768), dtype float32
    """
    return embed_texts([query])


def get_embedding_info() -> dict:
    """Return metadata about the active embedding model."""
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
    }
