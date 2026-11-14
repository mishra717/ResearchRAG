"""
Tests for the RAG Research Assistant backend.

Run with:  pytest tests/ -v
"""

import sys
import os
import asyncio
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_clean_text(self):
        from core.ingestion import _clean_text
        raw = "Hello   World\n\n\n\nThis is a test    line.\n"
        cleaned = _clean_text(raw)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_chunk_text_basic(self):
        from core.ingestion import _chunk_text
        # 500-word text should produce at least 2 chunks at target=400, overlap=50
        text = " ".join(["word"] * 500)
        chunks = _chunk_text(text, paper_name="TestPaper", page=1,
                             target_words=400, overlap_words=50)
        assert len(chunks) >= 2
        for c in chunks:
            assert "text" in c
            assert c["metadata"]["paper_name"] == "TestPaper"
            assert c["metadata"]["page"] == 1

    def test_chunk_text_empty(self):
        from core.ingestion import _chunk_text
        chunks = _chunk_text("", "Empty", 1)
        assert chunks == []

    def test_ingest_directory_not_found(self):
        from core.ingestion import ingest_directory
        with pytest.raises(FileNotFoundError):
            ingest_directory("/nonexistent/path/xyz")

    def test_ingest_directory_no_files(self, tmp_path):
        from core.ingestion import ingest_directory
        with pytest.raises(ValueError):
            ingest_directory(str(tmp_path))

    def test_ingest_txt_file(self, tmp_path):
        from core.ingestion import ingest_directory
        (tmp_path / "paper.txt").write_text(
            "This is a sample research paper. " * 100
        )
        chunks = ingest_directory(str(tmp_path))
        assert len(chunks) > 0
        assert chunks[0]["metadata"]["paper_name"] == "paper"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddings:
    @patch("core.embeddings._load_model")
    def test_embed_texts_shape(self, mock_load):
        """Embedding should return (N, 768) float32 array."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        mock_load.return_value = mock_model

        from core.embeddings import embed_texts
        result = embed_texts(["text1", "text2", "text3"])
        assert result.shape == (3, 768)
        assert result.dtype == np.float32

    @patch("core.embeddings._load_model")
    def test_embed_query_shape(self, mock_load):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        mock_load.return_value = mock_model

        from core.embeddings import embed_query
        result = embed_query("what is LoRA?")
        assert result.shape == (1, 768)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_chunks(self, n=10):
        return [
            {
                "text": f"Sample research text chunk number {i}. " * 20,
                "metadata": {"paper_name": f"paper_{i % 3}", "page": i, "chunk_index": i},
            }
            for i in range(n)
        ]

    @patch("core.vector_store.embed_texts")
    def test_build_and_search(self, mock_embed, tmp_path):
        """Build index with mock embeddings and verify search returns results."""
        try:
            import faiss
        except ImportError:
            pytest.skip("faiss-cpu not installed")

        from core.vector_store import VectorStoreManager

        n = 10
        dim = 768
        vecs = np.random.rand(n, dim).astype(np.float32)
        # L2-normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        mock_embed.return_value = vecs

        store = VectorStoreManager(store_dir=str(tmp_path))
        chunks = self._make_chunks(n)
        store.build(chunks)

        assert store._index.ntotal == n

        query_vec = np.random.rand(1, dim).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "paper_name" in r
            assert 0.0 <= r["similarity_score"] <= 1.0

    def test_search_without_build_raises(self):
        from core.vector_store import VectorStoreManager
        store = VectorStoreManager()
        with pytest.raises(RuntimeError, match="not initialised"):
            store.search(np.zeros((1, 768), dtype=np.float32))

    def test_status_before_build(self):
        from core.vector_store import VectorStoreManager
        store = VectorStoreManager()
        status = store.status()
        assert status["indexed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def _mock_chunks(self):
        return [
            {
                "text": "LoRA reduces trainable parameters by injecting low-rank matrices.",
                "paper_name": "LoRA_Paper",
                "page": 3,
                "chunk_index": 0,
                "similarity_score": 0.92,
            },
            {
                "text": "Full fine-tuning updates all model weights, requiring more GPU memory.",
                "paper_name": "FineTuning_Survey",
                "page": 7,
                "chunk_index": 2,
                "similarity_score": 0.85,
            },
        ]

    @pytest.mark.asyncio
    @patch("core.rag_pipeline.embed_query")
    @patch("core.rag_pipeline.get_vector_store")
    @patch("core.rag_pipeline.generate_answer", new_callable=AsyncMock)
    async def test_run_pipeline(self, mock_llm, mock_store_fn, mock_embed):
        from core.rag_pipeline import run_rag_pipeline

        mock_embed.return_value = np.zeros((1, 768), dtype=np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = self._mock_chunks()
        mock_store_fn.return_value = mock_store

        mock_llm.return_value = "LoRA is more efficient than full fine-tuning [LoRA_Paper, p.3]."

        response = await run_rag_pipeline("Compare LoRA and full fine-tuning", top_k=5)

        assert response.question == "Compare LoRA and full fine-tuning"
        assert "LoRA" in response.answer
        assert response.chunks_retrieved == 2
        assert 0.0 <= response.confidence_score <= 1.0
        assert response.latency_ms > 0
        assert len(response.citations) == 2

    @pytest.mark.asyncio
    @patch("core.rag_pipeline.embed_query")
    @patch("core.rag_pipeline.get_vector_store")
    async def test_pipeline_no_chunks(self, mock_store_fn, mock_embed):
        from core.rag_pipeline import run_rag_pipeline

        mock_embed.return_value = np.zeros((1, 768), dtype=np.float32)
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_store_fn.return_value = mock_store

        response = await run_rag_pipeline("Obscure question", top_k=5)
        assert response.confidence_score == 0.0
        assert response.chunks_retrieved == 0
        assert "No relevant content" in response.answer

    def test_confidence_scoring(self):
        from core.rag_pipeline import _compute_confidence
        chunks = [
            {"similarity_score": 0.9},
            {"similarity_score": 0.8},
            {"similarity_score": 0.7},
        ]
        conf = _compute_confidence(chunks, top_k=5)
        assert 0.0 <= conf <= 1.0

        # Perfect coverage + high relevance → near 1
        chunks_perfect = [{"similarity_score": 1.0}] * 5
        conf_perfect = _compute_confidence(chunks_perfect, top_k=5)
        assert conf_perfect > 0.8


# ─────────────────────────────────────────────────────────────────────────────
# API Route Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_status_endpoint(self, client):
        r = client.get("/api/v1/status")
        assert r.status_code == 200
        data = r.json()
        assert "indexed" in data

    def test_ask_without_index_returns_409(self, client):
        # Clear any existing index
        with patch("api.routes.get_vector_store") as mock_vs:
            mock_vs.return_value.status.return_value = {"indexed": False}
            r = client.post("/api/v1/ask", json={"question": "What is LoRA?", "top_k": 5})
        assert r.status_code == 409

    def test_ask_short_question_validation(self, client):
        r = client.post("/api/v1/ask", json={"question": "Hi", "top_k": 5})
        assert r.status_code == 422   # validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
