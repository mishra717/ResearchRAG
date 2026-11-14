"""
RAG Pipeline Orchestrator
"""

import logging
import time
from typing import Any, Dict, List

from embeddings import embed_query
from vector_store import get_vector_store
from llm import generate_answer
from schemas import Citation, QueryResponse

logger = logging.getLogger(__name__)


def _compute_confidence(chunks: List[Dict[str, Any]], top_k: int) -> float:
    if not chunks:
        return 0.0
    coverage  = len(chunks) / max(top_k, 1)
    relevance = sum(c["similarity_score"] for c in chunks) / len(chunks)
    return round(min(max(0.4 * coverage + 0.6 * relevance, 0.0), 1.0), 4)


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Citation]:
    return [
        Citation(
            paper_name=c["paper_name"],
            page=c["page"],
            chunk_index=c["chunk_index"],
            similarity_score=round(c["similarity_score"], 4),
            snippet=c["text"][:200].replace("\n", " ") + "…",
        )
        for c in chunks
    ]


async def run_rag_pipeline(question: str, top_k: int = 5) -> QueryResponse:
    t_start = time.perf_counter()

    query_vec = embed_query(question)
    store     = get_vector_store()
    chunks    = store.search(query_vec, top_k=top_k)

    if not chunks:
        return QueryResponse(
            question=question,
            answer="No relevant content found. Please ingest research papers first via POST /api/v1/ingest.",
            citations=[],
            confidence_score=0.0,
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
            chunks_retrieved=0,
        )

    confidence = _compute_confidence(chunks, top_k)
    answer     = await generate_answer(question, chunks)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return QueryResponse(
        question=question,
        answer=answer,
        citations=_build_citations(chunks),
        confidence_score=confidence,
        latency_ms=latency_ms,
        chunks_retrieved=len(chunks),
    )
