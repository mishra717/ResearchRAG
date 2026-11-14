"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="Compare LoRA and full fine-tuning for LLMs",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )

class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        default=None,
        description="Path to a folder containing PDFs. Defaults to ./data/papers/",
    )


# ─── Response Models ──────────────────────────────────────────────────────────

class Citation(BaseModel):
    paper_name: str
    page: int
    chunk_index: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    snippet: str = Field(..., description="First 200 chars of the source chunk")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    chunks_retrieved: int


class IngestResponse(BaseModel):
    status: str
    papers_processed: int
    total_chunks: int
    message: str


class StoreStatusResponse(BaseModel):
    indexed: bool
    total_chunks: int
    papers: List[str]
    embedding_model: str
    vector_db: str
