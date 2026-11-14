"""
API Routes
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ingestion import ingest_directory
from vector_store import get_vector_store
from rag_pipeline import run_rag_pipeline
from schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    StoreStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_PAPERS_DIR = "./data/raw_pdfs"


@router.post("/ingest", response_model=IngestResponse, summary="Ingest research papers")
async def ingest_papers(body: IngestRequest = IngestRequest()):
    directory = body.directory or DEFAULT_PAPERS_DIR
    logger.info(f"📥 Ingestion requested from: {directory}")
    try:
        chunks = ingest_directory(directory)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    try:
        store = get_vector_store()
        store.build(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    papers = list({c["metadata"]["paper_name"] for c in chunks})
    return IngestResponse(
        status="success",
        papers_processed=len(papers),
        total_chunks=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from {len(papers)} paper(s).",
    )


@router.post("/ask", response_model=QueryResponse, summary="Ask a research question")
async def ask_question(body: QueryRequest):
    store = get_vector_store()
    status = store.status()
    if not status["indexed"]:
        raise HTTPException(
            status_code=409,
            detail="No documents indexed yet. Call POST /api/v1/ingest first."
        )
    try:
        response = await run_rag_pipeline(question=body.question, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    return response


@router.get("/status", response_model=StoreStatusResponse, summary="Vector store status")
async def store_status():
    store = get_vector_store()
    return StoreStatusResponse(**store.status())


@router.delete("/store", summary="Clear the vector store")
async def clear_store():
    import shutil
    import vector_store as vs_module
    from vector_store import STORE_DIR
    try:
        if STORE_DIR.exists():
            shutil.rmtree(STORE_DIR)
        vs_module._store = None
        return JSONResponse({"status": "cleared", "message": "Vector store deleted."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear store: {e}")


@router.post("/upload", response_model=IngestResponse, summary="Upload and ingest PDFs")
async def upload_and_ingest(files: list[UploadFile] = File(...)):
    import shutil
    upload_dir = Path("./data/raw_pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if not upload.filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(status_code=415, detail=f"Unsupported: {upload.filename}")
        with open(upload_dir / upload.filename, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    try:
        chunks = ingest_directory(str(upload_dir))
        store = get_vector_store()
        store.build(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")

    papers = list({c["metadata"]["paper_name"] for c in chunks})
    return IngestResponse(
        status="success",
        papers_processed=len(papers),
        total_chunks=len(chunks),
        message=f"Indexed {len(chunks)} chunks from {len(papers)} file(s).",
    )
