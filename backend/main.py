"""
RAG Research Assistant - Main Application Entry Point
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routes import router
from vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting RAG Research Assistant...")
    try:
        vs = VectorStoreManager()
        vs.load()
        logger.info("✅ Vector store loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️  No existing vector store found: {e}")
    yield
    logger.info("🛑 Shutting down RAG Research Assistant...")


app = FastAPI(
    title="RAG Research Assistant API",
    description="RAG-powered research assistant with citation-backed answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {"message": "RAG Research Assistant API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "RAG Research Assistant"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
