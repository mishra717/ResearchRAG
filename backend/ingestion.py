"""
Ingestion Module
────────────────
Responsibilities:
  1. Parse PDFs using PyMuPDF (fitz)
  2. Clean extracted text (remove noise, headers/footers, artifacts)
  3. Chunk text into semantically coherent units (~400 words)
  4. Attach metadata: paper name, page number, chunk index
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CHUNK_WORD_TARGET = 400      # target words per chunk
CHUNK_OVERLAP_WORDS = 50     # overlap between consecutive chunks


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Remove PDF noise: repeated whitespace, ligatures, stray chars, etc."""
    # Normalise unicode ligatures
    replacements = {
        "\ufb01": "fi", "\ufb02": "fl", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    }
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)

    # Collapse runs of whitespace / blank lines
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # Drop lines that look like page numbers or headers (≤ 4 tokens)
    lines = raw.splitlines()
    filtered = [
        ln for ln in lines
        if len(ln.split()) > 4 or ln.strip() == ""
    ]
    return "\n".join(filtered).strip()


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    paper_name: str,
    page: int,
    target_words: int = CHUNK_WORD_TARGET,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """
    Split `text` into overlapping word-window chunks.
    Returns a list of chunk dicts with metadata.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "paper_name": paper_name,
                "page": page,
                "chunk_index": chunk_idx,
            },
        })

        chunk_idx += 1
        # Advance by (target - overlap) so consecutive chunks share context
        start += target_words - overlap_words

    return chunks


# ─── PDF Parsing ──────────────────────────────────────────────────────────────

def _parse_pdf_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract per-page text from a PDF using PyMuPDF."""
    pages: List[Dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean_text(raw)
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
    doc.close()
    return pages


def _parse_pdf_fallback(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Fallback: treat the file as plain text (useful for .txt test files
    when PyMuPDF is not installed).
    """
    try:
        content = pdf_path.read_text(encoding="utf-8", errors="replace")
        return [{"page": 1, "text": _clean_text(content)}]
    except Exception as e:
        logger.error(f"Fallback parsing failed for {pdf_path}: {e}")
        return []


def parse_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return a list of page dicts: [{page: int, text: str}, ...]"""
    if PYMUPDF_AVAILABLE:
        return _parse_pdf_pymupdf(pdf_path)
    logger.warning("PyMuPDF not available – using plain-text fallback.")
    return _parse_pdf_fallback(pdf_path)


# ─── Public API ───────────────────────────────────────────────────────────────

def ingest_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Walk `directory`, parse every PDF/TXT file, clean and chunk text.

    Returns
    -------
    List of chunk dicts:
        {
            "text": str,
            "metadata": {
                "paper_name": str,
                "page": int,
                "chunk_index": int,
            }
        }
    """
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir.resolve()}")

    # Accept PDFs and plain-text files
    files = list(data_dir.glob("**/*.pdf")) + list(data_dir.glob("**/*.txt"))
    if not files:
        raise ValueError(f"No PDF or TXT files found in: {data_dir.resolve()}")

    all_chunks: List[Dict[str, Any]] = []

    for pdf_path in files:
        paper_name = pdf_path.stem
        logger.info(f"📄 Processing: {paper_name}")

        pages = parse_pdf(pdf_path)
        paper_chunks = []

        for page_data in pages:
            chunks = _chunk_text(
                text=page_data["text"],
                paper_name=paper_name,
                page=page_data["page"],
            )
            paper_chunks.extend(chunks)

        logger.info(f"   → {len(pages)} pages → {len(paper_chunks)} chunks")
        all_chunks.extend(paper_chunks)

    logger.info(f"✅ Ingestion complete: {len(all_chunks)} total chunks from {len(files)} files.")
    return all_chunks
