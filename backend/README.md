# 🔬 RAG Research Assistant — Backend

A **Retrieval-Augmented Generation (RAG)** backend that enables research-level Q&A with citation-backed, grounded answers from academic PDFs.

---

## 🏗️ Architecture

```
Frontend  →  FastAPI  →  RAG Pipeline
                              ↓
                    Query Embedding (Sentence Transformers)
                              ↓
                    FAISS Similarity Search
                              ↓
                    LLM Generation (Mistral via Ollama)
                              ↓
                    Structured Response + Citations
```

---

## 📁 Project Structure

```
rag_backend/
├── main.py                   # FastAPI app entry point
├── requirements.txt
├── api/
│   └── routes.py             # All API endpoints
├── core/
│   ├── ingestion.py          # PDF parsing, cleaning, chunking
│   ├── embeddings.py         # Sentence Transformers (all-mpnet-base-v2)
│   ├── vector_store.py       # FAISS index management
│   ├── llm.py                # Mistral via Ollama integration
│   └── rag_pipeline.py       # End-to-end orchestrator
├── models/
│   └── schemas.py            # Pydantic request/response models
├── tests/
│   └── test_backend.py       # Full test suite (pytest)
└── data/
    ├── papers/               # ← Place your PDF files here
    └── vector_store/         # Auto-generated FAISS index
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama + pull Mistral (for real LLM answers)

```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull mistral
```

> **Without Ollama**, the backend still works — it returns mock answers with full retrieval pipeline active.

### 3. Add your research papers

```bash
mkdir -p data/papers
cp /path/to/your/papers/*.pdf data/papers/
```

### 4. Run the server

**Option A — use the launch script (recommended, sets PYTHONPATH automatically):**

```bash
# Linux / macOS
chmod +x run.sh
./run.sh          # development (hot-reload)
./run.sh prod     # production

# Windows
run.bat
```

**Option B — set PYTHONPATH manually, then run uvicorn:**

```bash
# Linux / macOS
export PYTHONPATH=$(pwd)
uvicorn main:app --reload --port 8000

# Windows (PowerShell)
$env:PYTHONPATH = (Get-Location).Path
uvicorn main:app --reload --port 8000

# Windows (CMD)
set PYTHONPATH=%cd%
uvicorn main:app --reload --port 8000
```

**Option C — install as a package (editable):**

```bash
pip install -e .
uvicorn main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Index PDFs from a directory |
| `POST` | `/api/v1/upload` | Upload PDF files directly |
| `POST` | `/api/v1/ask` | Ask a research question |
| `GET`  | `/api/v1/status` | Vector store health |
| `DELETE` | `/api/v1/store` | Wipe the index |
| `GET`  | `/health` | Service health check |

---

## 📋 Usage Examples

### Ingest papers

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "./data/papers"}'
```

### Upload papers directly

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "files=@lora_paper.pdf" \
  -F "files=@attention_is_all_you_need.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare LoRA and full fine-tuning for LLMs",
    "top_k": 5
  }'
```

### Example Response

```json
{
  "question": "Compare LoRA and full fine-tuning for LLMs",
  "answer": "LoRA reduces trainable parameters by injecting low-rank matrices into attention layers [LoRA_Paper, p.3], while full fine-tuning updates all model weights [FineTuning_Survey, p.7]...",
  "citations": [
    {
      "paper_name": "LoRA_Paper",
      "page": 3,
      "chunk_index": 0,
      "similarity_score": 0.9241,
      "snippet": "LoRA reduces trainable parameters by injecting low-rank matrices..."
    }
  ],
  "confidence_score": 0.847,
  "latency_ms": 1842.5,
  "chunks_retrieved": 5
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Embedding model | `core/embeddings.py` | `all-mpnet-base-v2` |
| Chunk size | `core/ingestion.py` | 400 words |
| Chunk overlap | `core/ingestion.py` | 50 words |
| Top-K retrieval | `models/schemas.py` | 5 |
| LLM model | `core/llm.py` | `mistral` |
| Ollama URL | `core/llm.py` | `http://localhost:11434` |
| Papers dir | `api/routes.py` | `./data/papers` |
| Vector store | `core/vector_store.py` | `./data/vector_store` |

---

## 🔮 Future Improvements

- **Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25)
- **Re-ranking**: Cross-encoder model for precision
- **Query Decomposition**: Multi-hop retrieval
- **Advanced Confidence**: Similarity score + citation density
- **Multi-language**: Cross-lingual embeddings
- **Paper Comparison Mode**: Side-by-side analysis
