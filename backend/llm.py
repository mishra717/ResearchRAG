"""
LLM Module (Mistral via Ollama)
"""

import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "mistral"
OLLAMA_TIMEOUT  = 120.0

SYSTEM_PROMPT = """You are a precise academic research assistant.
Answer the user's question EXCLUSIVELY using the provided context excerpts.
Rules:
1. Base every claim on the provided context.
2. If context is insufficient, say: "The provided papers do not contain enough information."
3. Cite sources inline: [PaperName, p.N]
4. Do NOT use external knowledge.
5. Be concise and academically rigorous.
"""

def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    lines = ["=== CONTEXT FROM RESEARCH PAPERS ===\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Source {i}] Paper: {chunk['paper_name']} | Page: {chunk['page']}\n"
            f"{chunk['text']}\n{'─' * 60}"
        )
    return "\n".join(lines)

def _mock_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    sources = ", ".join(f"[{c['paper_name']}, p.{c['page']}]" for c in chunks[:3])
    return (
        f"[MOCK — Ollama not running]\n\n"
        f"Retrieved sources: {sources}\n\n"
        f"To get real answers, start Ollama:\n"
        f"    ollama serve\n    ollama pull mistral"
    )

async def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = _build_context_block(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{context}\n\n=== QUESTION ===\n{question}\n\n=== ANSWER ==="},
    ]
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                      "options": {"temperature": 0.1, "top_p": 0.9}},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("Ollama unreachable. Using mock response.")
        return _mock_answer(question, chunks)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Error generating answer: {e}"
