import re
import json
import os
import asyncio
from uuid import uuid4
from typing import Optional

import httpx
import numpy as np
from openai import AsyncOpenAI

from db import get_conn

# ── Text chunking ─────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            slice_text = text[start:end]
            for br in ["。\n", "！\n", "？\n", "\n\n", "。", "！", "？", ".", "\n"]:
                idx = slice_text.rfind(br)
                if idx > chunk_size * 0.4:
                    end = start + idx + len(br)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ── Cosine similarity ─────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


# ── Keyword search (BM25-style fallback) ──────────────────────────────────


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens: set[str] = set()
    # Latin/digit words (keep as units, e.g. "yahoteam", "2024")
    for m in re.finditer(r"[a-z0-9_]{2,}", text):
        tokens.add(m.group())
    # CJK: individual chars + bigrams for better recall
    cjk = re.findall(r"[一-鿿㐀-䶿]", text)
    tokens.update(cjk)
    tokens.update(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def keyword_score(query: str, chunk: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(chunk)
    return len(q_tokens & c_tokens) / len(q_tokens)


# ── Embedding ─────────────────────────────────────────────────────────────

_GEMINI_HOST = "generativelanguage.googleapis.com"


async def _embed_gemini(text: str, base_url: str, api_key: str, model: str) -> list[float]:
    """Call Gemini native embedContent API."""
    url = f"{base_url.rstrip('/')}/models/{model}:embedContent"
    body = {"content": {"parts": [{"text": text}]}}
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(url, json=body, headers={"x-goog-api-key": api_key})
        res.raise_for_status()
        return res.json()["embedding"]["values"]


async def _embed_openai(text: str, base_url: str, api_key: str, model: str) -> list[float]:
    """Call any OpenAI-compatible embeddings API."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    res = await client.embeddings.create(model=model, input=text)
    return res.data[0].embedding


async def get_embedding(text: str) -> Optional[list[float]]:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("MIMO_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    try:
        if _GEMINI_HOST in base_url:
            return await _embed_gemini(text, base_url, api_key, model)
        return await _embed_openai(text, base_url, api_key, model)
    except Exception as e:
        print(f"[RAG] Embedding failed, falling back to keyword search: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────


async def add_document(title: str, content: str) -> str:
    doc_id = str(uuid4())
    import time
    created_at = int(time.time() * 1000)

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, title, content, file_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, title, content, "text", created_at),
        )

    chunks = chunk_text(content)

    # Embed in batches of 5 to respect rate limits
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = await asyncio.gather(*[get_embedding(c) for c in batch])
        with get_conn() as conn:
            for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                conn.execute(
                    "INSERT INTO chunks (id, document_id, content, embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    (str(uuid4()), doc_id, chunk, json.dumps(emb) if emb else None, i + j),
                )

    return doc_id


async def search_knowledge(query: str, top_k: int = 4) -> dict:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT c.id, c.document_id, c.content, c.embedding,
                   d.title AS document_title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """).fetchall()

    print(f"[RAG] chunks in db: {len(rows)}")
    if not rows:
        # Fallback: search document content directly (e.g. docs with no chunks yet)
        with get_conn() as conn:
            docs = conn.execute(
                "SELECT id, title, content FROM documents"
            ).fetchall()
        if not docs:
            return {"context": "", "sources": []}
        query_emb = await get_embedding(query)
        scored_docs = []
        for d in docs:
            if query_emb:
                doc_emb = await get_embedding(d["content"])
                score = cosine_similarity(query_emb, doc_emb) if doc_emb else keyword_score(query, d["content"])
            else:
                score = keyword_score(query, d["content"])
            scored_docs.append((score, d))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [(s, d) for s, d in scored_docs[:top_k] if s > 0.05]
        if not top_docs:
            return {"context": "", "sources": []}
        sources = [
            {
                "documentId": d["id"],
                "documentTitle": d["title"],
                "chunkContent": d["content"][:220] + ("…" if len(d["content"]) > 220 else ""),
                "score": round(s, 2),
            }
            for s, d in top_docs
        ]
        context = "\n\n---\n\n".join(
            f"[来源{i+1}] {d['title']}\n{d['content']}"
            for i, (_, d) in enumerate(top_docs)
        )
        return {"context": context, "sources": sources}

    has_embeddings = any(r["embedding"] for r in rows)
    query_emb = await get_embedding(query) if has_embeddings else None

    scored = []
    for r in rows:
        if query_emb and r["embedding"]:
            score = cosine_similarity(query_emb, json.loads(r["embedding"]))
        else:
            score = keyword_score(query, r["content"])
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    mode = "embedding" if (query_emb and scored and scored[0][1]["embedding"]) else "keyword"
    print(f"[RAG] mode={mode} query={query!r} top scores: {[round(s,3) for s,_ in scored[:5]]}")
    top = [(s, r) for s, r in scored[:top_k] if s > 0.05]

    if not top:
        print(f"[RAG] no results above threshold for: {query!r}")
        return {"context": "", "sources": []}

    sources = [
        {
            "documentId": r["document_id"],
            "documentTitle": r["document_title"],
            "chunkContent": r["content"][:220] + ("…" if len(r["content"]) > 220 else ""),
            "score": round(s, 2),
        }
        for s, r in top
    ]

    context = "\n\n---\n\n".join(
        f"[来源{i+1}] {r['document_title']}\n{r['content']}"
        for i, (_, r) in enumerate(top)
    )

    return {"context": context, "sources": sources}
