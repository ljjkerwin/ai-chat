import re
import json
import os
import math
import asyncio
from uuid import uuid4
from typing import Optional

import httpx
import numpy as np
import jieba
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


# ── Keyword search (Jieba + Okapi BM25) ───────────────────────────────────
# 关键词检索（使用 jieba 分词与 Okapi BM25 算法的兜底检索逻辑）

# 基础中英文停用词表，过滤常见高频无实际语义的词
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "人", "这", "中", "大", "来", "上", "国", "个", "到", "说", "们",
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "for", "with", "by", "about"
}


def _tokenize(text: str, filter_stopwords: bool = True) -> list[str]:  # 定义内部的分词函数，返回分词列表（保留词频）
    # 使用 jieba 进行精确分词并转为小写
    raw_tokens = jieba.lcut(text.lower())
    # 过滤掉空白字符与纯符号
    tokens = [t.strip() for t in raw_tokens if t.strip()]
    if filter_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS]
        # 如果过滤停用词后没有剩任何词，就用未过滤的分词以防止返回空结果
        if filtered:
            return filtered
    return tokens


class BM25:
    """轻量级自包含的 Okapi BM25 算法实现，用于关键词算分。"""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # 词频饱和度控制参数，通常介于 1.2 和 2.0 之间
        self.b = b  # 文档长度惩罚参数，通常为 0.75
        self.corpus_size = len(corpus)  # 语料库中文档（分块）总数
        # 计算语料库平均文档长度
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        
        # 计算每个词在语料库中的文档频率 DF (Document Frequency)
        self.doc_freqs = {}
        for doc in corpus:
            seen = set(doc)
            for word in seen:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
                
        # 预计算每个词的逆文档频率 IDF (Inverse Document Frequency)
        self.idf = {}
        for word, df in self.doc_freqs.items():
            # 使用标准的 BM25 IDF 公式，对数中加 1 保证 IDF 恒大于 0
            self.idf[word] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        """计算给定查询分词与文档分词之间的 BM25 得分。"""
        score = 0.0
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
            
        # 计算该文档中每个词的词频 TF (Term Frequency)
        doc_tf = {}
        for word in doc_tokens:
            doc_tf[word] = doc_tf.get(word, 0) + 1
            
        # 累加查询中每个词在当前文档的得分
        for word in query_tokens:
            if word not in doc_tf:
                continue
            tf = doc_tf[word]
            idf = self.idf.get(word, 0.0)
            
            # BM25 词频归一化公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
            
        return score


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
        
        # 初始化 BM25 计算文档得分
        corpus_tokens = [_tokenize(d["content"]) for d in docs]
        bm25 = BM25(corpus_tokens)
        q_tokens = _tokenize(query)

        scored_docs = []
        for i, d in enumerate(docs):
            if query_emb:
                doc_emb = await get_embedding(d["content"])
                score = cosine_similarity(query_emb, doc_emb) if doc_emb else bm25.get_score(q_tokens, corpus_tokens[i])
            else:
                score = bm25.get_score(q_tokens, corpus_tokens[i])
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

    # 初始化 BM25 计算分块得分
    corpus_tokens = [_tokenize(r["content"]) for r in rows]
    bm25 = BM25(corpus_tokens)
    q_tokens = _tokenize(query)

    scored = []
    for i, r in enumerate(rows):
        if query_emb and r["embedding"]:
            score = cosine_similarity(query_emb, json.loads(r["embedding"]))
        else:
            score = bm25.get_score(q_tokens, corpus_tokens[i])
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
