import json
import os
import math
import zlib
import asyncio
from uuid import uuid4
from typing import Optional

import httpx
import jieba
from openai import AsyncOpenAI

from db import get_conn
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

_milvus_client = None

import re

def validate_safe_id(id_str: str) -> bool:
    """Validate that the given ID contains only alphanumeric characters, dashes, or underscores."""
    return bool(re.match(r"^[a-zA-Z0-9_-]{1,64}$", id_str))


def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        uri = os.getenv("MILVUS_URI") or "milvus_rag.db"
        token = os.getenv("MILVUS_TOKEN", "")
        _milvus_client = MilvusClient(uri=uri, token=token)
    return _milvus_client


# ── BM25 / Sparse Vector Encoder ──────────────────────────────────────────

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "人", "这", "中", "大", "来", "上", "国", "个", "到", "说", "们",
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "for", "with", "by", "about"
}

def tokenize(text: str) -> list[str]:
    raw_tokens = jieba.lcut(text.lower())
    return [t.strip() for t in raw_tokens if t.strip() and t not in STOPWORDS]


def token_to_id(token: str) -> int:
    # Milvus requires uint32 keys, which fits in 0 to 4294967295
    return zlib.crc32(token.encode('utf-8')) & 0xFFFFFFFF


# 定义函数 get_corpus_stats，用于从数据库元数据中读取语料库统计指标
def get_corpus_stats() -> dict:
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT `key`, `value` FROM rag_metadata WHERE `key` IN ('total_chunks', 'total_tokens')")
                meta = {row["key"]: int(row["value"]) for row in cursor.fetchall()}
                N = meta.get("total_chunks", 0)
                total_tokens = meta.get("total_tokens", 0)
                avgdl = total_tokens / N if N > 0 else 0.0
                
                # 获取全量的 DF 数据作为向下兼容的备用
                cursor.execute("SELECT term, df FROM term_df")
                DF = {row["term"]: row["df"] for row in cursor.fetchall()}
                
                return {"N": N, "DF": DF, "avgdl": avgdl}
    except Exception as e:
        print(f"[RAG] Failed to get corpus stats from DB: {e}")
        return {"N": 0, "DF": {}, "avgdl": 0.0}


def get_sparse_vector_params(terms: list[str]) -> tuple[int, float, dict[str, int]]:
    """Fetch BM25 parameters (N, avgdl, DF) from DB only for the given terms."""
    if not terms:
        return 0, 0.0, {}
        
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                # 1. Fetch N and total_tokens
                cursor.execute("SELECT `key`, `value` FROM rag_metadata WHERE `key` IN ('total_chunks', 'total_tokens')")
                meta = {row["key"]: int(row["value"]) for row in cursor.fetchall()}
                N = meta.get("total_chunks", 0)
                total_tokens = meta.get("total_tokens", 0)
                avgdl = total_tokens / N if N > 0 else 0.0
                
                # 2. Fetch DF only for given terms
                placeholders = ",".join("%s" for _ in terms)
                cursor.execute(f"SELECT term, df FROM term_df WHERE term IN ({placeholders})", terms)
                df_map = {row["term"]: row["df"] for row in cursor.fetchall()}
                
                return N, avgdl, df_map
    except Exception as e:
        print(f"[RAG] Failed to fetch sparse vector parameters from DB: {e}")
        return 0, 0.0, {}


def update_incremental_stats_on_insert(chunks: list[str]) -> None:
    """Incrementally update term_df and rag_metadata when new chunks are inserted."""
    if not chunks:
        return
        
    term_counts = {}
    total_tokens = 0
    for chunk in chunks:
        tokens = tokenize(chunk)
        total_tokens += len(tokens)
        for token in set(tokens):
            term_counts[token] = term_counts.get(token, 0) + 1
            
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                # 1. Update term_df
                terms_data = [(term, df, df) for term, df in term_counts.items()]
                batch_size = 500
                for i in range(0, len(terms_data), batch_size):
                    batch = terms_data[i : i + batch_size]
                    cursor.executemany(
                        "INSERT INTO term_df (term, df) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE df = df + VALUES(df)",
                        batch
                    )
                
                # 2. Update rag_metadata
                cursor.execute("SELECT `key`, `value` FROM rag_metadata WHERE `key` IN ('total_chunks', 'total_tokens')")
                meta = {row["key"]: int(row["value"]) for row in cursor.fetchall()}
                
                new_chunks = meta.get("total_chunks", 0) + len(chunks)
                new_tokens = meta.get("total_tokens", 0) + total_tokens
                
                cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value` = %s", ("total_chunks", str(new_chunks), str(new_chunks)))
                cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value` = %s", ("total_tokens", str(new_tokens), str(new_tokens)))
    except Exception as e:
        print(f"[RAG] Failed to incrementally update stats on insert: {e}")


def update_incremental_stats_on_delete(doc_id: str) -> None:
    """Incrementally update term_df and rag_metadata when chunks are deleted."""
    try:
        # Fetch chunks content before deleting
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT content FROM chunks WHERE document_id = %s", (doc_id,))
                rows = cursor.fetchall()
        
        if not rows:
            return
            
        term_counts = {}
        total_tokens = 0
        for r in rows:
            tokens = tokenize(r["content"])
            total_tokens += len(tokens)
            for token in set(tokens):
                term_counts[token] = term_counts.get(token, 0) + 1
                
        with get_conn() as conn:
            with conn.cursor() as cursor:
                # 1. Update term_df: decrement df by the counts
                for term, count in term_counts.items():
                    cursor.execute("UPDATE term_df SET df = GREATEST(0, df - %s) WHERE term = %s", (count, term))
                # Delete terms where df <= 0
                cursor.execute("DELETE FROM term_df WHERE df <= 0")
                
                # 2. Update rag_metadata
                cursor.execute("SELECT `key`, `value` FROM rag_metadata WHERE `key` IN ('total_chunks', 'total_tokens')")
                meta = {row["key"]: int(row["value"]) for row in cursor.fetchall()}
                
                new_chunks = max(0, meta.get("total_chunks", 0) - len(rows))
                new_tokens = max(0, meta.get("total_tokens", 0) - total_tokens)
                
                cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value` = %s", ("total_chunks", str(new_chunks), str(new_chunks)))
                cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES (%s, %s) ON DUPLICATE KEY UPDATE `value` = %s", ("total_tokens", str(new_tokens), str(new_tokens)))
    except Exception as e:
        print(f"[RAG] Failed to incrementally update stats on delete: {e}")


_corpus_stats = None


def get_cached_corpus_stats() -> dict:
    global _corpus_stats
    if _corpus_stats is None:
        _corpus_stats = get_corpus_stats()
    return _corpus_stats


def reset_corpus_stats():
    global _corpus_stats
    _corpus_stats = None


def build_sparse_vector(text: str, corpus_stats: Optional[dict] = None) -> dict[int, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}

    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    N, avgdl, df_map = get_sparse_vector_params(list(tf.keys()))
    if N == 0:
        # Default fallback representation when corpus statistics are empty
        sparse_vec = {}
        for term, freq in tf.items():
            tid = token_to_id(term)
            sparse_vec[tid] = float(freq * 2.0)
        return sparse_vec

    dl = len(tokens)
    k1 = 1.5
    b = 0.75

    sparse_vec = {}
    for term, freq in tf.items():
        tid = token_to_id(term)
        df = df_map.get(term, 0)
        # Calculate IDF
        idf = max(0.0001, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
        # Calculate BM25 weight
        denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
        weight = idf * (freq * (k1 + 1.0)) / denom
        sparse_vec[tid] = float(round(weight, 4))

    return sparse_vec


# 定义函数 build_query_sparse_vector，根据查询文本与语料库统计构建查询的稀疏向量表示，用于稀疏检索
def build_query_sparse_vector(query: str, corpus_stats: Optional[dict] = None) -> dict[int, float]:
    # 对查询语句进行分词并滤除停用词，得到 token 列表
    tokens = tokenize(query)
    print(f"[RAG] Tokens: {tokens}")
    # 如果分词结果为空列表
    if not tokens:
        # 直接返回空的稀疏向量字典
        return {}

    # 初始化词频（Term Frequency，即词在当前查询中出现的次数）字典 tf
    tf = {}
    # 遍历查询分词后的所有 token
    for t in tokens:
        # 统计每个 token 在查询中出现的次数
        tf[t] = tf.get(t, 0) + 1

    N, avgdl, df_map = get_sparse_vector_params(list(tf.keys()))
    if N == 0:
        # 降级处理：直接将查询中各词的 token ID 映射到权重 1.0 并返回
        return {token_to_id(t): 1.0 for t in tokens}

    # 初始化存储最终稀疏向量结果的字典
    sparse_vec = {}
    # 遍历查询中各词的词频
    for term, q_tf in tf.items():
        # 通过哈希/CRC32 将词转换为对应的 uint32 类型 token ID 键
        tid = token_to_id(term)
        # 获取语料库中包含当前词的文档数，若没有则默认为 0
        df = df_map.get(term, 0)
        # 依据 BM25/TF-IDF 标准计算逆文档频率 IDF，限制最小值不低于 0.0001
        idf = max(0.0001, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
        # 计算该词在稀疏向量中的权重（查询词频 * IDF），四舍五入保留 4 位小数并转为 float
        sparse_vec[tid] = float(round(q_tf * idf, 4))
    # 返回构建好的查询稀疏向量（格式为 {token_id: weight}）
    print(f"[RAG] Sparse vector: {sparse_vec}")
    return sparse_vec


# ── Milvus Collection Management ──────────────────────────────────────────


def ensure_collection(client: MilvusClient, dimension: int) -> None:
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")

    # If the collection exists with the old schema (does not contain sparse_vector, tenant_id, or kb_id), drop it first
    if client.has_collection(collection_name=collection_name):
        desc = client.describe_collection(collection_name=collection_name)
        fields = [f.get("name") for f in desc.get("fields", [])]
        if "sparse_vector" not in fields or "tenant_id" not in fields or "kb_id" not in fields:
            print(f"[RAG] Old collection schema detected. Dropping {collection_name} to apply new dense+sparse+tenant partition schema...")
            client.drop_collection(collection_name)

    if not client.has_collection(collection_name=collection_name):
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            partition_key_field="tenant_id",
            num_partitions=128
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64, is_partition_key=True)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            metric_type="COSINE",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    # Always ensure collection is loaded
    client.load_collection(collection_name)


def sync_sqlite_to_milvus() -> None:
    """Sync existing MySQL chunk embeddings to Milvus collection if empty."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT c.id, c.document_id, c.content, c.embedding, c.chunk_index,
                           d.kb_id, d.tenant_id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding IS NOT NULL
                """)
                rows = cursor.fetchall()
        if not rows:
            return

        client = get_milvus_client()
        collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")

        if client.has_collection(collection_name=collection_name):
            desc = client.describe_collection(collection_name=collection_name)
            fields = [f.get("name") for f in desc.get("fields", [])]
            if "sparse_vector" not in fields or "tenant_id" not in fields or "kb_id" not in fields:
                print(f"[RAG] Old collection schema detected. Dropping {collection_name} to apply new dense+sparse+tenant partition schema...")
                client.drop_collection(collection_name)

        if not client.has_collection(collection_name=collection_name):
            first_emb = json.loads(rows[0]["embedding"])
            ensure_collection(client, len(first_emb))

        stats = client.get_collection_stats(collection_name=collection_name)
        milvus_count = int(stats.get("row_count", 0))

        if milvus_count == 0:
            print(f"[RAG] Syncing {len(rows)} existing chunks from MySQL to Milvus...")
            
            # Fetch global corpus stats
            corpus_stats = get_corpus_stats()
            
            milvus_data = []
            for r in rows:
                emb = json.loads(r["embedding"])
                milvus_data.append({
                    "id": r["id"],
                    "dense_vector": emb,
                    "sparse_vector": build_sparse_vector(r["content"], corpus_stats),
                    "document_id": r["document_id"],
                    "kb_id": r["kb_id"],
                    "tenant_id": r["tenant_id"]
                })

            batch_size = 100
            for i in range(0, len(milvus_data), batch_size):
                client.insert(collection_name=collection_name, data=milvus_data[i : i + batch_size])
            print(f"[RAG] Synced {len(milvus_data)} chunks to Milvus successfully.")
    except Exception as e:
        print(f"[RAG] MySQL to Milvus sync failed or skipped: {e}")


def delete_document(doc_id: str, tenant_id: str) -> None:
    """Delete a document from both MySQL and Milvus."""
    # First, update incremental stats
    update_incremental_stats_on_delete(doc_id)

    from db import delete_document_by_id
    delete_document_by_id(doc_id, tenant_id)

    try:
        client = get_milvus_client()
        collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")
        if client.has_collection(collection_name=collection_name):
            client.delete(
                collection_name=collection_name,
                filter=f"tenant_id == '{tenant_id}' and document_id == '{doc_id}'"
            )
            print(f"[RAG] Deleted chunks from Milvus for document {doc_id} under tenant {tenant_id}")
    except Exception as e:
        print(f"[RAG] Failed to delete chunks from Milvus for document {doc_id} under tenant {tenant_id}: {e}")


def delete_knowledge_base_vectors(kb_id: str, tenant_id: str) -> None:
    """Delete all chunks vectors belonging to a specific knowledge base from Milvus."""
    if not validate_safe_id(kb_id) or not validate_safe_id(tenant_id):
        raise ValueError("Invalid kb_id or tenant_id format")
    try:
        client = get_milvus_client()
        collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")
        if client.has_collection(collection_name=collection_name):
            client.delete(
                collection_name=collection_name,
                filter=f"tenant_id == '{tenant_id}' and kb_id == '{kb_id}'"
            )
            print(f"[RAG] Deleted all chunks from Milvus for kb {kb_id} under tenant {tenant_id}")
        
        # Reset stats on KB deletion
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO rag_metadata (`key`, `value`) VALUES ('initialized', 'false') ON DUPLICATE KEY UPDATE `value` = 'false'")
    except Exception as e:
        print(f"[RAG] Failed to delete chunks from Milvus or reset stats for kb {kb_id}: {e}")
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
        print(f"[RAG] Embedding failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────


async def add_document(title: str, content: str, kb_id: str, tenant_id: str) -> str:
    if not validate_safe_id(kb_id) or not validate_safe_id(tenant_id):
        raise ValueError("Invalid kb_id or tenant_id format")

    doc_id = str(uuid4())
    import time
    created_at = int(time.time() * 1000)

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (id, kb_id, tenant_id, title, content, file_type, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (doc_id, kb_id, tenant_id, title, content, "text", created_at),
            )

    chunks = chunk_text(content)

    # Embed in batches of 5 to respect rate limits
    batch_size = 5
    milvus_data = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = await asyncio.gather(*[get_embedding(c) for c in batch])
        with get_conn() as conn:
            with conn.cursor() as cursor:
                for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                    chunk_id = str(uuid4())
                    cursor.execute(
                        "INSERT INTO chunks (id, document_id, content, embedding, chunk_index) VALUES (%s, %s, %s, %s, %s)",
                        (chunk_id, doc_id, chunk, json.dumps(emb) if emb else None, i + j),
                    )
                    if emb:
                        milvus_data.append({
                            "id": chunk_id,
                            "dense_vector": emb,
                            "sparse_vector": build_sparse_vector(chunk),
                            "document_id": doc_id,
                            "kb_id": kb_id,
                            "tenant_id": tenant_id
                        })

    if milvus_data:
        try:
            client = get_milvus_client()
            collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")
            ensure_collection(client, len(milvus_data[0]["dense_vector"]))
            client.insert(collection_name=collection_name, data=milvus_data)
            print(f"[RAG] Inserted {len(milvus_data)} chunks into Milvus successfully.")
        except Exception as e:
            print(f"[RAG] Milvus insertion failed: {e}")

    update_incremental_stats_on_insert(chunks)

    return doc_id


async def search_knowledge(
    query: str, 
    kb_id: str, 
    tenant_id: str, 
    top_k: int = 4, 
    chat_history: Optional[list[dict]] = None
) -> dict:
    if not validate_safe_id(kb_id) or not validate_safe_id(tenant_id):
        print(f"[RAG] Invalid search identifiers: kb_id={kb_id}, tenant_id={tenant_id}")
        return {"context": "", "sources": []}

    print(f"[RAG] Searching knowledge for query: {query}, kb_id: {kb_id}, tenant_id: {tenant_id}, chat_history: {len(chat_history)}")

    # 1. 对话查询重写 (Query Rewriting)
    search_query = query
    if chat_history and len(chat_history) > 0:
        try:
            mimo_base_url = os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1")
            mimo_api_key = os.getenv("MIMO_API_KEY", "")
            mimo_model = os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL")
            
            if mimo_api_key:
                # 仅保留最近5轮对话以平衡性能和成本
                history_slice = chat_history[-5:]
                formatted_history = ""
                for msg in history_slice:
                    role_str = "用户" if msg.get("role") == "user" else "助手"
                    content_str = msg.get("content", "")
                    formatted_history += f"{role_str}: {content_str}\n"
                
                rewrite_prompt = f"""你是一个智能检索助手。请结合以下历史对话内容和用户最新的问题，生成一个最适合用来检索知识库的核心问题（Standalone Search Query）。
这个核心问题应当：
1. 包含历史上下文中必要的实体、名词以及指代词指向的具体对象（补全省略信息）；
2. 保持简洁，去除多余 of 问候语或解释性文字；
3. 直接输出这个独立的核心问题，不要包含任何多余的前言、解释或标记。

【对话历史】
{formatted_history}

【最新问题】
{query}

【核心问题】"""

                client = AsyncOpenAI(base_url=mimo_base_url, api_key=mimo_api_key)
                response = await client.chat.completions.create(
                    model=mimo_model,
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    temperature=0.3,
                    max_tokens=512
                )

                rewritten = response.choices[0].message.content
                if rewritten:
                    rewritten = rewritten.strip()
                    # Clean up <think>...</think> from content for reasoning models
                    if "<think>" in rewritten:
                        if "</think>" in rewritten:
                            rewritten = rewritten.split("</think>")[-1].strip()
                        else:
                            rewritten = rewritten.split("<think>")[0].strip()


                if rewritten:
                    rewritten = rewritten.strip('"\'`').strip()
                    if rewritten and len(rewritten) < 100:
                        print(f"[RAG] Rewrite query: '{query}' -> '{rewritten}'")
                        search_query = rewritten
        except Exception as e:
            print(f"[RAG] Query rewriting failed: {e}")

    query_emb = await get_embedding(search_query)

    if query_emb:  # 如果成功生成了查询文本的密集向量表示
        try:  # 开启 try-except 块以捕获向量搜索过程中的异常
            client = get_milvus_client()  # 获取 Milvus 数据库客户端实例
            collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")  # 从环境变量获取 Milvus 集合名称，默认为 "rag_chunks"
            if client.has_collection(collection_name=collection_name):  # 检查 Milvus 中是否存在该集合
                query_sparse = build_query_sparse_vector(search_query)  # 构建查询文本的稀疏向量表示，用于稀疏检索
                
                filter_expr = f"tenant_id == '{tenant_id}' and kb_id == '{kb_id}'"  # 构造租户与知识库逻辑隔离的分区路由过滤表达式

                # 2. 检查 Reranker 配置
                reranker_api_key = os.getenv("RERANKER_API_KEY")
                reranker_base_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1")
                reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
                
                try:
                    threshold_str = os.getenv("RERANKER_SCORE_THRESHOLD")
                    reranker_threshold = float(threshold_str) if threshold_str else None
                except ValueError:
                    reranker_threshold = None
                    
                use_reranker = bool(reranker_api_key)
                # 如果使用 Rerank，则 Milvus 检索增加召回数量 (limit)
                retrieval_limit = max(15, top_k * 3) if use_reranker else top_k

                req_dense = AnnSearchRequest(  # 实例化密集向量检索请求对象
                    data=[query_emb],  # 传入查询文本的密集向量数据列表
                    anns_field="dense_vector",  # 指定检索的向量字段为 "dense_vector"
                    param={"metric_type": "COSINE", "params": {}},  # 设置检索度量类型为余弦相似度（COSINE）
                    limit=retrieval_limit,  # 限制该子检索返回的最大相似结果数
                    expr=filter_expr  # 注入租户与知识库过滤表达式进行预过滤以限制检索范围
                )
                req_sparse = AnnSearchRequest(  # 实例化稀疏向量检索请求对象
                    data=[query_sparse] if query_sparse else [{}],  # 若稀疏向量非空则使用空字典列表
                    anns_field="sparse_vector",  # 指定检索的向量字段为 "sparse_vector"
                    param={"metric_type": "IP", "params": {}},  # 设置检索度量类型为内积（IP - Inner Product）
                    limit=retrieval_limit,  # 限制该子检索返回的最大相似结果数
                    expr=filter_expr  # 注入相同的过滤表达式进行预过滤以限制检索范围
                )
                
                search_res = client.hybrid_search(  # 执行密集加稀疏向量的双通道混合检索
                    collection_name=collection_name,  # 指定目标检索集合名称
                    reqs=[req_dense, req_sparse],  # 传入并行子检索请求列表
                    ranker=RRFRanker(),  # 指定互反排名融合算法（RRFRanker）对结果进行重排与排序融合
                    limit=retrieval_limit,  # 限制重排融合后最终返回的最大相似结果数
                    output_fields=["document_id"]  # 声明需要 Milvus 额外返回文档 ID（document_id）字段
                )

                if search_res and search_res[0]:  # 如果混合检索结果不为空且第一个查询有返回结果
                    hits = search_res[0]  # 提取检索返回的匹配结果列表（hits）
                    top_hits = [h for h in hits if h.get("distance", 0.0) > 0.0]  # 过滤相似度得分（distance）大于 0.0 的有效检索结果

                    if top_hits:  # 如果存在过滤后的有效相似匹配项
                        doc_ids = list({h.get("id") for h in top_hits})  # 提取并去重所有匹配项的主键（即 MySQL 中的 chunk ID）并转为列表
                        
                        chunk_details = {}  # 初始化字典，以 chunk ID 作为键存放关系库中的文本块详细信息
                        if doc_ids:  # 如果提取到的匹配 chunk ID 列表不为空
                            placeholders = ",".join("%s" for _ in doc_ids)  # 根据 ID 数量生成 SQL 参数占位符，如 "%s,%s"
                            query_sql = f"""
                                SELECT c.id, c.content, c.chunk_index, d.id AS doc_id, d.title AS doc_title 
                                FROM chunks c 
                                JOIN documents d ON d.id = c.document_id 
                                WHERE c.id IN ({placeholders}) AND d.tenant_id = %s
                            """  # 构造 SQL 查询，关联 chunks 表与 documents 表，强制加入 tenant_id 条件确保多租户安全
                            with get_conn() as conn:
                                with conn.cursor() as cursor:
                                    cursor.execute(query_sql, doc_ids + [tenant_id])
                                    # 获取所有查询到的结果行
                                    rows = cursor.fetchall()
                                    # 遍历每一行查询到的数据
                                    for r in rows:
                                        # 将 chunk 详细信息存入字典，以 chunk 的 ID 为键
                                        chunk_details[r["id"]] = r

                        # 3. 运行 Rerank 重排管道
                        if use_reranker and chunk_details:
                            candidate_chunks = []
                            for hit in top_hits:
                                cid = hit.get("id")
                                if cid in chunk_details:
                                    candidate_chunks.append({
                                        "id": cid,
                                        "content": chunk_details[cid]["content"],
                                        "score": hit.get("distance", 0.0)
                                    })
                            
                            if candidate_chunks:
                                try:
                                    rerank_url = f"{reranker_base_url.rstrip('/')}/rerank"
                                    headers = {
                                        "Authorization": f"Bearer {reranker_api_key}",
                                        "Content-Type": "application/json"
                                    }
                                    body = {
                                        "model": reranker_model,
                                        "query": search_query,
                                        "documents": [c["content"] for c in candidate_chunks],
                                        "top_n": top_k
                                    }
                                    async with httpx.AsyncClient(timeout=15) as http_client:
                                        res = await http_client.post(rerank_url, json=body, headers=headers)
                                        res.raise_for_status()
                                        reranked_data = res.json()
                                    
                                    results = reranked_data.get("results", [])
                                    reranked_chunks = []
                                    for r in results:
                                        idx = r.get("index")
                                        score = r.get("relevance_score", 0.0)
                                        # 过滤低于阈值的文档
                                        if reranker_threshold is not None and score < reranker_threshold:
                                            continue
                                        if 0 <= idx < len(candidate_chunks):
                                            c = candidate_chunks[idx]
                                            c["score"] = score
                                            reranked_chunks.append(c)
                                    
                                    print(f"[RAG] Reranker model '{reranker_model}' returned {len(reranked_chunks)} documents above threshold {reranker_threshold}")
                                    
                                    top_hits_processed = []
                                    for rc in reranked_chunks:
                                        top_hits_processed.append({
                                            "id": rc["id"],
                                            "distance": rc["score"]
                                        })
                                    top_hits = top_hits_processed[:top_k]
                                except Exception as re_err:
                                    print(f"[RAG] Reranking request failed: {re_err}")
                                    top_hits = top_hits[:top_k]
                        else:
                            top_hits = top_hits[:top_k]

                        # 初始化返回的数据源列表，用于返回给前端或调用方
                        sources = []
                        # 初始化上下文部分的文本块列表，用于合成最终 key context 字符串
                        context_parts = []
                        # 初始化用于标记来源编号的计数器，从 1 开始
                        valid_i = 1
                        # 遍历检索出的前几个最相关的匹配项
                        for hit in top_hits:
                            # 获取当前匹配项的 chunk ID
                            chunk_id = hit.get("id")
                            # 从数据库查询出来的详细信息字典中获取对应 chunk 的数据
                            r = chunk_details.get(chunk_id)
                            # 如果该 chunk 不在数据库的详细数据中，则跳过
                            if not r:
                                continue
                                
                            # 获取对应的文档 ID
                            doc_id = r["doc_id"]
                            # 获取对应的文本块具体内容
                            content = r["content"]
                            # 获取对应的文档标题，如果为空则默认为 "未知文档"
                            title = r["doc_title"] or "未知文档"
                            # 获取该匹配项的相似度分数（distance）
                            score = hit.get("distance", 0.0)

                            # 将该匹配数据源拼装后添加到 sources 列表中
                            sources.append({
                                # 文档 ID
                                "documentId": doc_id,
                                # 文档标题
                                "documentTitle": title,
                                # 截取前 220 个字符的文本块内容，若超出则加省略号 "…"
                                "chunkContent": content[:220] + ("…" if len(content) > 220 else ""),
                                # 对匹配得分保留 4 位小数
                                "score": round(score, 4),
                            })
                            # 将格式化后的文档标题与内容拼装并存入 context_parts
                            context_parts.append(f"[来源{valid_i}] {title}\n{content}")
                            # 增加来源计数器
                            valid_i += 1

                        # 将所有来源的文本片段用 "\n\n---\n\n" 拼接成一个大字符串，作为最终的 context
                        context = "\n\n---\n\n".join(context_parts)
                        # 在控制台打印当前 Milvus 检索并返回的结果数量
                        print(f"[RAG] Milvus search returned {len(sources)} results. Query used: '{search_query}'")
                        # 返回包含拼接好的上下文 context 和数据源 sources 的字典
                        return {"context": context, "sources": sources}
        # 捕捉在 Milvus 查询或 MySQL 查询中发生的任何异常
        except Exception as e:
            # 在控制台打印异常信息
            print(f"[RAG] Milvus search failed: {e}")

    # 如果无法生成查询的向量，或者搜索过程中发生异常/无结果，返回默认的空上下文和空数据源字典
    return {"context": "", "sources": []}
