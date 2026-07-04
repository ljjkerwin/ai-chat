import json
import os
import asyncio
from uuid import uuid4
from typing import Optional

import httpx
from openai import AsyncOpenAI
import json;

from db import get_conn
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker, WeightedRanker

_milvus_client = None
_http_client = None
_openai_client = None
_checked_collections = set()

def get_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    elif str(_openai_client.base_url).rstrip("/") != base_url.rstrip("/") or _openai_client.api_key != api_key:
        _openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _openai_client

def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=15)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

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


# ── BGE-M3 / Sparse Vector Encoder ────────────────────────────────────────

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

_bge_m3_ef = None

def get_bge_m3_client() -> BGEM3EmbeddingFunction:
    global _bge_m3_ef
    if _bge_m3_ef is None:
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        
        print(f"[RAG] Initializing local BGE-M3 model on device: {device.upper()}...")
        local_model_path = os.getenv("BGE_M3_MODEL_PATH", "BAAI/bge-m3")
        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if local_model_path != "BAAI/bge-m3" and not os.path.exists(local_model_path):
            print(f"[RAG] Configured BGE_M3_MODEL_PATH '{local_model_path}' not found, falling back to 'BAAI/bge-m3'...")
            local_model_path = "BAAI/bge-m3"
            
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=local_model_path,
            device=device,
            use_fp16=(device != "cpu")
        )
        print("[RAG] Local BGE-M3 model initialized successfully.")
    return _bge_m3_ef


def sparse_to_dict(sparse_array) -> dict[int, float]:
    """Convert scipy sparse array/matrix to python dict {token_id: weight} for Milvus."""
    coo = sparse_array.tocoo()
    return {int(col): float(val) for col, val in zip(coo.col, coo.data)}


# ── Milvus Collection Management ──────────────────────────────────────────


def ensure_collection(client: MilvusClient, dimension: int = 1024) -> None:
    collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")

    # If the collection exists with the old schema or incorrect dimension, drop it first
    if client.has_collection(collection_name=collection_name):
        desc = client.describe_collection(collection_name=collection_name)
        fields = desc.get("fields", [])
        field_names = [f.get("name") for f in fields]
        dense_field = next((f for f in fields if f.get("name") == "dense_vector"), None)
        dense_dim = dense_field.get("params", {}).get("dim") or dense_field.get("dim") if dense_field else None
        
        if "sparse_vector" not in field_names or "tenant_id" not in field_names or "kb_id" not in field_names or dense_dim != 1024:
            print(f"[RAG] Old collection schema or dimension mismatch (current: {dense_dim}) detected. Dropping {collection_name} to apply new 1024-dim dense+sparse+tenant partition schema...")
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
        index_params.add_index(
            field_name="kb_id",
            index_type="INVERTED"
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    # Check and dynamically create index on kb_id for existing collection if missing
    try:
        indexes = client.list_indexes(collection_name=collection_name)
        has_kbid_idx = False
        for idx_name in indexes:
            try:
                idx_desc = client.describe_index(collection_name=collection_name, index_name=idx_name)
                if idx_desc.get("field_name") == "kb_id":
                    has_kbid_idx = True
                    break
            except Exception:
                pass
        
        if not has_kbid_idx:
            print(f"[RAG] Index on kb_id not found for existing collection. Creating INVERTED index on kb_id...")
            try:
                client.release_collection(collection_name)
            except Exception:
                pass
            client.create_index(
                collection_name=collection_name,
                field_name="kb_id",
                index_type="INVERTED"
            )
    except Exception as e:
        print(f"[RAG] Check/Create index on kb_id failed or already exists: {e}")

    # Always ensure collection is loaded
    client.load_collection(collection_name)



def delete_document(doc_id: str, tenant_id: str) -> None:
    """Delete a document from both MySQL and Milvus."""
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


def merge_overlapping_strings(s1: str, s2: str) -> str:
    """Merge s1 and s2, identifying and deduplicating any overlapping suffix of s1 and prefix of s2."""
    s1_clean = s1.strip()
    s2_clean = s2.strip()
    if not s1_clean:
        return s2_clean
    if not s2_clean:
        return s1_clean
        
    # Detect up to 300 characters of overlap
    max_check_len = min(len(s1_clean), len(s2_clean), 300)
    best_overlap_len = 0
    for i in range(1, max_check_len + 1):
        if s1_clean[-i:] == s2_clean[:i]:
            best_overlap_len = i
            
    if best_overlap_len > 0:
        return s1_clean + s2_clean[best_overlap_len:]
    return s1_clean + "\n" + s2_clean


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
    if not chunks:
        return doc_id

    # Generate embeddings and sparse vectors locally using BGE-M3
    bge_m3 = get_bge_m3_client()
    res = await asyncio.to_thread(bge_m3, chunks)

    milvus_data = []
    with get_conn() as conn:
        with conn.cursor() as cursor:
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid4())
                dense_vec = res["dense"][idx].tolist()
                sparse_vec = sparse_to_dict(res["sparse"][idx])
                
                # Write to MySQL chunks table
                cursor.execute(
                    "INSERT INTO chunks (id, document_id, content, embedding, chunk_index) VALUES (%s, %s, %s, %s, %s)",
                    (chunk_id, doc_id, chunk, json.dumps(dense_vec), idx),
                )
                
                milvus_data.append({
                    "id": chunk_id,
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    "document_id": doc_id,
                    "kb_id": kb_id,
                    "tenant_id": tenant_id
                })

    if milvus_data:
        try:
            client = get_milvus_client()
            collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")
            ensure_collection(client, 1024)
            client.insert(collection_name=collection_name, data=milvus_data)
            print(f"[RAG] Inserted {len(milvus_data)} chunks into Milvus successfully.")
        except Exception as e:
            print(f"[RAG] Milvus insertion failed: {e}")

    return doc_id


async def search_knowledge(
    query: str, 
    kb_id: str, 
    tenant_id: str, 
    top_k: int = 4, 
    chat_history: Optional[list[dict]] = None
) -> dict:
    """
    RAG 知识检索核心方法。
    
    检索流程包含以下阶段：
    1. 查询重写 (Query Rewriting)：结合历史对话上下文将多轮问答问题重写为独立的检索 query。
    2. 向量化 (Embedding Generation)：使用 BGE-M3 模型生成密向量与稀疏向量表示。
    3. 混合检索与排序融合 (Milvus Hybrid Search & Fusion)：利用 Milvus 支持的 Dense 与 Sparse 向量进行混合检索，
       并根据配置的环境变量选择 WeightedRanker (加权融合) 或 RRFRanker (互反排名融合) 进行排序融合。
    4. 一次性元数据拉取 (Consolidated Metadata Fetch - 优化1)：一次性从 MySQL 获取候选 chunks 及其对应 documents 的元数据，
       避免多重回表查询，降低接口 RT。
    5. Rerank 重排管道 (Rerank Pipeline)：根据权重或分数进行 Cross-Encoder 二次精细打分。
    6. 上下文扩展与去重拼接 (Context Merging & Deduplication - 优化3)：为 top_k chunks 批量获取邻近块，
       对序号连续的块运用 merge_overlapping_strings 算法对 overlap 的 120 字符重叠段进行对齐去重，避免 prompt 噪音污染并节省 token 预算。
    7. 格式化构建 (XML Output Builder)：以 XML 节点形式组织文档段落，并应用 8000 字符的硬限预算以防 token 溢出。
    """
    # 验证输入的安全性和格式，防止恶意的 ID 输入
    if not validate_safe_id(kb_id) or not validate_safe_id(tenant_id):
        print(f"[RAG] Invalid search identifiers: kb_id={kb_id}, tenant_id={tenant_id}")
        return {"context": "", "sources": []}

    print(f"[RAG] Searching knowledge for query: {query}, kb_id: {kb_id}, tenant_id: {tenant_id}, chat_history: {len(chat_history) if chat_history else 0}")

    # 1. 对话查询重写 (Query Rewriting) with Timeout
    # 若包含历史对话，使用大语言模型补全当前 query 中可能缺失的代词或指代信息，生成独立的 Standalone 核心问题
    search_query = query
    if chat_history and len(chat_history) > 0:
        try:
            mimo_base_url = os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1")
            mimo_api_key = os.getenv("MIMO_API_KEY", "")
            mimo_model = os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL")
            
            if mimo_api_key:
                # 仅保留最近 5 轮对话以平衡性能和成本
                history_slice = chat_history[-5:]
                formatted_history = ""
                for msg in history_slice:
                    role_str = "用户" if msg.get("role") == "user" else "助手"
                    content_str = msg.get("content", "")
                    formatted_history += f"{role_str}: {content_str}\n"
                
                # 引导 LLM 生成最契合知识检索的问题
                rewrite_prompt = f"""你是一个智能检索助手。请结合以下对话历史和用户最新的问题，生成一个最适合用来检索知识库的核心问题（Standalone Search Query）。
这个核心问题应当：
1. 包含历史上下文中必要的实体、名词以及指代词指向的具体对象（补全省略信息）；
2. 保持简洁，去除多余的问候语或解释性文字；
3. 直接输出这个独立的核心问题，不要包含任何多余的前言、解释或标记。
4. 如果是某些领域的专有名词，则原路返回这个名词
5. 如果是一些打招呼、感谢语、感叹语之类的不需要检索知识库的语句，直接返回：None

【示例一】
对话历史：
用户：贵州茅台最新的财报怎么样？
助手：贵州茅台2024年年报显示营收同比增长15%，净利润创历史新高。
用户最新的问题：它的分红政策呢？
核心问题：贵州茅台的分红政策

【示例二】
对话历史：
用户：什么是市盈率？
助手：市盈率（PE）是股价除以每股收益，反映投资者为每单位收益愿意支付的价格。
用户最新的问题：你好
核心问题：None

【示例三】
用户最新的问题：No.23214
核心问题：No.23214

【对话历史】
{formatted_history}

【用户最新的问题】
{query}

【核心问题】"""

                client = get_openai_client(mimo_base_url, mimo_api_key)
                # 使用 wait_for 设定 2.5 秒超时防止影响聊天响应流畅度
                print("requesting rewrite query...")
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=mimo_model,
                        messages=[{"role": "user", "content": rewrite_prompt}],
                        temperature=0.3,
                        max_tokens=1024  # MiMo 等推理模型需要为 <think>思考链</think> 预留足够空间，实际答案本身很短
                    ),
                    timeout=30
                )

                rewritten = None
                if response and response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message and message.content:
                        rewritten = message.content.strip()

                print(f"[RAG STEP 1]{rewritten}")

                # 对思考型推理大模型的输出进行后处理，提取并过滤掉思考标签
                if rewritten:
                    if "<think>" in rewritten:
                        if "</think>" in rewritten:
                            rewritten = rewritten.split("</think>")[-1].strip()
                        else:
                            rewritten = rewritten.split("<think>")[0].strip()

                    rewritten = rewritten.strip('"\'\'`').strip()
                    if rewritten and len(rewritten) < 100:
                        print(f"Rewrite query: '{query}' -> '{rewritten}'")
                        search_query = rewritten
                    else:
                        # 重写后为空或过长，视为无明确检索意图，跳过知识库搜索
                        print(f"[RAG] Rewritten query empty or too long, skipping knowledge search.")
                        return {"context": "", "sources": []}
                else:
                    # rewritten 为 None，LLM 显式判断无明确问题（如闲聊），跳过知识库搜索
                    print(f"[RAG] Query rewriting returned None (no retrieval intent), skipping knowledge search.")
                    return {"context": "", "sources": []}
        except Exception as e:
            print(f"[RAG] Query rewriting bypassed or timed out: {e}")

    # 2. 向量化查询 (Embedding Generation)
    # 利用 BGE-M3 生成密集和稀疏特征表示
    try:
        bge_m3 = get_bge_m3_client()
        res = await asyncio.to_thread(bge_m3, [search_query])
        query_emb = res["dense"][0].tolist()
        query_sparse = sparse_to_dict(res["sparse"][0])
        print(f"[RAG STEP 2] dense and sparse convert: {search_query}")
        # print(query_emb)
        # print(query_sparse)
    except Exception as e:
        print(f"[RAG] BGE-M3 query vector generation failed: {e}")
        query_emb = None
        query_sparse = None

    if query_emb is not None:  # 如果成功生成了查询文本的密集向量表示
        try:  # 开启 try-except 块以捕获向量搜索过程中的异常
            client = get_milvus_client()  # 获取 Milvus 数据库客户端实例
            collection_name = os.getenv("MILVUS_COLLECTION", "rag_chunks")  # 从环境变量获取 Milvus 集合名称，默认为 "rag_chunks"
            
            # 使用全局缓存避免重复检查 collection 是否加载
            has_coll = False
            if collection_name in _checked_collections:
                has_coll = True
            else:
                has_coll = await asyncio.to_thread(client.has_collection, collection_name=collection_name)
                if has_coll:
                    _checked_collections.add(collection_name)

            if has_coll:  # 检查 Milvus 中是否存在该集合
                
                # 构造租户与知识库逻辑隔离的分区路由过滤表达式
                filter_expr = f"tenant_id == '{tenant_id}' and kb_id == '{kb_id}'"

                # 检查 Reranker 模型接口配置
                reranker_api_key = os.getenv("RERANKER_API_KEY")
                reranker_base_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1")
                reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
                
                try:
                    threshold_str = os.getenv("RERANKER_SCORE_THRESHOLD")
                    reranker_threshold = float(threshold_str) if threshold_str else None
                except ValueError:
                    reranker_threshold = None
                    
                use_reranker = bool(reranker_api_key)
                # 如果使用 Rerank，则 Milvus 检索增加召回数量 (limit) 以确保更优召回率
                retrieval_limit = max(15, top_k * 3) if use_reranker else top_k

                # 构造密集向量和稀疏向量的检索子请求
                req_dense = AnnSearchRequest(
                    data=[query_emb],
                    anns_field="dense_vector",
                    param={"metric_type": "COSINE", "params": {}},
                    limit=retrieval_limit,
                    expr=filter_expr
                )
                req_sparse = AnnSearchRequest(
                    data=[query_sparse] if query_sparse else [{}],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {}},
                    limit=retrieval_limit,
                    expr=filter_expr
                )
                
                # 3. 混合检索融合权重配置 (Hybrid Search Ranker Config - 优化4)
                # 支持通过环境变量配置 WeightedRanker 赋予 Dense 和 Sparse 不同的检索融合倾向
                dense_weight_str = os.getenv("HYBRID_DENSE_WEIGHT")
                sparse_weight_str = os.getenv("HYBRID_SPARSE_WEIGHT")
                
                ranker = RRFRanker()
                if dense_weight_str is not None and sparse_weight_str is not None:
                    try:
                        dense_weight = float(dense_weight_str)
                        sparse_weight = float(sparse_weight_str)
                        ranker = WeightedRanker(dense_weight, sparse_weight)
                        print(f"[RAG] Using WeightedRanker with weights: dense={dense_weight}, sparse={sparse_weight}")
                    except ValueError:
                        print(f"[RAG] Invalid weights for WeightedRanker: dense={dense_weight_str}, sparse={sparse_weight_str}. Falling back to RRFRanker.")

                # 在线程池中非阻塞执行 Milvus 混合向量检索
                search_res = await asyncio.to_thread(
                    client.hybrid_search,
                    collection_name=collection_name,  # 指定目标检索集合名称
                    reqs=[req_dense, req_sparse],  # 传入并行子检索请求列表
                    ranker=ranker,  # 指定融合算法对结果进行重排与排序融合
                    limit=retrieval_limit,  # 限制重排融合后最终返回的最大相似结果数
                    output_fields=["document_id"]  # 声明需要 Milvus 额外返回文档 ID（document_id）字段
                )

                print(f"[RAG STEP 3] search: {len(search_res[0])}")
                """
                [
                    {
                        "id": "ca1a11b4-a262-4239-853e-fe0a458eb81a",
                        "distance": 0.015384615398943424,
                        "entity": {
                            "document_id": "0877ade9-08fb-43cb-b356-609f6b577835"
                        }
                    }
                ]
                """

                if search_res and search_res[0]:  # 如果混合检索结果不为空且第一个查询有返回结果
                    hits = search_res[0]  # 提取检索返回的匹配结果列表（hits）
                    top_hits = [h for h in hits if h.get("distance", 0.0) > 0.0]  # 过滤相似度得分（distance）大于 0.0 的有效检索结果

                    if top_hits:  # 如果存在过滤后的有效相似匹配项
                        doc_ids = [h.get("id") for h in top_hits]
                        
                        # --- OPTIMIZATION: Consolidate Database Calls (Load complete metadata in one query - 优化1) ---
                        # 单次数据库批量反查：一次性拉取候选 Chunks 及其对应文档的所有关联数据（内容、序号、文档标题等）
                        candidate_metadata = {}
                        if doc_ids:
                            def fetch_candidate_metadata(ids, tenant):
                                placeholders = ",".join("%s" for _ in ids)
                                sql = f"""
                                    SELECT c.id, c.content, c.chunk_index, d.id AS doc_id, d.title AS doc_title 
                                    FROM chunks c 
                                    JOIN documents d ON d.id = c.document_id 
                                    WHERE c.id IN ({placeholders}) AND d.tenant_id = %s
                                """
                                with get_conn() as conn:
                                    with conn.cursor() as cursor:
                                        cursor.execute(sql, ids + [tenant])
                                        return cursor.fetchall()
                            try:
                                rows = await asyncio.to_thread(fetch_candidate_metadata, doc_ids, tenant_id)
                                for r in rows:
                                    candidate_metadata[r["id"]] = r

                                print(f"[RAG STEP 4] supplement candidate_metadata： {len(candidate_metadata)}")
                                """
                                {
                                    "f454a408-6761-45b1-bb11-82778c952e30": {
                                        "id": "f454a408-6761-45b1-bb11-82778c952e30",
                                        "content": "的投资范围、策略逻辑与风险收益特征，能够更准确地反映产品定位。对于基民而言",
                                        "chunk_index": 7,
                                        "doc_id": "0877ade9-08fb-43cb-b356-609f6b577835",
                                        "doc_title": "长新闻"
                                    },
                                    "2229e449-5a8b-4c0d-9510-a2f06df7d206": {
                                        "id": "2229e449-5a8b-4c0d-9510-a2f06df7d206",
                                        "content": "明星妈妈可以享受9折的商品优惠",
                                        "chunk_index": 0,
                                        "doc_id": "0de44f29-05df-4b9a-991c-d1270c967532",
                                        "doc_title": "明星妈妈"
                                    },
                                }
                                """
                            except Exception as db_err:
                                print(f"[RAG] Failed to fetch candidate metadata: {db_err}")

                        # 3. 运行 Rerank 重排管道
                        # 如果配置了 Rerank 密钥，使用 Cross-Encoder 二次精细化对比评分
                        reranked_hits = top_hits
                        if use_reranker and candidate_metadata:
                            candidate_chunks = []
                            for hit in top_hits:
                                cid = hit.get("id")
                                if cid in candidate_metadata:
                                    candidate_chunks.append({
                                        "id": cid,
                                        "content": candidate_metadata[cid]["content"],
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
                                    http_client = get_http_client()
                                    res = await http_client.post(rerank_url, json=body, headers=headers, timeout=3.0)
                                    res.raise_for_status()
                                    reranked_data = res.json()
                                    
                                    results = (
                                        reranked_data.get("results", [])
                                        if isinstance(reranked_data, dict)
                                        else []
                                    )
                                    filtered_chunks = []
                                    for r in results:
                                        if not isinstance(r, dict):
                                            continue
                                        idx = r.get("index")
                                        if idx is None or not isinstance(idx, (int, float)):
                                            continue
                                        idx = int(idx)
                                        score = r.get("relevance_score", 0.0)
                                        # 过滤低于 Reranker 阈值的文档
                                        if reranker_threshold is not None and score < reranker_threshold:
                                            continue
                                        if 0 <= idx < len(candidate_chunks):
                                            c = candidate_chunks[idx]
                                            c["score"] = score
                                            filtered_chunks.append(c)
                                    
                                    # 构建最终重排后的 top-k 节点列表
                                    reranked_hits = []
                                    for rc in filtered_chunks[:top_k]:
                                        reranked_hits.append({
                                            "id": rc["id"],
                                            "distance": rc["score"]
                                        })
                                    print(f"[RAG STEP 5] Reranker model '{reranker_model}' returned {len(filtered_chunks)} documents above threshold {reranker_threshold}")
                                    # print(json.dumps(filtered_chunks, ensure_ascii=False, indent=2))

                                except Exception as re_err:
                                    print(f"[RAG] Reranking request failed or timed out: {re_err}")
                                    reranked_hits = top_hits[:top_k]
                        else:
                            reranked_hits = top_hits[:top_k]

                        # --- OPTIMIZATION: Read Chunk Metadata from Memory (No DB query required - 优化1) ---
                        # 直接从内存 `candidate_metadata` 中快速抽取最终重排过后的 chunk 数据，无需二次数据库调用
                        final_chunk_ids = [h["id"] for h in reranked_hits]
                        chunk_metadata = {cid: candidate_metadata[cid] for cid in final_chunk_ids if cid in candidate_metadata}

                        # 邻近块扩展策略：收集最终匹配 chunks 的前后邻接索引，用于扩充上下文滑窗
                        adjacent_chunk_map = {}
                        query_tuples = []
                        for hit in reranked_hits:
                            r = chunk_metadata.get(hit.get("id"))
                            if r:
                                doc_id = r["doc_id"]
                                c_idx = r["chunk_index"]
                                if c_idx > 0:
                                    query_tuples.append((doc_id, c_idx - 1))
                                query_tuples.append((doc_id, c_idx + 1))
                        
                        # 邻接块查询目标去重
                        query_tuples = list(set(query_tuples))
                        
                        if query_tuples:
                            # 从数据库批量加载所有邻近块的文本
                            def fetch_adjacent_chunks(tuples):
                                conds = []
                                params = []
                                for d_id, c_idx in tuples:
                                    conds.append("(document_id = %s AND chunk_index = %s)")
                                    params.extend([d_id, c_idx])
                                adj_sql = f"""
                                    SELECT document_id, chunk_index, content 
                                    FROM chunks 
                                    WHERE {" OR ".join(conds)}
                                """
                                with get_conn() as conn:
                                    with conn.cursor() as cursor:
                                        cursor.execute(adj_sql, params)
                                        return cursor.fetchall()

                            try:
                                adj_rows = await asyncio.to_thread(fetch_adjacent_chunks, query_tuples)
                                for row in adj_rows:
                                    adjacent_chunk_map[(row["document_id"], row["chunk_index"])] = row["content"]
                            except Exception as adj_err:
                                print(f"[RAG] Failed to fetch adjacent chunks: {adj_err}")
                        print(f"adjacent_chunk len: {len(adjacent_chunk_map)}")

                        # 构建 (doc_id, chunk_index) → content 的统一内容索引表，供后续滑窗拼接使用
                        content_by_index = {}
                        # 第一步：写入命中块（来自内存缓存 chunk_metadata，无需再查库）
                        for detail in chunk_metadata.values():
                            content_by_index[(detail["doc_id"], detail["chunk_index"])] = detail["content"]
                        # 第二步：写入邻近扩展块（来自数据库补查的 adjacent_chunk_map），若与命中块索引重叠则覆盖（实际不会重叠）
                        for (d_id, idx), content in adjacent_chunk_map.items():
                            content_by_index[(d_id, idx)] = content

                        print(f"[RAG STEP 6] final enhanced doc: {len(content_by_index)}")
                        print(json.dumps({f"{k[0]}:{k[1]}": v for k, v in content_by_index.items()}, ensure_ascii=False, indent=2))

                        # 初始化返回的数据源列表，用于返回给前端或调用方
                        sources = []
                        # 统计每个文档所召回和扩展的块索引集
                        doc_indices = {}
                        doc_titles = {}
                        doc_max_scores = {}

                        for hit in reranked_hits:
                            chunk_id = hit.get("id")
                            r = chunk_metadata.get(chunk_id)
                            if not r:
                                continue
                            doc_id = r["doc_id"]
                            c_idx = r["chunk_index"]
                            title = r["doc_title"] or "未知文档"
                            score = hit.get("distance", 0.0)

                            doc_titles[doc_id] = title
                            doc_max_scores[doc_id] = max(doc_max_scores.get(doc_id, 0.0), score)

                            if doc_id not in doc_indices:
                                doc_indices[doc_id] = set()

                            # 将当前块和合法的邻近块加入该文档的待拼接索引列表
                            for idx in (c_idx - 1, c_idx, c_idx + 1):
                                if (doc_id, idx) in content_by_index:
                                    doc_indices[doc_id].add(idx)

                            # 组装返回数据源 sources（提供给前端做引用高亮和来源卡片）
                            sources.append({
                                "documentId": doc_id,
                                "documentTitle": title,
                                "chunkContent": r["content"][:220] + ("…" if len(r["content"]) > 220 else ""),
                                "score": round(score, 4),
                            })

                        # 将零散索引合并为连续区间的辅助函数
                        def merge_contiguous_indexes(indexes: list[int]) -> list[tuple[int, int]]:
                            if not indexes:
                                return []
                            sorted_idx = sorted(list(set(indexes)))
                            ranges = []
                            start = sorted_idx[0]
                            prev = start
                            for idx in sorted_idx[1:]:
                                if idx == prev + 1:
                                    prev = idx
                                else:
                                    ranges.append((start, prev))
                                    start = idx
                                    prev = idx
                            ranges.append((start, prev))
                            return ranges

                        # 按文档召回的最高相似分对文档进行降序排序
                        sorted_docs = sorted(doc_max_scores.keys(), key=lambda d: doc_max_scores[d], reverse=True)
                        
                        # 初始化上下文部分的文本块列表，用于合成最终 key context 字符串
                        context_parts = []
                        char_budget = 8000  # 设定 8000 字符的软限制保护 Prompt 不超上限
                        current_chars = 0

                        # 将匹配到的文档以规范 of XML 节点形式拼入 Context 中，并实时计算字符预算
                        for doc_id in sorted_docs:
                            if current_chars >= char_budget:
                                break
                            title = doc_titles[doc_id]
                            fetched_indexes = list(doc_indices.get(doc_id, []))
                            ranges = merge_contiguous_indexes(fetched_indexes)
                            
                            doc_segments = []
                            for start_idx, end_idx in ranges:
                                merged_range_content = ""
                                for idx in range(start_idx, end_idx + 1):
                                    chunk_content = content_by_index.get((doc_id, idx))
                                    if chunk_content:
                                        if not merged_range_content:
                                            merged_range_content = chunk_content
                                        else:
                                            # --- OPTIMIZATION: Overlap Deduplication (去重拼接 - 优化3) ---
                                            # 当拼接连续索引的块文本时，使用 merge_overlapping_strings 去除 chunk 之间的 overlap 重合内容
                                            merged_range_content = merge_overlapping_strings(merged_range_content, chunk_content)
                                if merged_range_content:
                                    doc_segments.append(merged_range_content)

                            if doc_segments:
                                # 同一个文档内不同连续区间使用隔离线相连
                                doc_body = "\n\n---\n\n".join(doc_segments)
                                xml_node = f'<document title="{title}">\n{doc_body}\n</document>'
                                
                                # 字符预算超限保护，支持在边界截断，防止系统出错
                                if current_chars + len(xml_node) > char_budget:
                                    remaining_budget = char_budget - current_chars
                                    if remaining_budget > 500:
                                        xml_node = f'<document title="{title}">\n{doc_body[:remaining_budget]}... [已截断]\n</document>'
                                        context_parts.append(xml_node)
                                    break
                                
                                context_parts.append(xml_node)
                                current_chars += len(xml_node)

                        # 将所有来源的文本片段拼接成一个大字符串，作为最终的 context
                        context = "\n".join(context_parts)
                        # 在控制台打印当前 Milvus 检索并返回的结果数量
                        print(f"[[RAG STEP 7] Milvus search returned {len(sources)} results. Character length: {len(context)}")
                        print(json.dumps(sources, ensure_ascii=False, indent=2))

                        # 返回包含拼接好的上下文 context 和数据源 sources 的字典
                        return {"context": context, "sources": sources}
        # 捕捉在 Milvus 查询或 MySQL 查询中发生的任何异常
        except Exception as e:
            # 在控制台打印异常信息
            print(f"[RAG] Milvus search failed: {e}")

    # 如果无法生成查询的向量，或者搜索过程中发生异常/无结果，返回默认的空上下文和空数据源字典
    return {"context": "", "sources": []}
