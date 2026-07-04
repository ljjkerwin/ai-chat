from typing import Optional  # 导入类型提示：可选类型
from uuid import uuid4  # 导入 uuid 用于生成知识库 ID

from fastapi import APIRouter, HTTPException, Depends  # 导入 APIRouter, HTTP 异常类和依赖注入
from pydantic import BaseModel  # 导入 Pydantic 基类，用于定义请求/响应数据模型

from services.db import (  # 导入数据库相关函数
    get_knowledge_bases,
    create_knowledge_base,
    delete_knowledge_base_by_id,
    get_all_documents,
    get_stats,
)
from services.rag import (  # 导入 RAG 相关函数
    validate_safe_id,
    delete_knowledge_base_vectors,
    add_document,
    delete_document,
)
from dependencies import get_current_tenant_id

# ── Router Setup ─────────────────────────────────────────────────────────

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────

class CreateKBRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class AddDocumentRequest(BaseModel):  # 定义新增知识库文档接口请求体的数据模型
    title: str  # 文档标题
    content: str  # 文档内容
    kbId: str  # 目标知识库 ID


# ── Knowledge base endpoints ──────────────────────────────────────────────

@router.get("/api/knowledge-bases")
def list_knowledge_bases(tenant_id: str = Depends(get_current_tenant_id)):
    return {"knowledgeBases": get_knowledge_bases(tenant_id)}


@router.post("/api/knowledge-bases")
def create_new_knowledge_base(req: CreateKBRequest, tenant_id: str = Depends(get_current_tenant_id)):
    kb_id = str(uuid4())
    if not validate_safe_id(kb_id) or not validate_safe_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid identifiers generated")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    create_knowledge_base(kb_id, req.name.strip(), req.description.strip() if req.description else "", tenant_id)
    return {"id": kb_id, "success": True}


@router.delete("/api/knowledge-bases/{kb_id}")
def remove_knowledge_base(kb_id: str, tenant_id: str = Depends(get_current_tenant_id)):
    if not validate_safe_id(kb_id):
        raise HTTPException(status_code=400, detail="Invalid kb_id format")
    # First delete from Milvus
    delete_knowledge_base_vectors(kb_id, tenant_id)
    # Next delete from DB
    delete_knowledge_base_by_id(kb_id, tenant_id)
    return {"success": True}


@router.get("/api/knowledge")  # 注册 GET /api/knowledge 路由
def list_knowledge(kb_id: str = "1", tenant_id: str = Depends(get_current_tenant_id)):  # 获取知识库列表的处理函数
    if not validate_safe_id(kb_id):
        raise HTTPException(status_code=400, detail="Invalid kb_id format")
    return {"documents": get_all_documents(kb_id, tenant_id), "stats": get_stats(kb_id, tenant_id)}  # 返回所有文档及统计信息


@router.post("/api/knowledge")  # 注册 POST /api/knowledge 路由
async def create_knowledge(req: AddDocumentRequest, tenant_id: str = Depends(get_current_tenant_id)):  # 新增知识库文档的处理函数
    if not req.title.strip() or not req.content.strip():  # 校验标题和内容是否为空（去除首尾空白后）
        raise HTTPException(status_code=400, detail="标题和内容不能为空")  # 参数无效时抛出 400 错误
    if not validate_safe_id(req.kbId):
         raise HTTPException(status_code=400, detail="Invalid kb_id format")
    doc_id = await add_document(req.title.strip(), req.content.strip(), req.kbId, tenant_id)  # 调用 rag 模块添加文档，返回文档 ID
    return {"id": doc_id, "success": True}  # 返回新建文档 of ID 及成功标志


@router.delete("/api/knowledge/{doc_id}")  # 注册 DELETE /api/knowledge/{doc_id} 路由
def remove_knowledge(doc_id: str, tenant_id: str = Depends(get_current_tenant_id)):  # 删除指定知识库文档的处理函数
    if not validate_safe_id(doc_id):
         raise HTTPException(status_code=400, detail="Invalid doc_id format")
    delete_document(doc_id, tenant_id)  # 根据文档 ID 执行删除从 SQLite 和 Milvus 中删除
    return {"success": True}  # 返回成功标志
