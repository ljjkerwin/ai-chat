from fastapi import APIRouter  # 导入 APIRouter

from services.db import (  # 从本地 db 模块导入会话相关数据库函数
    get_sessions,
    get_sessions_count,
    get_session_messages,
    delete_session_by_id,
)

# ── Router Setup ─────────────────────────────────────────────────────────

router = APIRouter()


# ── Session endpoints ─────────────────────────────────────────────────────

@router.get("/api/sessions")  # 注册 GET /api/sessions 路由
def list_sessions(page: int = 1, page_size: int = 30):  # 获取会话列表的处理函数，支持分页获取
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 30
    offset = (page - 1) * page_size
    sessions = get_sessions(limit=page_size, offset=offset)
    total = get_sessions_count()
    return {
        "sessions": sessions,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": offset + len(sessions) < total
    }


@router.get("/api/sessions/{session_id}/messages")  # 注册 GET /api/sessions/{session_id}/messages 路由
def session_messages(session_id: str):  # 获取指定会话消息记录的处理函数
    return {"messages": get_session_messages(session_id)}  # 返回该会话下的所有消息


@router.delete("/api/sessions/{session_id}")  # 注册 DELETE /api/sessions/{session_id} 路由
def remove_session(session_id: str):  # 删除指定会话的处理函数
    delete_session_by_id(session_id)  # 根据会话 ID 执行删除
    return {"success": True}  # 返回成功标志
