import json
import os
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from .db import (
    delete_document_by_id, get_all_documents, get_stats, init_db,
    create_session, get_sessions, get_session_messages,
    add_session_message, delete_session_by_id,
)
from .rag import add_document, search_knowledge

# Load env from project root's .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.local"))

app = FastAPI(title="MiMo RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Vercel-AI-Data-Stream"],
)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()


# ── Pydantic models ───────────────────────────────────────────────────────


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    ragEnabled: bool = False
    sessionId: Optional[str] = None


class AddDocumentRequest(BaseModel):
    title: str
    content: str


# ── Chat endpoint ─────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    print(f"[chat] ragEnabled={request.ragEnabled} messages={len(request.messages)}")
    client = AsyncOpenAI(
        base_url=os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=os.getenv("MIMO_API_KEY", ""),
    )
    model = os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL")

    system_prompt = "你是一个有帮助的 AI 助手，请用中文回答问题。"
    sources: list[dict] = []

    if request.ragEnabled and request.messages:
        last_user = next(
            (m for m in reversed(request.messages) if m.role == "user"), None
        )
        if last_user:
            rag = await search_knowledge(last_user.content)
            if rag["context"]:
                sources = rag["sources"]
                system_prompt = (
                    "你是一个有帮助的 AI 助手。请优先基于以下知识库内容回答用户的问题。"
                    "如知识库内容不足以完整回答，可结合自身知识补充，但需说明哪部分来自知识库，哪部分是推断。\n\n"
                    f"===知识库===\n{rag['context']}\n===END===\n\n请用中文回答。"
                )

    oai_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    # Session: create if needed, save user message
    session_id = request.sessionId
    last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if last_user:
        if not session_id:
            session_id = create_session(last_user.content[:30])
        add_session_message(session_id, "user", last_user.content)

    async def generate() -> AsyncGenerator[str, None]:
        # Vercel AI SDK data stream protocol:
        # 0:"text"   → text delta
        # 2:[item]   → custom data (appended to useChat's `data` array)
        # d:{...}    → stream finish

        if session_id:
            yield f'2:[{json.dumps({"type": "session", "sessionId": session_id})}]\n'

        stream = await client.chat.completions.create(
            model=model,
            messages=oai_messages,  # type: ignore[arg-type]
            stream=True,
        )

        full_content = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                yield f"0:{json.dumps(delta.content, ensure_ascii=False)}\n"

        if session_id and full_content:
            add_session_message(session_id, "assistant", full_content)

        yield 'd:{"finishReason":"stop"}\n'

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Vercel-AI-Data-Stream": "v1",
        },
    )


# ── Knowledge base endpoints ──────────────────────────────────────────────


@app.get("/api/knowledge")
def list_knowledge():
    return {"documents": get_all_documents(), "stats": get_stats()}


@app.post("/api/knowledge")
async def create_knowledge(req: AddDocumentRequest):
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="标题和内容不能为空")
    doc_id = await add_document(req.title.strip(), req.content.strip())
    return {"id": doc_id, "success": True}


@app.delete("/api/knowledge/{doc_id}")
def remove_knowledge(doc_id: str):
    delete_document_by_id(doc_id)
    return {"success": True}


# ── Session endpoints ─────────────────────────────────────────────────────


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": get_sessions()}


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: str):
    return {"messages": get_session_messages(session_id)}


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session_by_id(session_id)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
