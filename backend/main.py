import json  # 导入 json 模块，用于序列化流式响应数据
import os  # 导入 os 模块，用于读取环境变量和处理文件路径
from typing import AsyncGenerator, Optional  # 导入类型提示：异步生成器和可选类型

from dotenv import load_dotenv  # 导入 dotenv，用于从 .env 文件加载环境变量
from fastapi import FastAPI, HTTPException  # 导入 FastAPI 框架核心类和 HTTP 异常类
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件，用于处理跨域请求
from fastapi.responses import StreamingResponse  # 导入流式响应类，用于返回流式数据
from openai import AsyncOpenAI  # 导入 OpenAI 异步客户端，用于调用兼容 OpenAI 协议的模型接口
from pydantic import BaseModel  # 导入 Pydantic 基类，用于定义请求/响应数据模型

from db import (  # 从本地 db 模块导入数据库相关函数
    delete_document_by_id, get_all_documents, get_stats, init_db,  # 文档相关：删除、获取全部、获取统计、初始化数据库
    create_session, get_sessions, get_session_messages,  # 会话相关：创建会话、获取会话列表、获取会话消息
    add_session_message, delete_session_by_id,  # 会话相关：添加消息、删除会话
)
from rag import add_document, search_knowledge  # 从本地 rag 模块导入添加文档和检索知识库的函数

# Load env from project root's .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.local"))  # 加载项目根目录下 .env.local 文件中的环境变量

app = FastAPI(title="MiMo RAG Backend", version="1.0.0")  # 创建 FastAPI 应用实例，设置标题和版本号

app.add_middleware(  # 为应用添加中间件
    CORSMiddleware,  # 使用 CORS 中间件
    allow_origins=["http://localhost:3000"],  # 允许的跨域来源：本地前端开发地址
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["X-Vercel-AI-Data-Stream"],  # 允许前端读取的响应头：标识 AI 数据流协议
)


@app.on_event("startup")  # 注册应用启动事件回调
async def on_startup() -> None:  # 定义启动时执行的异步函数
    init_db()  # 初始化数据库（建表等）


# ── Pydantic models ───────────────────────────────────────────────────────


class Message(BaseModel):  # 定义单条聊天消息的数据模型
    role: str  # 消息角色（如 user、assistant、system）
    content: str  # 消息内容文本


class ChatRequest(BaseModel):  # 定义聊天接口请求体的数据模型
    messages: list[Message]  # 消息列表，包含完整的对话历史
    ragEnabled: bool = False  # 是否启用 RAG 检索增强，默认关闭
    sessionId: Optional[str] = None  # 会话 ID，可为空（表示新建会话）


class AddDocumentRequest(BaseModel):  # 定义新增知识库文档接口请求体的数据模型
    title: str  # 文档标题
    content: str  # 文档内容


# ── Chat endpoint ─────────────────────────────────────────────────────────


@app.post("/api/chat")  # 注册 POST /api/chat 路由
async def chat(request: ChatRequest) -> StreamingResponse:  # 聊天接口处理函数，返回流式响应
    print(f"[chat] ragEnabled={request.ragEnabled} messages={len(request.messages)}")  # 打印调试日志：是否启用 RAG 及消息数量
    client = AsyncOpenAI(  # 创建 OpenAI 异步客户端实例
        base_url=os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1"),  # 模型服务的 base_url，默认指向 SiliconFlow
        api_key=os.getenv("MIMO_API_KEY", ""),  # 模型服务的 API Key，从环境变量读取
    )
    model = os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL")  # 使用的模型名称，默认 MiMo-7B-RL

    system_prompt = "你是一个有帮助的 AI 助手，请用中文回答问题。"  # 默认系统提示词
    sources: list[dict] = []  # 用于存储 RAG 检索到的引用来源

    if request.ragEnabled and request.messages:  # 如果启用了 RAG 且消息列表非空
        last_user = next(  # 查找最后一条用户消息
            (m for m in reversed(request.messages) if m.role == "user"), None  # 从后往前遍历，找到第一条 role 为 user 的消息
        )
        if last_user:  # 如果找到了用户消息
            rag = await search_knowledge(last_user.content)  # 基于用户消息内容检索知识库
            if rag["context"]:  # 如果检索到了上下文内容
                sources = rag["sources"]  # 保存检索到的来源列表
                system_prompt = (  # 重新构造系统提示词，注入知识库内容
                    "你是一个有帮助的 AI 助手。请优先基于以下知识库内容回答用户的问题。"  # 提示模型优先使用知识库
                    "如知识库内容不足以完整回答，直接回复不知道即可，切勿编造信息\n\n"  # 提示模型在不足时如何处理
                    f"===知识库===\n{rag['context']}\n===END===\n\n请用中文回答。"  # 插入知识库内容并要求用中文回答
                )

    print(system_prompt)

    oai_messages = [{"role": "system", "content": system_prompt}] + [  # 构造发送给模型的消息列表，先放系统提示词
        {"role": m.role, "content": m.content} for m in request.messages  # 再依次附加用户传入的历史消息
    ]

    # Session: create if needed, save user message
    session_id = request.sessionId  # 读取请求中的会话 ID
    last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)  # 再次查找最后一条用户消息（用于落库）
    if last_user:  # 如果存在用户消息
        if not session_id:  # 如果没有传入会话 ID
            session_id = create_session(last_user.content[:30])  # 用用户消息前 30 字符作为标题创建新会话
        add_session_message(session_id, "user", last_user.content)  # 将用户消息写入该会话

    async def generate() -> AsyncGenerator[str, None]:  # 定义异步生成器，用于流式输出响应
        # Vercel AI SDK data stream protocol:
        # 0:"text"   → text delta
        # 2:[item]   → custom data (appended to useChat's `data` array)
        # d:{...}    → stream finish

        if session_id:  # 如果存在会话 ID
            yield f'2:[{json.dumps({"type": "session", "sessionId": session_id})}]\n'  # 先推送会话信息给前端（自定义数据帧）

        stream = await client.chat.completions.create(  # 调用模型接口，发起流式补全请求
            model=model,  # 指定使用的模型
            messages=oai_messages,  # type: ignore[arg-type]  # 传入构造好的消息列表
            stream=True,  # 开启流式返回
        )

        full_content = ""  # 用于累积完整的助手回复内容
        async for chunk in stream:  # 异步遍历模型返回的每个数据块
            if not chunk.choices:  # 如果该数据块不包含任何选项
                continue  # 跳过此次循环
            delta = chunk.choices[0].delta  # 取出本次增量内容
            if delta.content:  # 如果增量内容非空
                full_content += delta.content  # 累加到完整内容中
                yield f"0:{json.dumps(delta.content, ensure_ascii=False)}\n"  # 推送文本增量帧给前端

        if session_id and full_content:  # 如果存在会话 ID 且已生成回复内容
            add_session_message(session_id, "assistant", full_content)  # 将助手回复写入该会话

        yield 'd:{"finishReason":"stop"}\n'  # 推送流结束帧，标记完成原因为 stop

    return StreamingResponse(  # 返回流式响应对象
        generate(),  # 传入上面定义的异步生成器
        media_type="text/plain",  # 设置响应媒体类型为纯文本
        headers={  # 设置自定义响应头
            "Cache-Control": "no-cache",  # 禁止缓存
            "Connection": "keep-alive",  # 保持连接
            "X-Vercel-AI-Data-Stream": "v1",  # 标识为 Vercel AI 数据流协议 v1
        },
    )


# ── Knowledge base endpoints ──────────────────────────────────────────────


@app.get("/api/knowledge")  # 注册 GET /api/knowledge 路由
def list_knowledge():  # 获取知识库列表的处理函数
    return {"documents": get_all_documents(), "stats": get_stats()}  # 返回所有文档及统计信息


@app.post("/api/knowledge")  # 注册 POST /api/knowledge 路由
async def create_knowledge(req: AddDocumentRequest):  # 新增知识库文档的处理函数
    if not req.title.strip() or not req.content.strip():  # 校验标题和内容是否为空（去除首尾空白后）
        raise HTTPException(status_code=400, detail="标题和内容不能为空")  # 参数无效时抛出 400 错误
    doc_id = await add_document(req.title.strip(), req.content.strip())  # 调用 rag 模块添加文档，返回文档 ID
    return {"id": doc_id, "success": True}  # 返回新建文档的 ID 及成功标志


@app.delete("/api/knowledge/{doc_id}")  # 注册 DELETE /api/knowledge/{doc_id} 路由
def remove_knowledge(doc_id: str):  # 删除指定知识库文档的处理函数
    delete_document_by_id(doc_id)  # 根据文档 ID 执行删除
    return {"success": True}  # 返回成功标志


# ── Session endpoints ─────────────────────────────────────────────────────


@app.get("/api/sessions")  # 注册 GET /api/sessions 路由
def list_sessions():  # 获取会话列表的处理函数
    return {"sessions": get_sessions()}  # 返回所有会话


@app.get("/api/sessions/{session_id}/messages")  # 注册 GET /api/sessions/{session_id}/messages 路由
def session_messages(session_id: str):  # 获取指定会话消息记录的处理函数
    return {"messages": get_session_messages(session_id)}  # 返回该会话下的所有消息


@app.delete("/api/sessions/{session_id}")  # 注册 DELETE /api/sessions/{session_id} 路由
def remove_session(session_id: str):  # 删除指定会话的处理函数
    delete_session_by_id(session_id)  # 根据会话 ID 执行删除
    return {"success": True}  # 返回成功标志


if __name__ == "__main__":  # 当该文件作为主程序直接运行时
    import uvicorn  # 导入 uvicorn 服务器
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # 启动 ASGI 服务，监听所有网卡的 8000 端口，开启热重载
