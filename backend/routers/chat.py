import json  # 导入 json 模块，用于序列化流式响应数据
import os  # 导入 os 模块，用于读取环境变量和处理文件路径
from typing import AsyncGenerator, Optional  # 导入类型提示：异步生成器和可选类型

from dotenv import load_dotenv  # 导入 dotenv，用于从 .env 文件加载环境变量
from fastapi import APIRouter, HTTPException, Depends  # 导入 APIRouter, HTTP 异常类和依赖注入
from fastapi.responses import StreamingResponse  # 导入流式响应类，用于返回流式数据
from pydantic import BaseModel  # 导入 Pydantic 基类，用于定义请求/响应数据模型

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from services.rag import RAGRetriever, validate_safe_id
from services.db import create_session, add_session_message
from dependencies import get_current_tenant_id

# Load env from project root's .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env.local"))  # 加载项目根目录下 .env.local 文件中的环境变量

# ── LangChain Global Models & Chains Configuration ─────────────────────────

# Query Rewrite Model & Chain
rewrite_llm = ChatOpenAI(
    model=os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL"),
    openai_api_base=os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1"),
    openai_api_key=os.getenv("MIMO_API_KEY", ""),
    temperature=0.3
)

REWRITE_SYSTEM_PROMPT = """你是一个智能检索助手。请结合以下对话历史和用户最新的问题，生成一个最适合用来检索知识库的核心问题（Standalone Search Query）。
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
核心问题：

【示例三】
用户最新的问题：No.23214
核心问题：No.23214"""

rewrite_prompt_tmpl = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM_PROMPT),
    ("human", "【对话历史】\n{chat_history}\n\n【用户最新的问题】\n{query}\n\n【核心问题】")
])

rewrite_chain = rewrite_prompt_tmpl | rewrite_llm | StrOutputParser()

# Main Chat Model & Chain
chat_llm = ChatOpenAI(
    model=os.getenv("MIMO_MODEL", "Xiaomi/MiMo-7B-RL"),
    openai_api_base=os.getenv("MIMO_BASE_URL", "https://api.siliconflow.cn/v1"),
    openai_api_key=os.getenv("MIMO_API_KEY", ""),
    temperature=0.7
)

chat_prompt_tmpl = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("placeholder", "{chat_history_messages}"),
    ("human", "{question}")
])

chat_chain = chat_prompt_tmpl | chat_llm | StrOutputParser()


# ── Pydantic models ───────────────────────────────────────────────────────

class Message(BaseModel):  # 定义单条聊天消息的数据模型
    role: str  # 消息角色（如 user、assistant、system）
    content: str  # 消息内容文本


class ChatRequest(BaseModel):  # 定义聊天接口请求体的数据模型
    messages: list[Message]  # 消息列表，包含完整的对话历史
    ragEnabled: bool = False  # 是否启用 RAG 检索增强，默认关闭
    sessionId: Optional[str] = None  # 会话 ID，可为空（表示新建会话）
    kbId: Optional[str] = None  # 选中的知识库 ID


# ── Router Setup ─────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/api/chat")  # 注册 POST /api/chat 路由
async def chat(request: ChatRequest, tenant_id: str = Depends(get_current_tenant_id)) -> StreamingResponse:  # 聊天接口处理函数，返回流式响应
    print(f"[chat] tenant_id={tenant_id} ragEnabled={request.ragEnabled} kbId={request.kbId} messages={len(request.messages)}")  # 打印调试日志

    # 查找最后一条用户消息及其索引
    last_user_idx = -1
    for idx in range(len(request.messages) - 1, -1, -1):
        if request.messages[idx].role == "user":
            last_user_idx = idx
            break

    if last_user_idx == -1:
        raise HTTPException(status_code=400, detail="No user message found")

    last_user = request.messages[last_user_idx]

    # Session: create if needed, save user message
    session_id = request.sessionId  # 读取请求中的会话 ID
    if not session_id:  # 如果没有传入会话 ID
        session_id = create_session(last_user.content[:30])  # 用用户消息前 30 字符作为标题创建新会话
    add_session_message(session_id, "user", last_user.content)  # 将用户消息写入该会话

    system_prompt = "你是一个有帮助的 AI 助手，请用中文回答问题。"  # 默认系统提示词
    sources: list[dict] = []  # 用于存储 RAG 检索到的引用来源

    if request.ragEnabled:
        kb_id = request.kbId or "1"
        if not validate_safe_id(kb_id):
            raise HTTPException(status_code=400, detail="Invalid kb_id format")

        search_query = last_user.content
        history_messages = request.messages[:last_user_idx]

        if history_messages:
            # 仅保留最近 5 轮对话进行重写
            history_slice = history_messages[-5:]
            formatted_history = ""
            for msg in history_slice:
                role_str = "用户" if msg.role == "user" else "助手"
                formatted_history += f"{role_str}: {msg.content}\n"

            try:
                # 调用 LangChain 查询重写链
                print("[RAG] Rewriting query using LangChain...")
                rewritten = await rewrite_chain.ainvoke({
                    "chat_history": formatted_history,
                    "query": last_user.content
                })

                # 处理大语言模型的思考标签并提取最终 Query
                if rewritten:
                    if "<think>" in rewritten:
                        if "</think>" in rewritten:
                            rewritten = rewritten.split("</think>")[-1].strip()
                        else:
                            rewritten = rewritten.split("<think>")[0].strip()

                    rewritten = rewritten.strip('"\'\'`').strip()

                if rewritten == "None" or rewritten == "":
                    search_query = None
                elif rewritten and len(rewritten) < 100:
                    search_query = rewritten
                else:
                    search_query = None
                print(f"[RAG] Rewritten query: '{last_user.content}' -> '{search_query}'")
            except Exception as e:
                print(f"[RAG] Query rewriting failed: {e}")
                search_query = last_user.content

        # 如果存在有效检索问题，调用 Custom Retriever
        if search_query:
            try:
                print(f"[RAG] Retrieving from knowledge base using LangChain RAGRetriever...")
                retriever = RAGRetriever(kb_id=kb_id, tenant_id=tenant_id)
                retrieved_docs = await retriever.ainvoke(search_query)
                if retrieved_docs and retrieved_docs[0].page_content:
                    doc = retrieved_docs[0]
                    sources = doc.metadata.get("sources", [])
                    system_prompt = (
                        "你是一个有帮助的 AI 助手。请优先基于以下知识库内容回答用户的问题。"
                        "如知识库内容不足以完整回答，直接回复不知道即可，切勿编造信息\n\n"
                        f"===知识库===\n{doc.page_content}\n===END===\n\n请用中文回答。"
                    )
            except Exception as e:
                print(f"[RAG] Custom retriever execution failed: {e}")

    # 构造历史对话消息供 LangChain 提示词模版填充
    chat_history_messages = []
    for m in request.messages[:last_user_idx]:
        if m.role == "user":
            chat_history_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            chat_history_messages.append(AIMessage(content=m.content))

    async def generate() -> AsyncGenerator[str, None]:  # 定义异步生成器，用于流式输出响应
        if session_id:
            yield f'2:[{json.dumps({"type": "session", "sessionId": session_id})}]\n'  # 推送会话信息给前端

        full_content = ""  # 用于累积完整的助手回复内容
        inputs = {
            "system_prompt": system_prompt,
            "chat_history_messages": chat_history_messages,
            "question": last_user.content
        }

        try:
            # 运行 LangChain Chat Chain 并流式输出
            async for chunk in chat_chain.astream(inputs):
                full_content += chunk
                yield f"0:{json.dumps(chunk, ensure_ascii=False)}\n"  # 推送文本增量帧给前端
        except Exception as e:
            print(f"[chat] Chat chain generation failed: {e}")
            yield f"0:{json.dumps(f'生成失败: {str(e)}', ensure_ascii=False)}\n"

        if session_id and full_content:
            add_session_message(session_id, "assistant", full_content)  # 将助手回复写入会话

        yield 'd:{"finishReason":"stop"}\n'  # 推送流结束帧

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Vercel-AI-Data-Stream": "v1",
        },
    )
