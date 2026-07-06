import json  # 导入 json 模块，用于序列化流式响应数据
import os  # 导入 os 模块，用于读取环境变量和处理文件路径
import re
from typing import AsyncGenerator, Optional  # 导入类型提示：异步生成器和可选类型

from dotenv import load_dotenv  # 导入 dotenv，用于从 .env 文件加载环境变量
from fastapi import APIRouter, HTTPException, Depends  # 导入 APIRouter, HTTP 异常类和依赖注入
from fastapi.responses import StreamingResponse  # 导入流式响应类，用于返回流式数据
from pydantic import BaseModel  # 导入 Pydantic 基类，用于定义请求/响应数据模型

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

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


@tool
def get_current_weather(location: str) -> str:
    """获取指定城市的当前天气情况。当用户询问某个城市的天气、温度等相关信息时使用该工具。"""
    print(f"[Tool] get_current_weather called with location='{location}'")
    try:
        import urllib.parse
        import httpx
        encoded_location = urllib.parse.quote(location)
        url = f"https://wttr.in/{encoded_location}?format=4"
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            result = response.text.strip()
            if result and "<html" not in result.lower():
                print(f"[Tool] Real weather fetched successfully: {result}")
                return f"实时天气数据：{result}"
            else:
                return f"暂未获取到 {location} 的最新天气，接口返回了非结构化内容。"
        else:
            return f"获取 {location} 天气失败，服务响应状态码: {response.status_code}。"
    except httpx.TimeoutException:
        print(f"[Tool] Real-time weather fetch timed out.")
        return f"获取天气超时，无法连接到外部天气服务，请稍后再试。"
    except Exception as e:
        print(f"[Tool] Real-time weather fetch failed: {e}.")
        return f"获取 {location} 天气失败，原因：{str(e)}。"


@tool
def get_exchange_rate(base_currency: str = "CNY") -> str:
    """获取主要货币相对于基础货币的最新实时汇率。支持输入基础货币符号，如 USD, EUR, CNY 等，默认基础货币为 USD。"""
    print(f"[Tool] get_exchange_rate called with base_currency='{base_currency}'")
    try:
        import httpx
        base = base_currency.upper().strip()
        url = f"https://open.er-api.com/v6/latest/{base}"
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("result") == "success":
                rates = data.get("rates", {})
                target_currencies = ["CNY", "USD", "EUR", "GBP", "JPY", "HKD", "AUD", "CAD", "KRW", "SGD", "CHF", "THB", "TWD", "RUB"]
                filtered_rates = {cur: rates[cur] for cur in target_currencies if cur in rates}
                time_update = data.get("time_last_update_utc", "")
                return f"基础货币: {base}, 更新时间: {time_update}, 部分实时汇率：{filtered_rates}"
            else:
                return f"获取汇率失败，API 返回错误: {data.get('error-type', 'unknown')}"
        else:
            return f"获取汇率接口请求失败，状态码: {response.status_code}。"
    except Exception as e:
        print(f"[Tool] Exchange rate fetch failed: {e}.")
        return f"获取汇率失败，原因: {str(e)}"


chat_llm_with_tools = chat_llm.bind_tools([get_current_weather, get_exchange_rate])


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

    system_prompt = (
        "你是一个有帮助的 AI 助手，请用中文回答问题。\n"
        "【重要约束】不要主动向用户透露、介绍、列举或暗示你拥有任何特定的外部工具、插件、API 或功能。只有当用户明确要求你执行相关操作或查询时，你才静默调用工具来回答。在日常问候或普通闲聊中，请直接进行自然友好的回复，不要列出你的功能列表或进行自我宣传。"
    )
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
                print(f"[RAG STEP 1] Rewritten query: '{last_user.content}' -> '{search_query}'")
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
                        f"===知识库===\n{doc.page_content}\n===END===\n\n请用中文回答。\n"
                        "【重要约束】不要主动向用户透露、介绍、列举或暗示你拥有任何特定的外部工具、插件、API 或功能。只有当用户明确要求你执行相关操作或查询时，你才静默调用相关工具。"
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
        
        messages = [
            SystemMessage(content=system_prompt),
            *chat_history_messages,
            HumanMessage(content=last_user.content)
        ]

        try:
            print("[chat] Invoking LLM with tools (streaming)...")
            tool_call_detected = False
            text_buffer = ""
            response_chunk = None

            async for chunk in chat_llm_with_tools.astream(messages):
                # print(f"chunk: {chunk}")
                # 累加/合并所有流式 chunk，以便在循环结束后拼装成完整的 AIMessage 响应
                if response_chunk is None:
                    response_chunk = chunk
                else:
                    response_chunk += chunk

                # 如果当前 chunk 包含工具调用片段，标记已检测到工具调用
                if chunk.tool_call_chunks:
                    tool_call_detected = True

                # 如果当前 chunk 包含文本内容，则将其累加到文本缓存区
                if chunk.content:
                    text_buffer += chunk.content
                    
                    if not tool_call_detected:
                        stripped_buffer = text_buffer.strip()
                        # 如果缓存文本可能以 XML 标签开头（可能属于 XML 格式的工具调用）
                        if stripped_buffer.startswith('<'):
                            possible_prefixes = ["<tool_call", "<function", "<thinking"]
                            # 判断当前缓存的文本是否为工具调用标签前缀的一部分
                            is_prefix = any(p.startswith(stripped_buffer) or stripped_buffer.startswith(p) for p in possible_prefixes)
                            # 如果不是合法的 XML 工具调用前缀，或者缓存内容过长（超过 200 字符），判定非工具调用，刷新缓存发送给前端
                            if not is_prefix or len(text_buffer) > 200:
                                yield f"0:{json.dumps(text_buffer, ensure_ascii=False)}\n"
                                full_content += text_buffer
                                text_buffer = ""
                        else:
                            # 如果显然不包含 XML 工具调用前缀，直接流式输出当前内容给前端
                            yield f"0:{json.dumps(text_buffer, ensure_ascii=False)}\n"
                            full_content += text_buffer
                            text_buffer = ""

            # 循环结束后，处理 buffer 中可能残留的文本内容
            if text_buffer and not tool_call_detected:
                # 双重检查：如果残留文本中包含 XML 格式的工具调用标识，则判定为触发了工具调用，不向用户直接输出
                if "<tool_call>" in text_buffer or "<function=" in text_buffer:
                    tool_call_detected = True
                else:
                    # 确认不是工具调用，将缓存中剩余的普通文本流式输出给前端
                    yield f"0:{json.dumps(text_buffer, ensure_ascii=False)}\n"
                    full_content += text_buffer
                    text_buffer = ""

            # print(f"[chat] response_chunk: {response_chunk}")

            # 将流式响应的所有 chunk 融合成最终的 AIMessage，用作后续兼容判断及消息持久化
            if response_chunk is None:
                response = AIMessage(content="")
            else:
                response = response_chunk
            
            # 兼容处理：有些大模型在没有完全走 API 的 tool_calls 时，会以 XML 文本形式返回 tool call
            if not response.tool_calls and response.content and ("<tool_call>" in response.content or "<function=" in response.content):
                print(f"[chat] LLM返回了XML格式的tool call，处理兼容")
                
                # 解析 XML 格式的 tool call
                import time
                func_pattern = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
                param_pattern = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
                
                parsed_calls = []
                for func_match in func_pattern.finditer(response.content):
                    func_name = func_match.group(1)
                    params_str = func_match.group(2)
                    
                    args = {}
                    for param_match in param_pattern.finditer(params_str):
                        param_name = param_match.group(1)
                        param_value = param_match.group(2).strip()
                        args[param_name] = param_value
                        
                    parsed_calls.append({
                        "name": func_name,
                        "args": args,
                        "id": f"xml_call_{int(time.time() * 1000)}",
                        "type": "tool_call"
                    })
                
                if parsed_calls:
                    print(f"[chat] Parsed XML tool calls: {parsed_calls}")
                    # 重建含有 tool_calls 的 AIMessage，清空 content 避免 XML 标签泄漏到回答中
                    response = AIMessage(
                        content="",
                        tool_calls=parsed_calls,
                        id=response.id,
                        response_metadata=response.response_metadata
                    )
            

            # 如果有调用工具，则调用工具完再stream最终回答
            if response.tool_calls:
                print(f"[chat] LLM返回示意工具调用: {response.tool_calls}")
                messages.append(response)
                
                tools_map = {
                    "get_current_weather": get_current_weather,
                    "get_exchange_rate": get_exchange_rate
                }
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    if tool_name in tools_map:
                        tool_obj = tools_map[tool_name]
                        tool_output = tool_obj.invoke(tool_call["args"])
                        print(f"[chat] Tool output from {tool_name}: {tool_output}")
                        # Construct and append ToolMessage
                        tool_message = ToolMessage(
                            content=str(tool_output),
                            name=tool_name,
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)

                # Stream the final response from LLM using the tool output
                async for chunk in chat_llm.astream(messages):
                    full_content += chunk.content
                    yield f"0:{json.dumps(chunk.content, ensure_ascii=False)}\n"

            # 如果没有调用工具，则说明在上面循环中已经把流输出完了，无需重复yield
            else:
                print("[chat] 没有调用工具，流式输出已在上面处理完成")


        except Exception as e:
            print(f"[chat] Chat chain generation failed: {e}")
            yield f"0:{json.dumps(f'生成失败: {str(e)}', ensure_ascii=False)}\n"

        print('[chat] 消息入库')
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
