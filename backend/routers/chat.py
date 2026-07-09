from datetime import datetime
import asyncio
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
from langchain_core.utils.function_calling import convert_to_openai_tool
from ddgs import DDGS
from openai import AsyncOpenAI  # 导入原始异步 OpenAI 客户端用于捕获 reasoning_content

from services.rag import RAGRetriever, validate_safe_id
from services.db import create_session, add_session_message
from dependencies import get_current_tenant_id

# Load env from project root's .env.local (triggered reload for Qwen)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env.local"))  # 加载项目根目录下 .env.local 文件中的环境变量

# ── LangChain Global Models & Chains Configuration ─────────────────────────

# Query Rewrite Model & Chain
rewrite_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "Xiaomi/mimo-v2.5"),
    openai_api_base=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    openai_api_key=os.getenv("LLM_API_KEY", ""),
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
raw_openai_client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
)

chat_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "Xiaomi/mimo-v2.5"),
    openai_api_base=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    openai_api_key=os.getenv("LLM_API_KEY", ""),
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
        # 等待两秒钟再继续处理后面的逻辑
        import time
        time.sleep(2)
        
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



@tool
def web_search(query: str) -> str:
    """当用户询问当前发生的新闻、最新的事实、实时的数据或任何需要进行网页检索来获取最新信息的问题时，使用此工具进行联网搜索。"""
    print(f"[Tool] web_search called with query='{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            formatted_results = []
            for r in results:
                formatted_results.append(f"标题: {r.get('title')}\n链接: {r.get('href')}\n内容: {r.get('body')}\n")
            if not formatted_results:
                return "没有找到相关的搜索结果。"
            return "\n".join(formatted_results)
    except Exception as e:
        print(f"[Tool] Web search failed: {e}")
        return f"网页搜索失败，原因: {str(e)}"


@tool
async def web_fetch(url: str) -> str:
    """获取指定网页的文本内容。当需要读取或抓取某个特定网页、链接或 URL 的详细内容以回答用户问题时，使用此工具。"""
    print(f"[Tool] web_fetch called with url='{url}'")
    try:
        import httpx
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            
        if response.status_code != 200:
            return f"获取网页内容失败，状态码: {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除无用节点
        for element in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
            element.decompose()
            
        text = soup.get_text(separator="\n")
        
        # 清理多余的空白字符和空行
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        max_len = 6000
        if len(text) > max_len:
            return text[:max_len] + "\n\n[内容过长，已被截断...]"
        return text if text else "网页内容为空。"
    except Exception as e:
        print(f"[Tool] Web fetch failed: {e}")
        return f"获取网页内容失败，原因: {str(e)}"


chat_llm_with_tools = chat_llm.bind_tools([get_current_weather, get_exchange_rate, web_search, web_fetch])


# ── Pydantic models ───────────────────────────────────────────────────────

class Message(BaseModel):  # 定义单条聊天消息的数据模型
    role: str  # 消息角色（如 user、assistant、system）
    content: str  # 消息内容文本


class ChatRequest(BaseModel):  # 定义聊天接口请求体的数据模型
    messages: list[Message]  # 消息列表，包含完整的对话历史
    ragEnabled: bool = False  # 是否启用 RAG 检索增强，默认关闭
    sessionId: Optional[str] = None  # 会话 ID，可为空（表示新建会话）
    kbId: Optional[str] = None  # 选中的知识库 ID


# ── Helper Functions ──────────────────────────────────────────────────────

def extract_last_user_message(messages: list[Message]) -> tuple[Message, int]:
    """查找并返回最后一条用户消息及其在列表中的索引。如果未找到，抛出 HTTPException"""
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user":
            last_user_idx = idx
            break

    if last_user_idx == -1:
        raise HTTPException(status_code=400, detail="No user message found")

    return messages[last_user_idx], last_user_idx


def to_langchain_messages(messages: list[Message]) -> list:
    """将自定义消息列表转换为 LangChain 消息格式列表"""
    chat_history_messages = []
    for m in messages:
        if m.role == "user":
            chat_history_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            chat_history_messages.append(AIMessage(content=m.content))
    return chat_history_messages


async def prepare_rag_prompt(
    messages: list[Message],
    last_user_idx: int,
    rag_enabled: bool,
    kb_id: Optional[str],
    tenant_id: str
) -> tuple[str, list[dict]]:
    """准备 RAG 系统 Prompt 提示词与检索源。"""
    system_prompt = (
        "你是一个有帮助的 AI 助手，请用中文回答问题。\n"
        "【重要约束】不要主动向用户透露、介绍、列举或暗示你拥有任何特定的外部工具、插件、API 或功能。只有当用户明确要求你执行相关操作或查询时，你才静默调用工具来回答。在日常问候或普通闲聊中，请直接进行自然友好的回复，不要列出你的功能列表或进行自我宣传。"
    )
    sources: list[dict] = []

    if not rag_enabled:
        return system_prompt, sources

    kb_id = kb_id or "1"
    if not validate_safe_id(kb_id):
        raise HTTPException(status_code=400, detail="Invalid kb_id format")

    last_user = messages[last_user_idx]
    search_query = last_user.content
    history_messages = messages[:last_user_idx]

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

    return system_prompt, sources


def parse_xml_tool_calls(content: str) -> list[dict]:
    """解析大模型返回的 XML 格式工具调用指令"""
    import time
    func_pattern = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
    param_pattern = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

    parsed_calls = []
    for func_match in func_pattern.finditer(content):
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
    return parsed_calls


async def execute_tools(tool_calls: list[dict]) -> list[ToolMessage]:
    """并发执行工具调用，返回对应的 ToolMessage 列表"""
    import asyncio
    tools_map = {
        "get_current_weather": get_current_weather,
        "get_exchange_rate": get_exchange_rate
    }
    
    tasks = []
    matched_calls = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools_map:
            tool_obj = tools_map[tool_name]
            tasks.append(tool_obj.ainvoke(tool_call["args"]))
            matched_calls.append(tool_call)
            
    if not tasks:
        return []
        
    results = await asyncio.gather(*tasks)
    
    tool_messages = []
    for tool_call, tool_output in zip(matched_calls, results):
        print(f"[chat] Tool output from {tool_call['name']}: {tool_output}")
        tool_messages.append(ToolMessage(
            content=str(tool_output),
            name=tool_name,
            tool_call_id=tool_call["id"]
        ))
    return tool_messages


async def generate_chat_stream(
    system_prompt: str,
    chat_history_messages: list,
    last_user_content: str
) -> AsyncGenerator[tuple[str, str], None]:
    """执行 LLM 调用与工具分发流，向调用方 yield (event_type, payload) 格式的数据。
    event_type 可以是 'text' 或 'error'
    """
    model_name = os.getenv("LLM_MODEL", "Xiaomi/mimo-v2.5")
    
    # 格式化为 OpenAI API 所需 de messages
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history_messages:
        role = "user" if msg.type == "human" else "assistant"
        api_messages.append({"role": role, "content": msg.content})
    api_messages.append({"role": "user", "content": last_user_content})

    print(f"[chat] Invoking raw OpenAI API ({model_name}) with tools support...")
    
    # 检测是否为 mimo-v2.5 等推理模型以决定是否需要额外配置关闭思考
    is_reasoning_model = "mimo-v2.5" in model_name.lower() or "mimo" in model_name.lower()
    
    extra_params = {
        "temperature": 0.7
    }
    
    if is_reasoning_model:
        extra_params["extra_body"] = {
            "thinking": {
                "type": "disabled"
            }
        }

    # 准备 OpenAI 格式的工具列表
    tools_list = [get_current_weather, get_exchange_rate, web_search, web_fetch]
    openai_tools = [convert_to_openai_tool(t) for t in tools_list]

    tools_map = {
        "get_current_weather": get_current_weather,
        "get_exchange_rate": get_exchange_rate,
        "web_search": web_search,
        "web_fetch": web_fetch
    }

    try:
        current_turn = 0
        max_turns = 5

        # 循环调用大模型，最多支持 5 轮的 Tool Call / 思考交互（ReAct 循环）
        while current_turn < max_turns:
            current_turn += 1
            print(f"[chat] Turn {current_turn}: Invoking raw OpenAI API ({model_name})...")

            # 在达到最大轮数限制时，强制关闭工具调用与思考，避免产生死循环
            turn_params = extra_params.copy()
            if current_turn < max_turns:
                turn_tools = openai_tools
            else:
                turn_tools = None
                if is_reasoning_model:
                    turn_params["extra_body"] = {
                        "thinking": {
                            "type": "disabled"
                        }
                    }
            
            # 异步请求大模型的流式 chat completions 接口
            response = await raw_openai_client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
                tools=turn_tools,
                **turn_params
            )

            # 标识当前是否正处于模型的思维链/深度思考输出状态
            in_thinking = False
            # 累加与拼接标准 OpenAI 格式工具调用的字典，键为 tool_call 的 index，值为字典：{"id", "name", "arguments"}
            tool_calls_accumulator = {}
            # 累积记录模型本轮返回的完整正文内容，后续用于解析可能存在的 XML 格式工具调用
            accumulated_content = ""
            # 临时文本缓冲区，用来拦截并延迟流式输出可能是 XML 标签的文本前缀（如 "<tool_call"）
            text_buffer = ""
            # 标识本轮是否检测到了工具调用（包括标准工具调用与 XML 格式工具调用前缀），用于控制正文输出
            tool_call_detected = False

            # 流式处理响应数据包 (Chunk)
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # 1. 收集并拼接标准的 OpenAI tool_calls 信息
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    tool_call_detected = True
                    for tc in tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": tc.id,
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                        else:
                            if tc.id:
                                tool_calls_accumulator[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_accumulator[idx]["name"] = tc.function.name
                        
                        if tc.function and tc.function.arguments:
                            tool_calls_accumulator[idx]["arguments"] += tc.function.arguments

                # 2. 处理原生推理流字段 reasoning_content（例如 DeepSeek r1 的思维链内容）
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is None and hasattr(delta, "model_extra") and delta.model_extra:
                    reasoning = delta.model_extra.get("reasoning_content")

                if reasoning:
                    # 如果还未发送过开始思考的标签，则先流式输出 <think> 标识
                    if not in_thinking:
                        in_thinking = True
                        yield "text", "<think>"
                    yield "text", reasoning

                # 3. 处理最终回复文本 (缓存处理以应对大模型返回的 XML 格式 tool call，避免其泄露给用户)
                content = getattr(delta, "content", None)
                if content:
                    # 如果之前在思考，现在切换到正文，先输出闭合标签 </think>
                    if in_thinking:
                        in_thinking = False
                        yield "text", "</think>"

                    accumulated_content += content
                    text_buffer += content

                    # 如果尚未检测到标准的 tool_call，需提防本段文本是否是模型在吐 XML 标签格式的自定义工具调用
                    if not tool_call_detected:
                        stripped_buffer = text_buffer.strip()
                        # 如果缓存文本可能以 XML 标签开头（可能属于 XML 格式的工具调用前缀）
                        if stripped_buffer.startswith('<'):
                            possible_prefixes = ["<tool_call", "<function", "<thinking"]
                            is_prefix = any(p.startswith(stripped_buffer) or stripped_buffer.startswith(p) for p in possible_prefixes)
                            # 如果不是合法的 XML 工具调用前缀，或者缓存内容过长（>200字符），判定非工具调用，刷新缓存并向前端流式输出
                            if not is_prefix or len(text_buffer) > 200:
                                yield "text", text_buffer
                                text_buffer = ""
                        else:
                            # 正常文本内容，直接流式发送给前端
                            yield "text", text_buffer
                            text_buffer = ""

            # 流式结束时，如果还在思考模式中，补充闭合标签
            if in_thinking:
                yield "text", "</think>"

            # 处理可能残留的文本内容
            if text_buffer and not tool_call_detected:
                # 检查残留的缓冲区里是否包含 XML 工具调用的特定节点
                if "<tool_call>" in text_buffer or "<function=" in text_buffer:
                    tool_call_detected = True
                else:
                    yield "text", text_buffer
                    text_buffer = ""

            # 4. 解析与合并所有的工具调用 (标准 OpenAI 格式 + 兼容性 XML 格式)
            all_tool_calls = []

            # 4.1 提取与组装标准 OpenAI 格式的工具调用
            if tool_calls_accumulator:
                for idx, tc in sorted(tool_calls_accumulator.items()):
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except Exception:
                        args = {}
                    all_tool_calls.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": args,
                        "is_xml": False
                    })

            # 4.2 如果没有标准工具调用，尝试解析 XML 格式的工具调用（支持某些模型自定义输出的 XML 片段）
            if not all_tool_calls and ("<tool_call>" in accumulated_content or "<function=" in accumulated_content):
                tool_call_detected = True
                xml_calls = parse_xml_tool_calls(accumulated_content)
                for xc in xml_calls:
                    all_tool_calls.append({
                        "id": xc["id"],
                        "name": xc["name"],
                        "args": xc["args"],
                        "is_xml": True
                    })

            # 5. 如果本次大模型输出没有触发任何工具调用，说明已经生成了最终答案，退出 ReAct 循环
            if not all_tool_calls:
                break

            # 格式化 assistant 的 tool_calls，将其转为 OpenAI 标准的格式对象
            openai_tool_calls = []
            for tc in all_tool_calls:
                openai_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False)
                    }
                })

            assistant_msg = {
                "role": "assistant",
                "content": accumulated_content
            }
            if tool_calls_accumulator:
                # 标准 OpenAI tool calls 必须在消息体里带上 tool_calls 结构
                assistant_msg["tool_calls"] = openai_tool_calls
                # 且标准调用下，assistant content 最好为空或前置思考内容（移除 XML 字段）
                assistant_msg["content"] = ""

            # 将 assistant 响应（包含 tool_calls）追加到当前上下文消息中
            api_messages.append(assistant_msg)

            # 筛选出有对应后端映射实现的工具，准备并行执行
            tasks = []
            matched_calls = []
            for tc in all_tool_calls:
                name = tc["name"]
                if name in tools_map:
                    tool_func = tools_map[name]
                    tasks.append(tool_func.ainvoke(tc["args"]))
                    matched_calls.append(tc)

            # 并行执行匹配到的后端工具
            if tasks:
                print(f"[chat] Executing tools: {[tc['name'] for tc in matched_calls]}")
                results = await asyncio.gather(*tasks)
                # 将工具执行结果作为 role="tool" 的消息依次追加入上下文
                for tc, output in zip(matched_calls, results):
                    api_messages.append({
                        "role": "tool",
                        "name": tc["name"],
                        "tool_call_id": tc["id"],
                        "content": str(output)
                    })

                # 查看本轮所有消息
                print("\n\n".join(str(i) for i in api_messages))

            else:
                # 如果有工具调用但没有匹配到任何后端工具，为了避免死循环，直接返回内容并退出
                print(f"[chat] Warning: No matched tool functions for calls: {[tc['name'] for tc in all_tool_calls]}")
                if accumulated_content:
                    yield "text", accumulated_content

                # 查看本轮所有消息
                print("\n\n".join(str(i) for i in api_messages))
                break

    except Exception as e:
        print(f"[chat] Raw API generation failed: {e}")
        yield "error", f"生成失败: {str(e)}"


# ── Router Setup ─────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/api/chat")  # 注册 POST /api/chat 路由
async def chat(request: ChatRequest, tenant_id: str = Depends(get_current_tenant_id)) -> StreamingResponse:  # 聊天接口处理函数，返回流式响应
    print(f"[chat] tenant_id={tenant_id} ragEnabled={request.ragEnabled} kbId={request.kbId} messages={len(request.messages)}")  # 打印调试日志

    # 1. 查找并验证最后一条用户消息
    last_user, last_user_idx = extract_last_user_message(request.messages)

    # 2. 会话管理 (Session): create if needed, save user message
    session_id = request.sessionId
    if not session_id:
        session_id = await asyncio.to_thread(create_session, last_user.content[:30])
    await asyncio.to_thread(add_session_message, session_id, "user", last_user.content)

    # 3. 处理 RAG 预处理与系统 Prompt 生成，得到带知识库的系统提示词
    system_prompt, sources = await prepare_rag_prompt(
        messages=request.messages,
        last_user_idx=last_user_idx,
        rag_enabled=request.ragEnabled,
        kb_id=request.kbId,
        tenant_id=tenant_id
    )

    # 4. 转换历史对话消息为 LangChain 格式
    chat_history_messages = to_langchain_messages(request.messages[:last_user_idx])

    # 5. 定义流式生成器
    async def generate() -> AsyncGenerator[str, None]:
        if session_id:
            yield f'2:[{json.dumps({"type": "session", "sessionId": session_id})}]\n'  # 推送会话信息给前端

        full_content = ""  # 用于累积完整的助手回复内容

        try:
            # 6. 调用流式生成器，接收并推送流数据
            async for event_type, value in generate_chat_stream(
                system_prompt=system_prompt,
                chat_history_messages=chat_history_messages,
                last_user_content=last_user.content
            ):
                if event_type == "text":
                    yield f"0:{json.dumps(value, ensure_ascii=False)}\n"
                    full_content += value
                elif event_type == "error":
                    yield f"0:{json.dumps(value, ensure_ascii=False)}\n"

        except Exception as e:
            print(f"[chat] Chat chain generation failed: {e}")
            yield f"0:{json.dumps(f'生成失败: {str(e)}', ensure_ascii=False)}\n"

        print('[chat] 消息入库')
        if session_id and full_content:
            await asyncio.to_thread(add_session_message, session_id, "assistant", full_content)  # 将助手回复写入会话

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
