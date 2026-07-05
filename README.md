# AI Chat RAG

基于 RAG（检索增强生成）的智能问答应用，支持上传私有知识库文档，结合向量检索与大语言模型实现精准问答。

## 功能特性

- **流式对话**：基于 Vercel AI SDK，实时流式输出 AI 回复
- **RAG 检索**：上传文档后，AI 可优先基于知识库内容回答，标注引用来源
- **向量 + 关键词双路检索**：优先使用 Embedding 向量相似度，无 Embedding 时自动降级为 BM25 关键词匹配
- **会话管理**：自动创建并保存对话历史，支持多会话切换与删除
- **知识库管理**：上传、查看、删除文档，数据持久化于本地 SQLite

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Next.js 15 (App Router) · TypeScript · Tailwind CSS |
| AI 集成 | Vercel AI SDK v4 (`useChat`) |
| 后端 | Python 3.11 · FastAPI · uvicorn |
| 向量计算 | numpy (cosine similarity) |
| 数据库 | SQLite（stdlib `sqlite3`） |
| LLM | MiMo-7B-RL（或任意 OpenAI 兼容模型） |
| Embedding | BAAI/bge-m3（或 Gemini Embedding API） |

## 快速开始

### 环境要求

- Node.js 18+
- Python 3.11+

### 1. 安装前端依赖

```bash
npm install
```

### 2. 安装后端依赖

```bash
cd backend
pip3 install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env.local`：

```env
# 聊天模型（OpenAI 兼容接口）
MIMO_BASE_URL=https://api.siliconflow.cn/v1
MIMO_API_KEY=your_api_key
MIMO_MODEL=Xiaomi/MiMo-7B-RL

# Embedding 模型（可选，不填则降级为关键词检索）
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL=BAAI/bge-m3
```

> `EMBEDDING_*` 变量不填时，RAG 检索自动使用关键词匹配作为替代。

### 4. 启动服务

**后端**（端口 8000）：

```bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

**前端**（端口 3200）：

```bash
npm run dev
```

打开 [http://localhost:3200](http://localhost:3200) 即可使用。

## 项目结构

```
.
├── app/                    # Next.js App Router 页面
│   └── (main)/
│       ├── chat/           # 聊天页面
│       └── knowledge/      # 知识库管理页面
├── components/
│   ├── chat/               # 聊天界面组件
│   ├── knowledge/          # 知识库组件
│   └── layout/             # 侧边栏等布局组件
├── contexts/
│   ├── RAGContext.tsx       # RAG 开关状态（localStorage）
│   └── SessionContext.tsx   # 当前会话状态
├── backend/
│   ├── main.py             # FastAPI 应用，API 路由
│   ├── rag.py              # 文本分块、Embedding、检索逻辑
│   ├── db.py               # SQLite CRUD
│   └── requirements.txt
├── data/
│   └── rag.db              # SQLite 数据库（自动创建）
└── next.config.ts          # /api/* 反向代理到后端
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 流式对话（支持 RAG） |
| GET | `/api/knowledge` | 获取文档列表及统计 |
| POST | `/api/knowledge` | 上传新文档 |
| DELETE | `/api/knowledge/:id` | 删除文档 |
| GET | `/api/sessions` | 获取会话列表 |
| GET | `/api/sessions/:id/messages` | 获取会话消息 |
| DELETE | `/api/sessions/:id` | 删除会话 |

## RAG 工作原理

1. **文档入库**：文本按 600 字符分块（120 字符重叠），每块异步生成 Embedding 并存入 SQLite
2. **检索**：用户提问时，对问题生成 Embedding，与所有 chunk 计算余弦相似度，取 top-4（阈值 0.05）
3. **降级**：Embedding API 不可用时，自动使用 BM25 风格的关键词匹配（CJK 字符 + 二元组）
4. **生成**：将检索到的上下文注入 system prompt，模型优先基于知识库内容作答并标注来源
