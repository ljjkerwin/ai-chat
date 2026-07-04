import os  # 导入 os 模块，用于读取环境变量和处理文件路径

from dotenv import load_dotenv  # 导入 dotenv，用于从 .env 文件加载环境变量
from fastapi import FastAPI  # 导入 FastAPI 框架核心类
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件，用于处理跨域请求

from services.db import init_db  # 从本地 db 模块导入数据库初始化函数
from routers.chat import router as chat_router
from routers.knowledge import router as knowledge_router
from routers.sessions import router as sessions_router

# Load env from project root's .env.local
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.local"))  # 加载项目根目录下 .env.local 文件中的环境变量

app = FastAPI(title="MiMo RAG Backend", version="1.0.0")  # 创建 FastAPI 应用实例，设置标题和版本号

app.add_middleware(  # 为应用添加中间件
    CORSMiddleware,  # 使用 CORS 中间件
    allow_origins=["http://localhost:3200"],  # 允许的跨域来源：本地前端开发地址
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["X-Vercel-AI-Data-Stream"],  # 允许前端读取的响应头：标识 AI 数据流协议
)

# Include Routers
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(sessions_router)


@app.on_event("startup")  # 注册应用启动事件回调
async def on_startup() -> None:  # 定义启动时执行 of 异步函数
    init_db()  # 初始化数据库（建表等）
    from services.db import initialize_incremental_stats
    initialize_incremental_stats()  # 初始化增量 BM25 统计信息


@app.on_event("shutdown")
async def on_shutdown() -> None:
    from services.rag import close_http_client
    await close_http_client()


if __name__ == "__main__":  # 当该文件作为主程序直接运行时
    import uvicorn  # 导入 uvicorn 服务器
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # 启动 ASGI 服务，监听所有网卡的 8000 端口，开启热重载
