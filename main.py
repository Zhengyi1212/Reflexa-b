# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- 从新的 services.py 文件中导入初始化函数 ---
from services.services import initialize_services
# --- 从新的 api 模块导入主路由 ---
from routes.routes import api_router
# --- 假设的导入路径 ---
from utility.config import settings

# --- 应用的生命周期管理 (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动时，调用独立的初始化函数来设置所有服务。
    """
    print("--- 应用启动，开始初始化服务 ---")
    initialize_services()
    yield
    # 这里可以放置应用关闭时需要执行的清理代码
    print("--- 应用正在关闭 ---")


# --- FastAPI 应用实例配置 ---
app = FastAPI(
    title="Design Loop AI Backend",
    version="1.0.0",
    lifespan=lifespan  # 附加生命周期事件
)

# --- 中间件配置 ---
# 建议明确指定允许的源，而不是使用 "*"
origins = [
    "http://localhost",
    "http://localhost:5173", # 假设这是你的前端地址
    # "http://your-production-domain.com",
]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- 注册 API 路由 ---
# 将所有 API 路由包含进来，可以加一个统一的前缀，如 /api/v1
app.include_router(api_router)

@app.get("/", summary="Health Check", tags=["System"])
def read_root():
    """根路径提供一个简单的健康检查端点。"""
    return {"status": "ok", "message": "Welcome to the Design Loop AI Backend!"}

# --- 启动命令 ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=settings.APP_PORT, 
        reload=True # 在开发环境中非常有用
    )
