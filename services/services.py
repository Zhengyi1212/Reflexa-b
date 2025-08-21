# services/services.py
import chromadb
from typing import Dict, List
import os

# --- 核心依赖：从 LangChain 和项目配置导入 ---
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema.messages import SystemMessage, HumanMessage

# --- 假设的导入路径，请根据你的项目结构进行调整 ---
from utility.config import settings
# --- ‼️【修改】导入新的 InspirationService，移除旧的 RAGService ---
from .inspiration_service import InspirationService

SUMMARY_PROPMT = """
# 你是一个P5.js代码分析专家。

# 你的任务是分析给定的p5.js代码片段，并生成一个简洁的、40字以内的摘要。
# 重点描述代码的功能和**视觉表现**。

# 任务要求：
- **只关注代码的视觉表现，用最直观地方法描述code，避免解释具体语法或细节**。
- **绝对不要提到代码语言或代码层级的细节**。
- **摘要必须清晰、简洁，仅包含功能描述**。

# 返回要求：
- **摘要内容只能使用中文，严禁使用任何其他语言！**
- **摘要长度严格限制在40字以内！**
"""
# --- 全局变量定义，用于持有初始化后的服务实例 ---
# 这些变量将由 initialize_services 函数在应用启动时填充
vector_store: Chroma = None
summarizer_service: "CodeSummarizerService" = None
# --- ‼️【修改】用 inspiration_service 替换 rag_service ---
inspiration_service: InspirationService = None

# --- 服务类定义 ---
class CodeSummarizerService:
    """封装与代码摘要相关的 LLM 调用。(这个类保持不变)"""
    def __init__(self):
        """初始化 Azure Chat LLM 客户端。"""
        self._llm = AzureChatOpenAI(
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_MODEL_NAME,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=0.7
        )
        print("✅ Azure Chat LLM for Summarizer 已初始化。")

    async def summarize_code(self, code: str) -> str:
        """调用 LLM 为提供的代码生成简洁的摘要。"""
        messages = [
            SystemMessage(
                content=SUMMARY_PROPMT
            ),
            HumanMessage(
                content=f"请为以下代码生成摘要:\n\n```javascript\n{code}\n```"
            )
        ]
        try:
            response = await self._llm.ainvoke(messages)
            summary = response.content
            print("✅ 成功生成代码摘要。")
            return summary
        except Exception as e:
            print(f"❌ 在代码摘要过程中发生错误: {e}")
            return "生成 AI 摘要失败。"

# --- 集中初始化函数 ---
def initialize_services():
    """
    执行所有昂贵的初始化操作，并将实例赋给本模块的全局变量。
    这个函数将在 main.py 的 lifespan 中被调用一次。
    """
    # --- ‼️【修改】将 inspiration_service 加入 global ---
    global vector_store, summarizer_service, inspiration_service
    
    print("--- 核心服务初始化开始 ---")
    
    # 1. 初始化 Embedding 模型 (供 ChromaDB 使用)
    try:
        embeddings_model = AzureOpenAIEmbeddings(
            model=settings.AZURE_OPENAI_EMBEDDING_MODEL,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        print("✅ Azure OpenAI Embedding 模型已初始化。")
    except Exception as e:
        print(f"❌ 初始化 Embedding 模型失败: {e}")
        raise e

    # 2. 连接到 ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        collection_name = "version_graph_memory"
        vector_store = Chroma(
            client=chroma_client, 
            collection_name=collection_name,
            embedding_function=embeddings_model
        )
        print(f"✅ ChromaDB 向量数据库已连接。正在使用集合: '{collection_name}'")
    except Exception as e:
        print(f"❌ 连接到 ChromaDB 时出错: {e}")
        raise e
        
    # 3. ‼️【修改】初始化新的 InspirationService
    try:
        inspiration_service = InspirationService()
        # 假设 data/ 文件夹在项目根目录
        data_path = os.path.join(os.path.dirname(__file__),  "data", "p5_examples.json")
        inspiration_service.load_examples(filepath=data_path)
        print("✅ Inspiration 服务已初始化并加载p5.js灵感库。")
    except Exception as e:
        print(f"❌ 初始化 InspirationService 时出错: {e}")
        raise e

    # 4. 初始化代码摘要服务
    try:
        summarizer_service = CodeSummarizerService()
    except Exception as e:
        print(f"❌ 初始化 CodeSummarizerService 时出错: {e}")
        raise e
        
    print("--- 核心服务初始化完成 ---")

# --- 依赖注入函数 (供路由使用) ---

def get_vector_store() -> Chroma:
    """一个 FastAPI 的 Depends 函数，用于向路由提供 vector_store 实例。"""
    if vector_store is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Vector Store 服务未初始化，请检查服务器日志。")
    return vector_store

def get_summarizer() -> CodeSummarizerService:
    """一个 FastAPI 的 Depends 函数，用于向路由提供 summarizer 实例。"""
    if summarizer_service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Summarizer 服务未初始化，请检查服务器日志。")
    return summarizer_service

# --- ‼️【修改】为新的 InspirationService 创建依赖注入函数 ---
def get_inspiration_service() -> InspirationService:
    """一个 FastAPI 的 Depends 函数，用于向路由提供 inspiration_service 实例。"""
    if inspiration_service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Inspiration 服务未初始化，请检查服务器日志。")
    return inspiration_service
