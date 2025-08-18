# api/versions.py
from fastapi import APIRouter, HTTPException, status, Depends
from langchain_community.vectorstores import Chroma

# --- 从核心服务和工具模块导入 ---
from services.services import get_summarizer, get_vector_store, CodeSummarizerService
from utility.schemas import AddVersionRequest, DeleteVersionRequest

router = APIRouter()

def _generate_doc_id(session_id: str, version_id: str) -> str:
    """生成用于向量数据库的唯一文档 ID。"""
    return f"{session_id}_{version_id}"

@router.post("/add_version_node", status_code=status.HTTP_200_OK) # ‼️【修改点】: 状态码从 202 改为 200
async def add_version_node(
    request: AddVersionRequest, 
    # ‼️【修改点】: 移除 BackgroundTasks
    summarizer: CodeSummarizerService = Depends(get_summarizer),
    vector_store: Chroma = Depends(get_vector_store)
):
    """
    接收一个新版本，为其生成摘要，存入数据库，然后同步返回生成的摘要。
    """
    print(f"同步处理版本: {request.session_id}_{request.version_id}")
    try:
        # 1. 生成 AI 摘要 (这是主要的耗时操作)
        ai_summary = await summarizer.summarize_code(request.code)
        
        # 2. 准备文档和元数据
        doc_id = _generate_doc_id(request.session_id, request.version_id)
        document_content = (
           
            f"代码内容总结: {ai_summary}\n\n"
            f"--- 源代码 ---\n{request.code}"
        )
        metadata = {
            "session_id": request.session_id, 
            "version_id": request.version_id, 
           
            "ai_summary": ai_summary
        }
        
        # 3. 将数据存入向量数据库
        vector_store.add_texts(ids=[doc_id], texts=[document_content], metadatas=[metadata])
        print(ai_summary)
        print(f"✅ 版本记忆已存入/更新，ID: {doc_id}")

        # 4. ‼️【修改点】: 在响应中返回生成的摘要
        return {
            "message": "Version processed and summary generated successfully.",
            "summary": ai_summary 
        }

    except Exception as e:
        print(f"❌ 同步处理版本失败，版本 ID {request.version_id}: {e}")
        # 在出错时抛出标准的 HTTP 异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process version '{request.version_id}': {str(e)}"
        )

# delete_version_node 函数保持不变...
@router.post("/delete_version", status_code=status.HTTP_200_OK)
async def delete_version_node(
    request: DeleteVersionRequest,
    vector_store: Chroma = Depends(get_vector_store)
):
    """从后端删除特定版本的记忆。"""
    try:
        doc_id = _generate_doc_id(request.session_id, request.version_id)
        vector_store.delete(ids=[doc_id])
        print(f"✅ 版本记忆已删除，ID: {doc_id}")
        return {"message": f"Version '{request.version_id}' deleted successfully."}
    except Exception as e:
        print(f"❌ 删除版本失败，ID {request.version_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete version '{request.version_id}': {e}"
        )