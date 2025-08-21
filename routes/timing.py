# api/timing.py
import json
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Pydantic 数据模型定义 ---

class TimeSegment(BaseModel):
    """定义一个时间段的模型"""
    start: int
    end: int
    duration: int

class ActionCounts(BaseModel):
    """定义操作计数模型"""
    delete: int
    modify: int
    merge: int
    duplicate: int

class UsageData(BaseModel):
    """定义用户行为埋点数据的模型"""
    totalVersions: int = Field(..., alias="totalVersions")
    versionConversations: Dict[str, List[Dict[str, Any]]] = Field(..., alias="versionConversations")
    previewClicks: int = Field(..., alias="previewClicks")
    actionCounts: ActionCounts = Field(..., alias="actionCounts")
    # [!code ++] 新增：用于存储每个版本代码的字段
    versionCodes: Dict[str, str] = Field(..., alias="versionCodes", description="一个以节点ID为键，代码字符串为值的字典")


class SessionDataRequest(BaseModel):
    """
    定义接收完整会话数据的请求体模型
    """
    session_id: str = Field(..., description="当前会话的唯一ID")
    user_id: str = Field(..., description="用户的ID")
    task: str = Field(..., description="当前执行的任务 (e.g., TaskA, TaskB)")
    
    timingData: Optional[Dict[str, List[TimeSegment]]] = Field(None, alias="timingData", description="各区域使用时间段")
    usageData: Optional[UsageData] = Field(None, alias="usageData", description="用户行为埋点数据")


# --- FastAPI 路由 ---

router = APIRouter()

# 日志文件路径
LOG_FILE_PATH = "session_logs.jsonl" 

@router.post("/timing", status_code=status.HTTP_200_OK)
async def save_session_data(request: SessionDataRequest):
    """
    接收并记录前端发送的完整会话数据（计时、埋点、代码快照）。
    """
    print(f"接收到来自会话 {request.session_id} (用户: {request.user_id}) 的完整会话数据。")
    
    try:
        # 将 Pydantic 模型转换为字典，以便序列化
        # 使用 by_alias=True 来确保字段名与前端发送的一致
        log_entry = request.dict(by_alias=True, exclude_none=True)
        
        # 将日志条目以 JSON Lines 格式追加到文件中
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        print(f"✅ 完整会话数据已成功记录到 {LOG_FILE_PATH}")

        return {
            "message": "Session data received and logged successfully."
        }

    except Exception as e:
        print(f"❌ 记录会话数据时发生错误: {e}")
        # 抛出标准的 HTTP 异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log session data: {str(e)}"
        )

