# services/code_context.py
from typing import Dict, Optional

"""
代码上下文管理器 (Code Context Manager)

职责:
这是一个简单的内存缓存，用作会话的“代码暂存区”。
它独立于对话历史，专门用于存储每个会话当前正在操作的完整代码。
代码在会话中只存储一份最新版本，通过 session_id 引用，绝不进入对话记忆，从而避免了成本浪费和上下文污染。

实现:
使用一个全局字典作为内存存储。在生产环境中，可以替换为 Redis 等外部缓存系统以支持多实例部署。
"""

# 使用一个简单的字典作为内存缓存。key: session_id, value: code_string
_code_context_cache: Dict[str, str] = {}

def update_code(session_id: str, code: str) -> None:
    """
    更新或存入一个会话的当前代码。

    Args:
        session_id (str): 唯一的会话 ID。
        code (str): 当前的完整代码字符串。
    """
    _code_context_cache[session_id] = code
    print(f"Code context updated for session: {session_id}")

def get_code(session_id: str) -> Optional[str]:
    """
    根据会话 ID 获取当前的代码。

    Args:
        session_id (str): 唯一的会话 ID。

    Returns:
        Optional[str]: 如果存在，返回代码字符串；否则返回 None。
    """
    return _code_context_cache.get(session_id)

def clear_code(session_id: str) -> None:
    """
    清除一个会话的代码上下文，例如在会话结束时。
    
    Args:
        session_id (str): 唯一的会话 ID。
    """
    if session_id in _code_context_cache:
        del _code_context_cache[session_id]
        print(f"Code context cleared for session: {session_id}")

