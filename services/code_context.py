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



'''
{
    "id": "p5_adv_012",
    "tag": "袅袅青烟",
    "image": "/image/12.png",
    "code": "let particles = [];\n\nfunction setup() {\n  createCanvas(800, 1200);\n}\n\nfunction draw() {\n  background(0);\n  for (let i = 0; i < 5; i++) {\n    let p = new Particle();\n    particles.push(p);\n  }\n  for (let i = particles.length - 1; i >= 0; i--) {\n    particles[i].update();\n    particles[i].show();\n    if (particles[i].finished()) {\n      particles.splice(i, 1);\n    }\n  }\n}\n\nclass Particle {\n  constructor() {\n    this.x = width / 2;\n    this.y = height - 50;\n    this.vx = random(-1, 1);\n    this.vy = random(-5, -1);\n    this.alpha = 255;\n    this.size = random(16, 32);\n  }\n\n  finished() {\n    return this.alpha < 0;\n  }\n\n  update() {\n    this.x += this.vx;\n    this.y += this.vy;\n    this.alpha -= 5;\n  }\n\n  show() {\n    noStroke();\n    fill(255, this.alpha);\n    ellipse(this.x, this.y, this.size);\n  }\n}"
  },
  {
    "id": "p5_adv_013",
    "tag": "粒子流场",
    "image": "/image/13.png",
    "code": "let particles = [];\nconst num = 1000;\nconst noiseScale = 0.01;\n\nfunction setup() {\n  createCanvas(800, 1200);\n  for (let i = 0; i < num; i++) {\n    particles.push(createVector(random(width), random(height)));\n  }\n  stroke(255, 50);\n  strokeWeight(2);\n}\n\nfunction draw() {\n  background(0, 10);\n  for (let i = 0; i < num; i++) {\n    let p = particles[i];\n    point(p.x, p.y);\n    let n = noise(p.x * noiseScale, p.y * noiseScale);\n    let a = TAU * n;\n    p.x += cos(a);\n    p.y += sin(a);\n    if (!onScreen(p)) {\n      p.x = random(width);\n      p.y = random(height);\n    }\n  }\n}\n\nfunction onScreen(v) {\n  return v.x >= 0 && v.x <= width && v.y >= 0 && v.y <= height;\n}"
  },
'''