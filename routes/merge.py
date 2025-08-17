# api/merge.py
from fastapi import APIRouter, HTTPException, status
from utility.schemas import MergeRequest
from typing import Dict

# --- LangChain 和自定义模块导入 ---
from langchain_openai import AzureChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utility.config import settings

# --- Pydantic 模型定义 ---

# --- 初始化 Router ---
router = APIRouter()

# --- LLM 和 Prompt 设置 ---
# 复用现有的 llm 实例，或根据需要创建一个新的
llm = AzureChatOpenAI(
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_deployment=settings.AZURE_OPENAI_MODEL_NAME,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    temperature=0.2 # 对于代码合并，使用较低的温度以获得更可预测的结果
)

SYSTEM_PROMPT = """

# 你是一位资深的创意技术顾问与p5.js专家，擅长将不同的代码逻辑进行解构与重组，以实现富有创意的功能融合。你只能用中文回答。
# 你的核心任务是：接收两段独立的p5.js代码（版本A和版本B），并根据用户的“融合指令”，将它们合二为一，创造出一个和谐、统一且功能完整的全新作品。这不仅是代码的合并，更是两种创意的策略性结合。

#思维链条 (Chain of Thought):

## 逻辑解构: 首先，深入分析两段代码。它们各自的核心功能是什么？是视觉表现（色彩、构图）、运动逻辑（物理模拟、噪声场），还是交互模式？
## 确立融合主体 (Anchor): 根据用户的指令和两段代码的特性，判断哪一个版本更适合作为这次融合的“基石”（Anchor）。这个选择将决定最终代码的核心结构。
## 制定融合策略: 将非主体代码的“核心功能”视为一种独特的“模块”或“特性”。思考如何最巧妙地将这个“模块”集成到主体的代码结构中，是添加新的变量、修改绘图函数，还是引入新的交互事件？
## 执行优雅融合: 严格按照你的策略，以最清晰、最无缝的方式进行代码合并。解决技术冲突，同时保证逻辑上的和谐，确保最终代码既稳定运行，又实现了预期效果。
## 撰写融合阐述 (Rationale): 在完成代码后，撰写一段清晰、有启发性的融合说明（50字）。用专业而富有创意的口吻，讲述这次融合的思路，先赞美两种功能的巧妙结合，并点明你是如何将它们的精髓结合在一起的。

输出格式:
你必须严格按照有效的JSON格式进行响应。JSON对象必须包含两个键：
`code`: 内容是经过你融合后的、完整的p5.js代码。
`rationale`: 一个markdown格式，作为你的融合阐述。只能用`###`和`-`
<Few-shot Example>
用户输入会包含如下信息:
代码版本A:``` function setup() {{ createCanvas(400, 400); }} function draw() {{ background(220); ellipse(200, 200, 50, 50); }}```
代码版本B:``` let x, y; function setup() {{ createCanvas(400, 400); x = 200; y = 200; }} function draw() {{ x = mouseX; y = mouseY; }}```
融合指令: "用B的交互性来控制A的圆圈"
你的理想输出应为:
{{
  "code": "",
  "rationale": "- ✨你有顶级的洞察力和艺术编程细胞！此次融合巧妙地将B中的动态交互与A的静态圆形结合，通过鼠标追踪逻辑，为A的圆形赋予了动态生命力。
               - 我将B的动态鼠标位置应用到A的圆形绘制中，使得原本静止的圆形随着鼠标移动，呈现出流动感和互动性。
               - 可以进一步加入物理模拟效果，或引入更复杂的用户输入，完全可以提升作品的互动深度和视觉吸引力。🚀"
}}
<Few-shot Example>
# 回复语言逻辑清晰，具有交互性和艺术洞悉，合理使用emoji。
"""

# --- User Prompt: 提供了所有需要合并的信息 ---
USER_PROMPT_TEMPLATE = """
这是需要合并的两个代码版本。

**代码版本1 (ID: {version_id_1})**
*描述*: {description_1}
*代码*:
```javascript
{code_1}
```

**代码版本2 (ID: {version_id_2})**
*描述*: {description_2}
*代码*:
```javascript
{code_2}
```

**合并指令：**
"{instruction}"

请根据我的指令合并这两个版本，并以要求的JSON格式提供合并后的代码和你的理由。
"""
# --- LangChain Chain 定义 ---
merge_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE)
])

# 使用 .with_structured_output 来确保返回的是我们期望的 JSON 结构
# 注意：这需要较新版本的 langchain-openai
# 如果不可用，则使用 JsonOutputParser
chain = merge_prompt | llm | JsonOutputParser()


@router.post("/merge", response_model=Dict[str, str])
async def merge_code_versions(request: MergeRequest):
    """
    接收两个代码版本和一条指令，使用 LLM 进行智能合并。
    """
    print(f"Received merge request for session: {request.session_id}")
    try:
        chain_input = {
            "version_id_1": request.version_id_1,
            "code_1": request.code_1,
            "description_1": request.description_1,
            "version_id_2": request.version_id_2,
            "code_2": request.code_2,
            "description_2": request.description_2,
            "instruction": request.instruction,
        }

        # 调用 LLM chain
        response = await chain.ainvoke(chain_input)
        
        # 验证返回结果
        if "code" not in response or "rationale" not in response:
            raise HTTPException(status_code=500, detail="Invalid response format from LLM.")

        print("Successfully merged code.")
        print(response)
        return response

    except Exception as e:
        print(f"❌ Error during merge process: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during the merge process: {str(e)}"
        )
