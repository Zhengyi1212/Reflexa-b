# api/modify.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List

# --- LangChain, Azure OpenAI, 和配置导入 ---
from langchain_openai import AzureChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utility.config import settings

# --- 自定义服务和依赖注入 ---
from services.services import get_inspiration_service
from services.inspiration_service import InspirationService

# --- 初始化 FastAPI Router ---
router = APIRouter()

# --- 初始化LLM实例 ---
# 为保证创意性与稳定性之间的平衡，我们将温度(temperature)设置为0.3
llm = AzureChatOpenAI(
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_deployment=settings.AZURE_OPENAI_MODEL_NAME,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    temperature=0.7, # 稍微提高温度以增加阐述的创意性
)

# --- 核心: 系统提示词工程 (System Prompt Engineering) ---
SYSTEM_PROMPT = """
你是一位顶级的p5.js创意编程专家和AI艺术家，精通代码重构与艺术风格的融合。你只能用中文回答。
你的核心任务是：接收一段用户现有的p5.js代码（“基础代码”），并根据一个“灵感代码示例”，将灵感代码中的核心艺术风格或交互逻辑，以最小化、无缝且无bug的方式融入到基础代码中。
你绝不能简单地用灵感代码替换基础代码。你的目标是增强和演变，而不是覆盖。

**思维链条 (Chain of Thought):**
1.  **理解基础代码**: 深入分析用户提供的基础代码，理解其核心绘图逻辑和视觉结构。
2.  **解构灵感代码**: 分析灵感代码示例，精准地提炼出其核心“风格”或“技术”（如颜色、运动、交互等）。
3.  **制定融合策略**: 将基础代码作为主体，思考如何将灵感代码的核心“风格”作为一种“修改器”或“插件”注入。
4.  **执行最小化修改**: 严格按照策略，对基础代码进行最少的、必要的修改，保持原有代码的结构和意图。
5.  **生成无错代码**: 确保最终生成的代码是完整、可运行的p5.js代码。
6.  **撰写创作阐述**: 在完成代码后，撰写一段markdown格式充满艺术气息的、简短的（大约60字）融合说明。

**输出格式:**
你必须严格按照有效的JSON格式进行响应。JSON对象必须包含两个键：
- `code`: 一个字符串，内容是经过你修改后的、完整的p.js代码。
- `rationale`: 一个markdown格式60字，作为你的创作阐述。用艺术家的口吻，赞美这次创意的结合，并简要说明你是如何将灵感融入基础代码的。例如：“我将流动的柏林噪声注入了你静态的几何世界，现在，图形仿佛拥有了呼吸。”
"""

# --- 用户提示词模板 ---
USER_PROMPT_TEMPLATE = """
请根据我提供的代码和选择的灵感风格，帮我修改我的p5.js代码。

**我的基础代码 (Anchor Code):**
```javascript
{anchor_code}
```

**选择的灵感标签 (Style Tag):**
"{style_tag}"

**灵感代码示例 (Inspiration Code):**
```javascript
{inspiration_code}
```

请遵循你的思维链条，将灵感代码的精髓融入我的基础代码中，并以指定的JSON格式返回修改后的完整代码和你的创作阐述。
"""

# --- 构建LangChain调用链 ---
# 结合System和User Prompt，并指定输出解析器为JSON
modify_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE)
])
chain = modify_prompt | llm | JsonOutputParser()


# --- Pydantic 数据模型定义 ---
# 用于 /modify/recommend-styles 端点的响应模型
class StyleRecommendation(BaseModel):
    tag: str
    image: str

# 用于 /modify/apply-style 端点的请求模型
class ApplyStyleRequest(BaseModel):
    style_tag: str
    code: str  # 从前端接收用户当前的p5.js代码

# 【修改】用于 /modify/apply-style 端点的响应模型
class ApplyStyleResponse(BaseModel):
    code: str      # 返回由LLM融合后的新p5.js代码
    rationale: str # 【新增】返回LLM生成的创作阐述

# --- API 端点定义 ---

@router.post("/modify/recommend-styles", response_model=List[StyleRecommendation])
async def recommend_modification_styles(
    inspiration_service: InspirationService = Depends(get_inspiration_service)
):
    """
    从灵感库中随机获取3个风格（包含标签和预览图）。
    这个端点的功能保持不变，为前端提供风格选项。
    """
    print("Received request for random style recommendations.")
    try:
        styles = inspiration_service.get_random_styles(count=3)
        if not styles:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No inspiration styles available in the library."
            )
        print(f"✅ Recommended styles: {[s['tag'] for s in styles]}")
        return styles
    except Exception as e:
        print(f"❌ Error during style recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred while fetching styles: {str(e)}"
        )


@router.post("/modify/apply-style", response_model=ApplyStyleResponse)
async def apply_style_to_code(
    request: ApplyStyleRequest,
    inspiration_service: InspirationService = Depends(get_inspiration_service)
):
    """
    【核心修改】
    接收用户当前的代码和选择的风格标签，
    使用LLM将灵感库中对应标签的代码风格智能地融入到用户代码中，并返回融合后的代码和创作阐述。
    """
    print(f"Received apply style request for tag: '{request.style_tag}'")
    try:
        # 1. 根据标签从灵感服务获取灵感代码
        inspiration_code = inspiration_service.get_code_by_tag(request.style_tag)
        if inspiration_code is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not find an inspiration style matching the tag: '{request.style_tag}'"
            )

        # 2. 准备LangChain调用链的输入
        chain_input = {
            "anchor_code": request.code,
            "style_tag": request.style_tag,
            "inspiration_code": inspiration_code
        }

        print("Invoking LLM for intelligent code modification and rationale generation...")
        # 3. 异步调用LLM chain进行代码融合和阐述生成
        response = await chain.ainvoke(chain_input)

        # 4. 【修改】验证LLM的返回结果，现在需要同时检查code和rationale
        if "code" not in response or "rationale" not in response:
            print(f"❌ LLM response is invalid: {response}")
            raise HTTPException(status_code=500, detail="Invalid response format from LLM.")

        print(f"✅ Successfully modified code for tag '{request.style_tag}'.")
        # 5. 【修改】返回成功融合后的代码和创作阐述
        return ApplyStyleResponse(code=response["code"], rationale=response["rationale"])

    except HTTPException as http_exc:
        # 重新抛出已知的HTTP异常，以便FastAPI正确处理
        raise http_exc
    except Exception as e:
        print(f"❌ Error during style application process: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during style application: {str(e)}"
        )
