# utility/deep_chat.py
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import Dict

# ‼️ 修改点: 导入新的、分模式的Vague Prompt
from .prompt import (
    TRANSITION_SYSTEM_PROMPT,
    VAGUE_EXPLAINABLE_PROMPT,
    VAGUE_EXPLORATIVE_PROMPT,
    VAGUE_TRANSFORMATIVE_PROMPT,
    DEEP_EXPLAINABLE_PROMPT,
    DEEP_TRANSFORMATIVE_PROMPT,
    DEEP_EXPLORATIVE_PROMPT
    

)

DEEP_REFLECTION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "explainable": {
        "动机说明": "💬 你提到这个 {{topic}} 时，背后想表达的核心感受或体验是什么？这个想法与你以往的创作、经历或目标有什么联系？",
        "视觉目标澄清": "💬 你实现 {{topic}} 这个功能时，想要呈现的视觉体验或交互感受是什么？它与整个项目的创意目标之间有何关联？",
        "细节决策说明": "💬 你做出这个 {{topic}} 这个细节调整时，背后的设计动机或想营造的感受是什么？这个细节是否强化了你的表达？"
    },
    "explorative": {
        "概念联系探索": "💬 你的灵感 {{topic}} 中有没有哪些元素可以结合起来，产生新的想象或叙事线索？",
        "模块体验关系": "💬 你能否思考一下当前的 {{topic}} 这几个功能模块，它们之间是否能更协调地服务于整体的视觉叙事或交互体验？",
        "视觉情感一致性": "💬 在你的作品中，{{topic}} 这些视觉元素之间是否保持了统一的风格和情绪？有没有可以更好融合它们的方式？"
    },
    "transformative": {
        "创意方向转变": "💬 如果从另一种角度（例如 {{topic}}）讲述这个故事，比如换一种情绪基调，会发生什么变化？",
        "功能方法重思": "💬 目前 {{topic}} 的功能效果是否与你想象中的体验存在偏差？如果是，有没有其他方式可以更贴切地表达你的意图？",
        "视觉风格调整": "💬 现在 {{topic}} 的整体风格是否与你想传达的核心感受完全契合？如果偏离了，你愿意在哪些部分做出调整以重建风格一致性？"
    }
}
# --- 模板库 (深度反思，保持不变) ---
DEEP_REFLECTION: Dict[str, Dict[str, str]] = {
    "explainable": {
        "动机说明": "💬 你提到{topic}时，背后想表达的核心感受或体验是什么？这个想法与你以往的创作、经历或目标有什么联系？",
        "视觉目标澄清": "💬 你实现{topic}这个功能时，想要呈现的视觉体验或交互感受是什么？它与整个项目的创意目标之间有何关联？",
        "细节决策说明": "💬 你做出{topic}这个细节调整时，背后的设计动机或想营造的感受是什么？这个细节是否强化了你的表达？"
    },
    "explorative": {
        "概念联系探索": "💬 你的灵感{topic}中有没有哪些元素可以结合起来，产生新的想象或叙事线索？",
        "模块体验关系": "💬 你能否思考一下当前的{topic} 几个功能模块，它们之间是否能更协调地服务于整体的视觉叙事或交互体验？",
        "视觉情感一致性": "💬 在你的作品中，{topic} 这些视觉元素之间是否保持了统一的风格和情绪？有没有可以更好融合它们的方式？"
    },
    "transformative": {
        "创意方向转变": "💬 如果从另一种角度（例如 {topic}）讲述这个故事，比如换一种情绪基调，会发生什么变化？",
        "功能方法重思": "💬 目前 {topic} 的功能效果是否与你想象中的体验存在偏差？如果是，有没有其他方式可以更贴切地表达你的意图？",
        "视觉风格调整": "💬 现在 {topic} 的整体风格是否与你想传达的核心感受完全契合？如果偏离了，你愿意在哪些部分做出调整以重建风格一致性？"
    }
}



# --- 关键词库 (用于明确意图匹配，保持不变) ---
DEEP_REFLECTION_KEYWORDS: Dict[str, Dict[str, str]] = {
    "explainable": {
        "动机说明": "动机说明",
        "阐明目标": "视觉目标澄清",
        "细节决策说明": "细节决策说明"
    },
    "explorative": {
        "概念联系探索": "概念联系探索",
        "模块体验关系": "模块体验关系",
        "情感视觉一致性": "视觉情感一致性"
    },
    "transformative": {
        "创意方法改变": "创意方向转变",
        "功能方法重思": "功能方法重思",
        "视觉风格调整": "视觉风格调整"
    }
}

# --- 主题提取 (用于明确意图，保持不变) ---
TOPIC_EXTRACTION_PROMPT_TEMPLATE = """
# 角色: 对话核心主题提取器
你的任务是阅读用户的最新问题，并用2到8个字的短语精准地概括出其中的核心创作主题或概念。
# 示例:
- 对话中: "我感觉现在的动画太僵硬了，像个机器人，没有生命感。" -> 核心主题: "像机器人的僵硬动画"
- 对话中:"我想让颜色更鲜艳，更有冲击力。" -> 核心主题: "鲜艳且有冲击力的颜色"
# 你的任务:
请从下面对话中提取讨论的核心主题。你的回答只能包含这个短语，不要有任何其他文字或标点。
## 对话上下文背景:
*** 相关的历史版本（记忆） ***
基于我们之前的探索，这里是一些过去代码版本的摘要，你可能会觉得有用。请使用这些信息来理解项目的演变和过去的想法。
{memory}

*** 当前对话（短期历史） ***
这是我们在用户最新提问之前的即时对话历史。
{history}

核心主题:
"""

# --- 用户Prompt模板 (用于过渡层，保持不变) ---
USER_PROMPT_TEMPLATE = """
{user_question}

我们对话的背景信息：
*** 相关的历史版本（记忆） ***
{memory}

*** 当前对话（短期历史） ***
{history}

这是我们讨论的代码：
```javascript
{current_code}
```
"""

# --- 明确意图响应生成器 (保持不变) ---
async def generate_deep_reflection_response(
    user_question: str,
   
    mode: str,
    llm: AzureChatOpenAI,
    history ,
    memory 
) -> Dict[str, str]:
    """
    为深度反思的“模糊意图”场景生成一个结构化的四段式响应。
    它会根据当前模式选择合适的思维链Prompt，动态注入反思模板，并一次性生成所有内容。
    """
    # 1. 根据模式选择对应的System Prompt模板
    PROMPT_MAPPING = {
        "explainable": DEEP_EXPLAINABLE_PROMPT,
        "explorative": DEEP_EXPLORATIVE_PROMPT,
        "transformative": DEEP_TRANSFORMATIVE_PROMPT,
    }
    system_prompt_template = PROMPT_MAPPING.get(mode)
    if not system_prompt_template:
        # 提供一个健壮的错误处理
        raise ValueError(f"无效的反思模式: '{mode}'。无法找到对应的Prompt。")

    # 2. 根据模式获取对应的反思问题模板库
    templates_for_mode = DEEP_REFLECTION_TEMPLATES.get(mode, {})
    # 将模板格式化为字符串，注入到System Prompt中
    formatted_templates = "\n".join([f"- {key}: \"{value}\"" for key, value in templates_for_mode.items()])

    # 3. 定义Human Message，包含用户的模糊想法和当前代码
    human_prompt = """
    
    {user_question}
    *** 当前代码与描述 ***
    这是我们目前正在讨论版本的完整代码。
   
    我们对话的背景信息：
    *** 相关的历史版本（记忆） ***
    基于我们之前的探索，这里是一些过去代码版本的摘要，你可能会觉得有用。请使用这些信息来理解项目的演变和过去的想法。
    {memory}

    *** 当前对话（短期历史） ***
    这是我们在用户最新提问之前的即时对话历史。
    {history}

    *** 你的任务 ***
    基于以上所有信息（历史记忆、近期对话以及当前代码），继续对话回答我的问题。
    """
    # 4. 组装完整的Chat Prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_template),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    # 5. 创建并调用LangChain链
    chain = chat_prompt | llm | JsonOutputParser()

    response = await chain.ainvoke({
        "reflection_templates": formatted_templates,
        "current_code": current_code,
        "user_question": user_question,
        "history": history,
        "memory": memory
    })

    return response

# --- 过渡层响应生成器 (保持不变) ---
async def generate_transition_response(
    user_question: str,
    current_code: str,
    memory: str,
    history: str,
    llm: AzureChatOpenAI
) -> Dict[str, str]:
    """
    为深度对话的第二轮生成一个包含总结和代码的过渡响应。
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(TRANSITION_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE)
    ])
    
    chain = chat_prompt | llm | JsonOutputParser()
    
    response = await chain.ainvoke({
        "current_code": current_code,
        "user_question": user_question,
        "history": history,
        "memory": memory
    })
    
    return response

# --- ‼️ 核心修改: 重构后的模糊意图响应生成器 ---
async def generate_vague_deep_reflection_response(
    user_question: str,
    current_code: str,
    mode: str,
    llm: AzureChatOpenAI,
    history ,
    memory 
) -> Dict[str, str]:
    """
    为深度反思的“模糊意图”场景生成一个结构化的四段式响应。
    它会根据当前模式选择合适的思维链Prompt，动态注入反思模板，并一次性生成所有内容。
    """
    # 1. 根据模式选择对应的System Prompt模板
    PROMPT_MAPPING = {
        "explainable": VAGUE_EXPLAINABLE_PROMPT,
        "explorative": VAGUE_EXPLORATIVE_PROMPT,
        "transformative": VAGUE_TRANSFORMATIVE_PROMPT,
    }
    system_prompt_template = PROMPT_MAPPING.get(mode)
    if not system_prompt_template:
        # 提供一个健壮的错误处理
        raise ValueError(f"无效的反思模式: '{mode}'。无法找到对应的Prompt。")

    # 2. 根据模式获取对应的反思问题模板库
    templates_for_mode = DEEP_REFLECTION_TEMPLATES.get(mode, {})
    # 将模板格式化为字符串，注入到System Prompt中
    formatted_templates = "\n".join([f"- {key}: \"{value}\"" for key, value in templates_for_mode.items()])

    # 3. 定义Human Message，包含用户的模糊想法和当前代码
    human_prompt = """
    
    {user_question}
    *** 当前代码与描述 ***
    这是我们目前正在讨论版本的完整代码。
   
    代码是：{current_code}
    我们对话的背景信息：

    *** 相关的历史版本（记忆） ***
    基于我们之前的探索，这里是一些过去代码版本的摘要，你可能会觉得有用。请使用这些信息来理解项目的演变和过去的想法。
    {memory}

    *** 当前对话（短期历史） ***
    这是我们在用户最新提问之前的即时对话历史。
    {history}

    *** 你的任务 ***
    基于以上所有信息（历史记忆、近期对话以及当前代码），继续对话回答我的问题。
    """
    # 4. 组装完整的Chat Prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_template),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    # 5. 创建并调用LangChain链
    chain = chat_prompt | llm | JsonOutputParser()

    response = await chain.ainvoke({
        "reflection_templates": formatted_templates,
        "current_code": current_code,
        "user_question": user_question,
        "history": history,
        "memory": memory
    })

    return response
