# services/summarizer.py
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from utility.config import settings

SUMMARY_PROPMT = """
# 你是一个P5.js代码分析专家。

# 你的任务是分析给定的p5.js代码片段，并生成一个简洁的、30字以内的摘要。重点描述代码的功能、目的和整体行为。

# 任务要求：
- **只关注代码的功能与目的，避免解释具体语法或细节**。
- **绝对不要提到代码语言或代码层级的细节**。
- **摘要必须清晰、简洁，仅包含功能描述**。

# 返回要求：
- **摘要内容只能使用中文，严禁使用任何其他语言！**
- **摘要长度严格限制在30字以内！**
"""


class CodeSummarizerService:
    def __init__(self):
        self._llm = AzureChatOpenAI(
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_MODEL_NAME,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            temperature=0.2 # Lower temperature for more factual summaries
        )
    

    async def summarize_code(self, code: str) -> str:
        """
        Calls an LLM to generate a concise summary of the provided code.
        """
        messages = [
            SystemMessage(
                content=SUMMARY_PROPMT
            ),
            HumanMessage(
                content=f"请总结这个代码:\n\n```\n{code}\n```"
            )
        ]
        try:
            response = await self._llm.ainvoke(messages)
            summary = response.content
            print("Successfully generated code summary.")
            return summary
        except Exception as e:
            print(f"Error during code summarization: {e}")
            return "Failed to generate AI summary." # Return a fallback string

# Singleton instance
summarizer_service = CodeSummarizerService()