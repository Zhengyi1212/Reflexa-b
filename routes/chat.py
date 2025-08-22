# api/chat.py
from fastapi import APIRouter, HTTPException, status, Depends
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from services.services import get_vector_store
from utility.schemas import ChatRequest
from utility.config import settings
from utility.prompt import (
    USER_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    EXPLAINABLE_SYSTEM_PROMPT,
    EXPLORATIVE_SYSTEM_PROMPT,
    TRANSFORMATIVE_SYSTEM_PROMPT
)

# ‼️ 修改点: 更新导入的函数
from utility.deep_chat import (
    generate_deep_reflection_response,
    generate_transition_response,
    DEEP_REFLECTION_KEYWORDS,
    generate_vague_deep_reflection_response # 导入重构后的函数
)

router = APIRouter()

# --- LLM 设置 (保持不变) ---
llm = AzureChatOpenAI(
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_deployment=settings.AZURE_OPENAI_MODEL_NAME,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    temperature=0.7
)

# --- 辅助函数 (保持不变) ---
def format_memories_for_prompt(memories: list) -> str:
    if not memories:
        return "No relevant historical versions found."
    formatted = []
    for i, mem in enumerate(memories):
        formatted.append(
            f"Memory {i+1} (Version ID: {mem.get('version_id', 'N/A')}):\n"
            f"- 版本总结: {mem.get('ai_summary', 'N/A')}\n"
        )
    return "\n".join(formatted)

def format_history_for_prompt(history: list) -> str:
    if not history:
        return "No recent conversation history."
    return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in history])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    vector_store: Chroma = Depends(get_vector_store)
):
    """
    处理聊天请求。
    根据交互模式和次数，路由到普通聊天、过渡层或深度反思。
    """
    try:
        # --- 数据准备 (保持不变) ---
        retrieval_query = f"{request.code_description}\n{request.user_question}"
        where_clause = {
            "$and": [
                {"session_id": {"$eq": request.session_id}},
                {"version_id": {"$ne": request.version_id}}
            ]
        }
        results = vector_store.similarity_search(query=retrieval_query, k=3, filter=where_clause)
        retrieved_metadatas = [doc.metadata for doc in results]
        formatted_memories = format_memories_for_prompt(retrieved_metadatas)
        formatted_history = format_history_for_prompt(request.short_term_history)

        # --- 核心路由逻辑 ---
        if request.type in ['explainable', 'explorative', 'transformative']:
            
            # --- 第二次交互: 过渡层 (保持不变) ---
            if request.interaction_count == 2:
                print(f"🌀 [会话: {request.session_id}] 进入过渡反思层 (第 {request.interaction_count} 次)。")
                transition_response = await generate_transition_response(
                    user_question=request.user_question,
                    current_code=request.code,
                    memory=formatted_memories,
                    history=formatted_history,
                    llm=llm
                )
                transition_sentences = {
                    "explainable": "💡如果你愿意，我们可以从**动机说明**,**阐明目标**或**细节决策说明**选择一个方向继续进行思考",
                    "explorative": "💡如果你愿意，我们可以从**概念联系探索**,**模块体验关系**或**情感视觉一致性**选择一个方向继续进行思考",
                    "transformative": "💡如果你愿意，我们可以从**创意方法改变**,**功能方法重思**或**视觉风格调整**选择一个方向继续进行思考"
                }
                transition_response['advice'] = transition_sentences.get(request.type, "")
                print(f"✅ [会话: {request.session_id}] 过渡层响应已生成。")
                print(transition_response)
                return transition_response

            # --- ‼️ 第三次及以上交互: 深度反思 (核心修改点) ---
            elif request.interaction_count >= 3:
                print(f"🚀 [会话: {request.session_id}] 进入深度反思模式 (第 {request.interaction_count} 次)。")
                
                # --- 情况1: 明确意图 - 通过关键词直接匹配 ---
                matched_category = None
                keywords_for_mode = DEEP_REFLECTION_KEYWORDS.get(request.type, {})
                for keyword, category in keywords_for_mode.items():
                    if keyword in request.user_question:
                        matched_category = category
                        print(f"🎯 [会话: {request.session_id}] 匹配到关键词 '{keyword}'，意图明确为 '{category}'。")
                        break
                
                if matched_category:
                    reflection_string = await generate_deep_reflection_response(
                        mode=request.type,
                        category=matched_category,
                        history= formatted_history,
                        memory = formatted_memories,
                        llm=llm,
                        
                    )
                    print(f"💬 [会话: {request.session_id}] 已生成模板化反思问题。")
                    return {"reflection": reflection_string}

                # --- 情况2: 模糊意图 - 调用新的结构化响应生成器 ---
                else:
                    print(f"🌀 [会话: {request.session_id}] 未匹配到关键词，启动模糊意图结构化响应。")
                    
                    # 直接调用重构后的函数，它将处理所有逻辑
                    structured_response = await generate_vague_deep_reflection_response(
                        user_question=request.user_question,
                        current_code=request.code,
                        mode=request.type,
                        llm=llm,
                        history = formatted_history,
                        memory = formatted_memories
                    )
                    
                    print(f"✅ [会话: {request.session_id}] 已生成四段式模糊反思响应。")
                    return structured_response

        # --- 普通聊天流程 (保持不变) ---
        print(f"💬 [会话: {request.session_id}] 普通聊天模式 (第 {request.interaction_count} 次)。")
        system_prompts = {
            'explainable': EXPLAINABLE_SYSTEM_PROMPT,
            'explorative': EXPLORATIVE_SYSTEM_PROMPT,
            'transformative': TRANSFORMATIVE_SYSTEM_PROMPT,
            'general': GENERAL_SYSTEM_PROMPT
        }
        system_prompt = system_prompts.get(request.type, GENERAL_SYSTEM_PROMPT)
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(USER_PROMPT)
        ])
        chain = chat_prompt | llm | JsonOutputParser()
        chain_input = {
            "retrieved_memories": formatted_memories,
            "short_term_history": formatted_history,
            "code_description": request.code_description,
            "current_code": request.code,
            "user_question": request.user_question,
        }
        response = await chain.ainvoke(chain_input)
        print(f"✅ [会话: {request.session_id}] 普通聊天响应已生成。")
        print(response)
        return response

    except Exception as e:
        print(f"❌ 在 /chat 端点发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"An internal error occurred while processing the chat: {str(e)}"
        )
