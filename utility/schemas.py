# utility/schemas.py
from pydantic import BaseModel
from typing import List, Dict

# --- Schemas for other parts of the application (unchanged) ---

class AddVersionRequest(BaseModel):
    session_id: str
    version_id: str
    code: str
    description: str

class DeleteVersionRequest(BaseModel):
    session_id: str
    version_id: str

class ChatRequest(BaseModel):
    session_id: str
    version_id: str
    code: str
    code_description: str
    short_term_history: List[Dict[str, str]]
    user_question: str
    type: str
    interaction_count: int

class MergeRequest(BaseModel):
    session_id: str
    version_id_1: str
    code_1: str
    description_1: str
    version_id_2: str
    code_2: str
    description_2: str
    instruction: str
    mode: str

# --- ‼️【修改】Schemas for the 'modify' feature ---

# StyleRecommendRequest is no longer needed as the new endpoint takes no arguments.
# We can safely remove it.
