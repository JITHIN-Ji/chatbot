from pydantic import BaseModel
from typing import List, Optional
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str # "user" or "assistant" or "system"
    content: str

from app.core.config import settings

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None
    document_ids: Optional[List[str]] = None
    llm_provider: Optional[str] = settings.DEFAULT_LLM_PROVIDER
    language: Optional[str] = "auto"  # ✅ Default to auto-detect instead of hardcoding "en"


class ChatResponse(BaseModel):
    answer: str
    # accept full‑featured dicts coming from RAGPipeline
    sources: Optional[List[Dict[str, Any]]] = None
    # If you’d like a stricter schema, define a separate Source model instead.
# --------------------------------

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
