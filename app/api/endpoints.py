from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse
from app.services.agent import AgentService
from app.deps.auth import get_current_user_id
from app.services.ocr_reader import OCRReader
from app.api.audio_routes import router as audio_router
from app.core.config import settings
import os, shutil

router = APIRouter()
router.include_router(audio_router, prefix="/upload-audio", tags=["Audio"])

ocr_reader = OCRReader()

# Dependency that provides a user‑scoped AgentService
async def get_agent_service(user_id: str = Depends(get_current_user_id)):
    return AgentService(user_id)

@router.post("/upload/", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    agent_service: AgentService = Depends(get_agent_service),
):
    ...

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # ✅ Support PDF and image extensions
    ext = file.filename.lower().split(".")[-1]
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        raise HTTPException(status_code=400, detail="Only PDF or image files are allowed.")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        file.file.close()

    # ✅ PDF → embed flow | Image → OCR + embed
    if ext == "pdf":
        result = await agent_service.handle_document_upload(file_path, file.filename)
    else:
        text = ocr_reader.extract_text(file_path)
        result = await agent_service.handle_image_upload(text, file.filename)

    return result

@router.post("/chat/", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest, 
    user_id: str = Depends(get_current_user_id),      # ← NEW
    agent_service: AgentService = Depends(get_agent_service)
):
    response = await agent_service.handle_chat_query(request, user_id)  # ← NEW
    return response
