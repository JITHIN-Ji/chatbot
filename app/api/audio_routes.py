from fastapi import APIRouter, File, UploadFile
import os
from datetime import datetime
import whisper

router = APIRouter()

UPLOAD_DIR = "./uploads/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Whisper model once
model = whisper.load_model("base")  # or "small", "medium", etc.

@router.post("/")
async def upload_audio(audio: UploadFile = File(...)):
    # 1. Save file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{audio.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await audio.read())

    # 2. Transcribe with Whisper (auto-detect language)
    try:
        result = model.transcribe(file_path, language=None)  # automatic language detection
        transcript = result["text"]
        detected_lang = result.get("language", "en")
    except Exception as e:
        return {
            "message": "Audio saved, but transcription failed",
            "filename": filename,
            "error": str(e)
        }

    # 3. Return everything including detected language
    # 3. Clean up the audio file
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temp audio file {file_path}: {e}")

    # 4. Return result
    return {
        "message": "Audio transcribed successfully",
        "filename": filename,
        "transcript": transcript,
        "language": detected_lang
    }
