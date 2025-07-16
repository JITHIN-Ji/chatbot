from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    # --- Project Settings ---
    PROJECT_NAME: str = "Chatbot API"
    API_V1_STR: str = "/api/v1"

    # --- General LLM & Embedding Settings --- 
    # Generic LLM API Key (optional, specific keys below are preferred)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") 

    # Default embedding model (Sentence Transformers, used if Gemini embedding is not configured)
    DEFAULT_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Gemini Settings --- 
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"  # Or your preferred Gemini generation model
    GEMINI_EMBEDDING_MODEL_NAME: str = "models/text-embedding-004" # Or your preferred Gemini embedding model

    # --- Groq Settings --- 
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME: str = "llama3-8b-8192" # Or your preferred Groq model (e.g., mixtral-8x7b-32768)

    # --- Storage and Processing Settings --- 
    # Data directory where source documents are stored (can be overridden by .env)
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # Vector store path (can be overridden by .env)
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store_data/faiss_index")

    # Upload directory (can be overridden by .env)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # FAISS metadata file name
    FAISS_METADATA_FILE: str = "faiss_metadata.json"

    # Chunking parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    TOP_K: int = 10     

    # --- Agent Settings ---
    DEFAULT_LLM_PROVIDER: str = "gemini" # Can be 'gemini' or 'groq'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from .env that are not defined in Settings

settings = Settings()

# Determine active embedding model based on API key availability
ACTIVE_EMBEDDING_MODEL_NAME = settings.GEMINI_EMBEDDING_MODEL_NAME if settings.GEMINI_API_KEY else settings.DEFAULT_EMBEDDING_MODEL_NAME
USING_GEMINI_EMBEDDINGS = bool(settings.GEMINI_API_KEY)
print(f"[Config] UPLOAD_DIR: {settings.UPLOAD_DIR}")
print(f"[Config] VECTOR_STORE_PATH: {settings.VECTOR_STORE_PATH}")
print(f"[Config] Active embedding model: {ACTIVE_EMBEDDING_MODEL_NAME}")
print(f"[Config] Using Gemini embeddings: {USING_GEMINI_EMBEDDINGS}")
print(f"[Config] Gemini API Key Loaded: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
print(f"[Config] Groq API Key Loaded: {'Yes' if settings.GROQ_API_KEY else 'No'}")

