import numpy as np
from typing import List, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, gemini_client=None): 
        self.embedding_model = None
        from app.core.config import USING_GEMINI_EMBEDDINGS
        self.using_gemini = USING_GEMINI_EMBEDDINGS and settings.GEMINI_API_KEY

        if self.using_gemini:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=settings.GEMINI_EMBEDDING_MODEL_NAME,
                    task_type="retrieval_document" 
                )
                logger.info(f"EmbeddingGenerator configured to use LangChain Gemini model: {settings.GEMINI_EMBEDDING_MODEL_NAME}")
            except ImportError:
                logger.error("Package 'langchain-google-genai' not found. Please install it.")
                self.using_gemini = False
            except Exception as e:
                logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
                self.using_gemini = False
#    settings.SENTENCE_TRANSFORMER_MODEL to    settings.DEFAULT_EMBEDDING_MODEL_NAME  
        if not self.using_gemini:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(settings.DEFAULT_EMBEDDING_MODEL_NAME)
                logger.info(f"Falling back to SentenceTransformer model: {settings.DEFAULT_EMBEDDING_MODEL_NAME}")
            except ImportError:
                logger.error("Package 'sentence-transformers' not found. Please install it.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")

    async def generate_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> Optional[np.ndarray]:
        if not self.embedding_model:
            logger.error("No embedding model is available.")
            return None

        try:
            if self.using_gemini:
                if task_type == "retrieval_query":
                    embeddings = self.embedding_model.embed_query(texts[0]) 
                    return np.array([embeddings], dtype=np.float32)
                else:
                    embeddings = self.embedding_model.embed_documents(texts)
                    return np.array(embeddings, dtype=np.float32)
            else:
                embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
                return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return None

