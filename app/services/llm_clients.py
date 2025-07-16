import google.generativeai as genai
from groq import Groq, AsyncGroq
from app.core.config import settings
from typing import List, Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API Key not found. GeminiClient will not be functional.")
            self.gen_model = None
            self.emb_model_name = None
            return
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.gen_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        self.emb_model_name = settings.GEMINI_EMBEDDING_MODEL_NAME
        logger.info(f"GeminiClient initialized with generation model: {settings.GEMINI_MODEL_NAME} and embedding model: {self.emb_model_name}")

    async def generate_text(self, prompt: str, history: Optional[List[Any]] = None) -> str:
        if not self.gen_model:
            return "Gemini client not configured due to missing API key."
        
        gemini_history = []
        if history:
            for item in history:
                role = 'user' if item.get('role') == 'user' else 'model'
                gemini_history.append({'role': role, 'parts': [item.get('content', '')]})
        
        try:
            chat_session = self.gen_model.start_chat(history=gemini_history if gemini_history else None)
            response = await chat_session.send_message_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            return f"Error from Gemini: {str(e)}"


class GroqClient:
    def __init__(self):
        if not settings.GROQ_API_KEY:
            logger.warning("Groq API Key not found. GroqClient will not be functional.")
            self.client = None
            self.model_name = None
            return

        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model_name = settings.GROQ_MODEL_NAME
        logger.info(f"GroqClient initialized with model: {self.model_name}")

    async def generate_text(self, prompt: str, history: Optional[List[Any]] = None) -> str:
        if not self.client:
            return "Groq client not configured due to missing API key."
        
        messages = []
        if history:
            for item in history:
                messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": prompt})
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Groq: {e}")
            return f"Error from Groq: {str(e)}"

# class WebSearchClient:
#     def __init__(self):
#         self.ddgs = DDGS()
#         logger.info("WebSearchClient initialized with DuckDuckGo.")
# 
#     async def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
#         logger.info(f"Performing web search for: {query} (max_results: {max_results})")
#         try:
#             # DDGS().text() is synchronous, using DDGS().atext() for async if available
#             # As of duckduckgo-search 5.3.0b1, atext is available.
#             # If using an older version, consider running in a thread pool executor.
#             # For simplicity, we'll assume a version that supports await self.ddgs.atext(...)
#             # If not, this part would need to be run in an executor.
#             
#             # Check if atext method exists, otherwise fall back to sync version in executor
#             if hasattr(self.ddgs, 'atext'):
#                 results = await self.ddgs.atext(query, max_results=max_results)
#             else: # Fallback for older versions or if atext is not available
#                 loop = asyncio.get_event_loop()
#                 results = await loop.run_in_executor(None, self.ddgs.text, query, max_results)
#             
#             formatted_results = []
#             if results:
#                 for res in results:
#                     formatted_results.append({
#                         "title": res.get('title', 'N/A'),
#                         "snippet": res.get('body', 'N/A'), # 'body' usually contains the snippet
#                         "url": res.get('href', 'N/A')
#                     })
#             logger.info(f"Web search returned {len(formatted_results)} results.")
#             return formatted_results
#         except Exception as e:
#             logger.error(f"Error during web search: {e}")
#             return []


