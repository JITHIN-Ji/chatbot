from app.services.rag_pipeline import RAGPipeline
from app.models.chat_models import ChatRequest, ChatResponse, DocumentUploadResponse,  Message
import logging
import os
from langdetect import detect  
import pycountry
from typing import List, Dict
from openai import OpenAI
from app.models import question_store   # ← NEW

logger = logging.getLogger(__name__)

# ── NEW IMPORTS ───────────────────────────────────────────────



# Pick up the key you just added to .env
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rephrase_query_with_context(
    history: List[Dict[str, str]],
    latest_query: str,
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
) -> str:
    """Rewrite a follow‑up question so it’s a full, standalone question."""
    if not history:
        return latest_query

    history_str = "\n".join([f"- {h['query']}" for h in reversed(history)])
    prompt = f"""
You are an AI assistant that rewrites the Latest Query into a complete question
by adding context from the Chat History.

Chat History:
{history_str}

Latest Query:
{latest_query}

Only respond with the rephrased query.
"""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Rephrase failed: {e}")
        return latest_query


class AgentService:
    def __init__(self, user_id: str):
        
        self.rag_pipeline = RAGPipeline(user_id)
        logger.info("AgentService initialized.")

    async def handle_document_upload(self, file_path: str, document_id: str) -> DocumentUploadResponse:
        """
        Agent decides to process and embed the uploaded document.
        """
        logger.info(f"Agent handling document upload: {document_id} at {file_path}")

        # ✅ Clear previous FAISS index and metadata before processing new document
        
        logger.info("Cleared previous FAISS index and metadata.")

        # Now process and embed the uploaded document
        success, message = await self.rag_pipeline.process_and_embed_document(file_path, document_id)

        if not success:
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up failed upload: {file_path}")
            except Exception:
                logger.warning(f"Failed to clean up upload file {file_path} after error: {message}")
            return DocumentUploadResponse(message=message, document_id=document_id, filename=document_id)

        return DocumentUploadResponse(message=message, document_id=document_id, filename=document_id)
    
    async def handle_image_upload(self, extracted_text: str, document_id: str) -> DocumentUploadResponse:
        logger.info(f"Agent handling image upload for: {document_id}")

        # ✅ 1)  CLEAR any previous embeddings (same logic as in handle_document_upload)
        
        logger.info("Cleared previous FAISS index and metadata.")

        # 2) pretend we have a one‑page “document” that contains the OCR text
        pages = [{"index": 0, "label": "1", "text": extracted_text}]

        # 3) chunk, embed, store
        chunks, meta_per_chunk = self.rag_pipeline.doc_processor.chunk_text(pages)
        if not chunks:
            msg = f"No content found in OCR for {document_id}"
            logger.warning(msg)
            return DocumentUploadResponse(message=msg, document_id=document_id, filename=document_id)

        await self.rag_pipeline.process_and_embed_document_from_chunks(
            chunks, meta_per_chunk, document_id
        )

        return DocumentUploadResponse(
            message="Image processed and indexed via OCR.",
            document_id=document_id,
            filename=document_id,
        )


   # ───────────────────────────────────────────────────────────────
    async def handle_chat_query(self, request: ChatRequest, user_id: str) -> ChatResponse:
        """
        Process a chat query, using a per‑user 10‑question memory.
        `user_id` comes from the Google JWT (payload['sub']).
        """
        logger.info(f"Agent handling chat query: {request.query}")

        # 1. Load this user's last‑10 questions for context
        chat_history_list = await question_store.get_recent(user_id)

        # 2. Re‑phrase follow‑up to a standalone query
        standalone_query = rephrase_query_with_context(
            history=chat_history_list,
            latest_query=request.query,
        )

        # 3. Language handling (same logic as before)
        language_code = request.language or "auto"

        if language_code == "auto":
            try:
                
                language_code = detect(standalone_query)
            except Exception:
                language_code = "en"   # fallback to English

        try:
            language_name = pycountry.languages.get(alpha_2=language_code).name
        except Exception:
            language_name = "English"

        modified_query = f"Answer in {language_name}:\n{standalone_query}"

        logger.info(f"Modified query: {modified_query}")

        # 4. Run RAG pipeline
        result = await self.rag_pipeline.query(
            user_query=modified_query,
            llm_provider=request.llm_provider,
            document_id_filters=request.document_ids,
            chat_history=chat_history_list,
        )
        if not result["sources"]:                     # nothing found in PDFs
    # Ask again with *no* document context so the LLM can rely
    # on its own knowledge base.
            result["answer"] = await self.rag_pipeline.generate_answer(
                query=standalone_query,               # original question
                context_chunks=[],                    # empty context
                llm_provider=request.llm_provider,
                chat_history=chat_history_list,
            )

        # 5. Save this question for future context (per user)
        await question_store.save_query(user_id, standalone_query)

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
        )
# ───────────────────────────────────────────────────────────────

# Example usage (for testing - requires running within an async context if using await)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # This part is tricky to run directly due to async and dependencies
    # You would typically test this through API endpoints or dedicated test scripts
    async def main_test():
        agent = AgentService()
        
        logger.info("--- Testing Chat Query --- ")
        # Test chat query (assuming some documents are already processed and indexed)
        # This requires the vector store to be populated for meaningful RAG results
        chat_req_gemini = ChatRequest(
            query="What is retrieval augmented generation?",
            chat_history=[
                Message(role="user", content="Tell me about RAG."),
                Message(role="assistant", content="Retrieval Augmented Generation combines retrieval with generation models.")
            ],
            llm_provider="gemini" # Test with gemini
        )
        response_gemini = await agent.handle_chat_query(chat_req_gemini)
        logger.info(f"Chat response (Gemini): {response_gemini.answer}")
        if response_gemini.sources:
            logger.info(f"Sources: {response_gemini.sources}")

        chat_req_groq = ChatRequest(
            query="Explain it like I am five.",
            chat_history=[
                Message(role="user", content="Tell me about RAG."),
                Message(role="assistant", content="Retrieval Augmented Generation combines retrieval with generation models."),
                Message(role="user", content="What is retrieval augmented generation?"),
                Message(role="assistant", content=response_gemini.answer) # Continue conversation
            ],
            llm_provider="groq" # Test with groq
        )
        response_groq = await agent.handle_chat_query(chat_req_groq)
        logger.info(f"Chat response (Groq): {response_groq.answer}")
        if response_groq.sources:
            logger.info(f"Sources: {response_groq.sources}")

        # Test document upload (requires a dummy PDF)
        # logger.info("--- Testing Document Upload --- ")
        # dummy_pdf_filename = "test_agent_upload.pdf"
        # dummy_pdf_path = os.path.join(settings.UPLOAD_DIR, dummy_pdf_filename)
        # Ensure UPLOAD_DIR exists: 
        # if not os.path.exists(settings.UPLOAD_DIR):
        #    os.makedirs(settings.UPLOAD_DIR)
        #    logger.info(f"Created UPLOAD_DIR: {settings.UPLOAD_DIR}")

        # if not os.path.exists(dummy_pdf_path):
        #     try:
        #         with open(dummy_pdf_path, "w") as f: # Create a tiny dummy file (not a real PDF)
        #             f.write("dummy pdf content for agent test") 
        #         logger.info(f"Created dummy file for upload test: {dummy_pdf_path}")
        #     except Exception as e:
        #         logger.error(f"Could not create dummy file {dummy_pdf_path}: {e}")
        
        # if os.path.exists(dummy_pdf_path):
        #     upload_response = await agent.handle_document_upload(dummy_pdf_path, dummy_pdf_filename)
        #     logger.info(f"Upload response: {upload_response.message}")
        # else:
        #     logger.warning(f"Skipping upload test, dummy file not found or creatable: {dummy_pdf_path}")

    import asyncio
    try:
        asyncio.run(main_test())
    except Exception as e:
        logger.error(f"Error running main_test: {e}", exc_info=True)
    logger.info("AgentService main_test finished. Run full tests via API or dedicated test scripts.")
