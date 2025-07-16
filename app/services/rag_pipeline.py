from app.services.document_processor import DocumentProcessor
from app.services.embedding_generator import EmbeddingGenerator

from app.core.config import settings
from app.services.llm_clients import GeminiClient, GroqClient
from typing import List, Dict, Any, Optional
import os
import logging
import numpy as np
from app.deps.vector import get_vs
import asyncio
import time
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, user_id: str):
        # Instantiate clients first based on API key availability
        self.gemini_client = GeminiClient() if settings.GEMINI_API_KEY else None
        self.groq_client = GroqClient() if settings.GROQ_API_KEY else None

        # Now instantiate services, injecting dependencies as needed
        self.doc_processor = DocumentProcessor()
        self.embed_generator = EmbeddingGenerator(gemini_client=self.gemini_client)
        
        self.vector_store = get_vs(user_id) 
        logger.info("RAGPipeline initialized.")

     # at top of the file if not already imported

    async def process_and_embed_document(self, file_path: str, document_id: str) -> tuple[bool, str]:
        logger.info(f"Processing document: {file_path} with ID: {document_id}")
        
        # ✅ Measure Text Extraction Time
        start = time.time()
        pages = self.doc_processor.extract_text_from_pdf(file_path)
        logger.info(f"Text extraction took {time.time() - start:.2f} seconds")

        if not pages:
            message = f"No text extracted from {file_path}"
            logger.warning(message)
            return False, message

        # ✅ Chunking
        start = time.time()
        chunks, meta_per_chunk = self.doc_processor.chunk_text(pages)
        logger.info(f"Chunking took {time.time() - start:.2f} seconds")

        if not chunks:
            message = f"No chunks created for {file_path}"
            logger.warning(message)
            return False, message

        logger.info(f"Generated {len(chunks)} chunks for document {document_id}.")

        # ✅ Embedding
        start = time.time()
        BATCH_SIZE = 64 
        async def embed_all_batches(chunks, batch_size):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Embedding batch {i} to {i + len(batch)}...")
                tasks.append(self.embed_generator.generate_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))
            results = await asyncio.gather(*tasks)
            return [embedding for batch in results if batch is not None for embedding in batch]

        # Call the async batch function
        all_embeddings = await embed_all_batches(chunks, BATCH_SIZE)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Embedding generation for all chunks took {time.time() - start:.2f} seconds")

        if embeddings is None or embeddings.size == 0:
            message = f"Failed to generate embeddings for {document_id}"
            logger.error(message)
            return False, message

        # ✅ Store in vector DB
        self.vector_store.add_embeddings(
            embeddings,
            [
                {
                    'doc_id': document_id,
                    'chunk_text': chunk,
                    'chunk_index': i,
                    'page': meta_per_chunk[i]['page'], # ✅ include page number
                    'label': meta_per_chunk[i]['label']  
                }
                for i, chunk in enumerate(chunks)
            ]
        )

        message = f"Embeddings for {document_id} added to vector store."
        logger.info(message)
        return True, message
    
    async def process_and_embed_document_from_chunks(self, chunks: List[str], meta_per_chunk: List[dict], document_id: str) -> tuple[bool, str]:
        """
        Used when OCR text (from image) is already available and chunked.
        """
        logger.info(f"Embedding OCR-based chunks for document {document_id}")
        BATCH_SIZE = 64

        async def embed_all_batches(chunks, batch_size):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                tasks.append(self.embed_generator.generate_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))
            results = await asyncio.gather(*tasks)
            return [embedding for batch in results if batch is not None for embedding in batch]

        all_embeddings = await embed_all_batches(chunks, BATCH_SIZE)
        embeddings = np.array(all_embeddings, dtype=np.float32)

        if embeddings is None or embeddings.size == 0:
            return False, f"Failed to embed OCR chunks for {document_id}"

        self.vector_store.add_embeddings(
            embeddings,
            [
                {
                    "doc_id": document_id,
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "page": meta_per_chunk[i].get("page", -1),
                    "label": meta_per_chunk[i].get("label", "1")
                }
                for i, chunk in enumerate(chunks)
            ]
        )

        return True, f"OCR document {document_id} embedded successfully."



    async def retrieve_relevant_chunks(self, query: str, k: int = settings.TOP_K, document_id_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant text chunks for a given query, optionally filtered by document_ids."""
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            logger.warning("Vector store is empty or not initialized. Cannot retrieve.")
            return []
             
        query_embedding = await self.embed_generator.generate_embeddings([query], task_type="RETRIEVAL_QUERY")
        if query_embedding is None or query_embedding.size == 0:
            logger.error("Failed to generate query embedding.")
            return []
        
        results = self.vector_store.search(query_embedding, k=k, document_id_filters=document_id_filters)
        
        retrieved_chunks = []
        for res_doc_id, meta, distance in results:
            if meta:
                chunk_info = {
                    'document_id': meta.get('doc_id', res_doc_id),  # ✅ Fix: use meta.get('doc_id')
                    'text': meta.get('chunk_text', ''),
                    'chunk_index': meta.get('chunk_index', -1),
                    'page': meta.get('page', -1),  # ✅ Add page number here
                    'label'      : meta.get('label', 'N/A'), 
                    'score': 1 - distance # Convert distance to similarity score
                }
                retrieved_chunks.append(chunk_info)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
        return retrieved_chunks

    async def generate_answer(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        llm_provider: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """Generates an answer using an LLM based on query, context, and chat history."""
        context_str = "\n".join([
            f"Document: {chunk['document_id']} (Page {chunk.get('page', 'N/A')}):\n{chunk['text']}"
            for chunk in context_chunks
        ])

        
        if context_str:
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the following context from uploaded documents and the chat history. If the context is not sufficient or irrelevant, say so and try to answer from your general knowledge if appropriate.

Relevant Context from Documents:
---
{context_str}
---

User Question: {query}"""
        else:
             prompt = f"""You are a helpful AI assistant. Answer the user's question based on the chat history. No specific context from documents was found for this query.

User Question: {query}"""

        logger.debug(f"--- Prompt for LLM ({llm_provider}) ---\nHistory: {chat_history}\nContext length: {len(context_str)}\nPrompt: {prompt[:500]}...\n-----------------------")

        llm_response = "No LLM provider was selected or available."
        
        if llm_provider == "gemini":
            if self.gemini_client:
                llm_response = await self.gemini_client.generate_text(prompt, history=chat_history)
            else:
                llm_response = "Gemini client is not available (e.g., API key missing)."
        elif llm_provider == "groq":
            if self.groq_client:
                llm_response = await self.groq_client.generate_text(prompt, history=chat_history)
            else:
                llm_response = "Groq client is not available (e.g., API key missing)."
        
        return llm_response

    async def query(self, user_query: str, llm_provider: str, document_id_filters: Optional[List[str]] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Handles a user query, performs RAG, and returns an answer."""
        logger.info(f"RAG pipeline received query: '{user_query}', LLM: {llm_provider}, Doc Filters: {document_id_filters}")
        
        retrieved_chunks_data = await self.retrieve_relevant_chunks(
            query=user_query,
            k=settings.TOP_K,                     # ✅ pull TOP_K from config.py
            document_id_filters=document_id_filters,
        )

        
        answer = await self.generate_answer(user_query, retrieved_chunks_data, llm_provider, chat_history)
        
        # ✅ Only keep one source per (document_id, page) combination
        unique_sources = {}
        for chunk in retrieved_chunks_data:
            key = (chunk['document_id'], chunk.get('page', -1))
            if key not in unique_sources:
                unique_sources[key] = {
                    'document_id': chunk['document_id'],
                    'page': chunk.get('page', -1),
                    'label': chunk.get('label', 'N/A'),
                    'score': chunk['score']
                }

        return {
            "answer": answer,
            "sources": list(unique_sources.values())
        }
