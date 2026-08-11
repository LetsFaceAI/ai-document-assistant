"""
File: backend/app/services/chat_services.py
Description: Business logic layer orchestrating Retrieval -> Prompt Construction -> LLM Execution.
"""

import logging
from typing import Dict, Any

from app.clients.llm_client import NvidiaClient
from app.services.prompt_builder_service import PromptBuilderService
# Adjust import path to match your Retrieval Service class
from app.services.retrieval_service import RetrievalService 
from app.core.config import settings
from app.models.chat import SourceCitation
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStore

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.llm_client = NvidiaClient()
        self.prompt_builder = PromptBuilderService()
        self.retrieval_service = RetrievalService(
            embedding_service=EmbeddingService(), 
            vector_store=ChromaVectorStore())

    async def process_user_message(self, user_message: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full RAG Pipeline execution.
        
        1. Retrieve top-k context chunks via Vector Search & RRF.
        2. Construct token-bounded system + user messages.
        3. Generate grounded answer from the LLM.
        """
        logger.info(f"Processing user query: '{user_message}'")

        # Step 1: Retrieval Pipeline
        retrieval_result, _ = await self.retrieval_service.retrieve(
            question=user_message, 
            top_k=top_k
        )

        # Step 2: Prompt Builder (Converts RetrievalResult into PromptContext)
        prompt_context = self.prompt_builder.build_prompt(
            question=user_message,
            retrieved_chunks=retrieval_result.retrieved_chunks
        )

        # Step 3: Convert Pydantic LLMMessage objects to dicts for LLM client
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in prompt_context.messages
        ]

        # Step 4: Delegate API transport to LLM Client
        model_name = settings.openrouter_base_model
        llm_response = await self.llm_client.get_chat_completion(
            model=model_name, 
            messages=formatted_messages
        )

        # Step 5: Extract source citations from chunks that actually made it into the prompt
        used_chunks = retrieval_result.retrieved_chunks[:prompt_context.chunks_used]
        citations = [
            SourceCitation(
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index
            )
            for chunk in used_chunks
        ]

        return {
            "response": llm_response,
            "sources": citations,
            "tokens_used": prompt_context.estimated_tokens
        }