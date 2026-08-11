"""
File: backend/app/services/prompt_builder_service.py
Description: Transforms user questions and retrieved vector chunks into 
structured, token-safe LLM prompts.
"""

import logging
from typing import List, Optional

from app.core.config import settings
from app.models.domain import RetrievedChunk
from app.models.llm import LLMMessage, MessageRole, PromptContext
from app.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

class PromptBuilderService:
    def __init__(self):
        self.system_prompt = settings.SYSTEM_PROMPT
        self.user_template = settings.USER_PROMPT_TEMPLATE
        self.max_tokens = settings.MAX_CONTEXT_TOKENS

    def build_prompt(
        self, 
        question: str, 
        retrieved_chunks: List[RetrievedChunk], 
        chat_history: Optional[List[LLMMessage]] = None
    ) -> PromptContext:
        """
        Builds the complete LLM message array safely within token limits.
        """
        logger.info(f"Building prompt for question: '{question}' with {len(retrieved_chunks)} candidate chunks.")

        # 1. Calculate static overhead (System prompt, User template shell, Question)
        static_overhead_text = self.system_prompt + self.user_template.format(formatted_context="", question=question)
        static_tokens = count_tokens(static_overhead_text)
        
        # Calculate how many tokens we have left exclusively for document context
        available_context_tokens = self.max_tokens - static_tokens
        
        # 2. Iteratively format chunks until we hit the token budget
        formatted_context_blocks = []
        context_tokens_used = 0
        chunks_used = 0
        is_truncated = False

        for chunk in retrieved_chunks:
            # Format the individual chunk with citation metadata
            chunk_text = (
                f"--- Document: {chunk.filename}, Page: {chunk.page_number} ---\n"
                f"{chunk.text}\n"
            )
            
            chunk_token_count = count_tokens(chunk_text)

            # Check if adding this chunk exceeds our budget
            if context_tokens_used + chunk_token_count > available_context_tokens:
                logger.info(f"Context limit reached. Truncating remaining {len(retrieved_chunks) - chunks_used} chunks.")
                is_truncated = True
                break

            # Add chunk to our payload
            formatted_context_blocks.append(chunk_text)
            context_tokens_used += chunk_token_count
            chunks_used += 1

        # 3. Assemble the final formatted User Message string
        final_context_string = "\n".join(formatted_context_blocks)
        user_message_content = self.user_template.format(
            formatted_context=final_context_string if final_context_string else "No relevant context found.",
            question=question
        )

        # 4. Construct the strict Message Array (System -> [History] -> User)
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
        ]
        
        if chat_history:
            messages.extend(chat_history)
            
        messages.append(
            LLMMessage(role=MessageRole.USER, content=user_message_content)
        )

        # 5. Calculate final telemetry
        total_estimated_tokens = static_tokens + context_tokens_used
        
        logger.info(f"Prompt built successfully. Used {chunks_used}/{len(retrieved_chunks)} chunks. "
                    f"Estimated Tokens: {total_estimated_tokens}/{self.max_tokens}.")

        return PromptContext(
            messages=messages,
            estimated_tokens=total_estimated_tokens,
            chunks_used=chunks_used,
            total_chunks_provided=len(retrieved_chunks),
            is_truncated=is_truncated
        )