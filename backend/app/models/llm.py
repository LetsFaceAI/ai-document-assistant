"""
File: backend/app/models/llm.py
Description: Domain models representing LLM chat messages, roles, and generated prompt contexts.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """Represents a single message in an LLM chat conversation."""
    role: MessageRole
    content: str


class PromptContext(BaseModel):
    """
    The structured output of PromptBuilderService.
    Contains the full message array ready for the LLM client, along with telemetry metrics.
    """
    messages: List[LLMMessage] = Field(
        ..., 
        description="The ordered list of messages (System, History, User) formatted for the LLM."
    )
    estimated_tokens: int = Field(
        ..., 
        description="Estimated total token count across all generated messages."
    )
    chunks_used: int = Field(
        ..., 
        description="Number of retrieved chunks included in the final context before token budget exhaustion."
    )
    total_chunks_provided: int = Field(
        ..., 
        description="Total number of candidate chunks supplied from vector retrieval."
    )
    is_truncated: bool = Field(
        default=False, 
        description="Indicates whether one or more chunks were omitted due to token limit constraints."
    )