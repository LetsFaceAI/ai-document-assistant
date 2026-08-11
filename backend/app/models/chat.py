"""
File: backend/app/models/chat.py
Description: API request and response schemas for chat completions.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input question.")
    top_k: Optional[int] = Field(default=5, description="Number of context chunks to retrieve.")


class SourceCitation(BaseModel):
    filename: str
    page_number: int
    chunk_index: int


class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated AI response.")
    sources: List[SourceCitation] = Field(
        default=[], 
        description="List of document sources used to generate the answer."
    )
    tokens_used: Optional[int] = Field(
        default=None, 
        description="Estimated token count of the prompt context."
    )