"""
File: backend/app/api/v1/endpoints/chat.py
Description: FastAPI route handlers for RAG chat functionality.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_services import ChatService

logger = logging.getLogger(__name__)
router = APIRouter()
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    try:
        result = await chat_service.process_user_message(
            user_message=request.message,
            top_k=request.top_k
        )
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            tokens_used=result["tokens_used"]
        )
        
    except Exception as e:
        logger.error(f"Failed to process chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to generate grounded AI response."
        )