from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_services import ChatService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
chat_service = ChatService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # The router does not know we are using NVIDIA or what a prompt is.
        answer = await chat_service.process_user_message(request.message)
        
        return ChatResponse(response=answer)
        
    except Exception as e:
        logger.error(f"Failed to process chat: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve response from AI service."
        )