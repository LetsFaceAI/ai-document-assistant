from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}