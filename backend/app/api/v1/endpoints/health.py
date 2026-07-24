from fastapi import APIRouter
from app.models import HealthResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="123",
        app_name="FastAPI-Backend",
        environment="development"
    )