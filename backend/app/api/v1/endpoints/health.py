from fastapi import APIRouter
from app.models import HealthResponse
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="HEALTHY",
        app_name=settings.app_name,
        environment=settings.environment
    )