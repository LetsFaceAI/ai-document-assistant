from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.api import api_router
from app.core.logging_config import setup_logging
import logging

# Initialize logging
setup_logging()

# Initialize the FastAPI app
app = FastAPI(
    title= settings.app_name,
    description="API for uploading PDFs and querying them via RAG",
    version="0.1.0"
)

logger = logging.getLogger(__name__)

logger.info(f"Starting {settings.app_name} on port {settings.port}")

# Include the API router
app.include_router(api_router,prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Document Assistant API"}

@app.get("/info")
async def get_env():
    logger.info("Project Info")
    return {
        "app_name": settings.app_name,
        "debug_mode": settings.debug_mode,
    }


