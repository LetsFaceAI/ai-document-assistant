from fastapi import FastAPI
from app.core.config import settings

# Initialize the FastAPI app
app = FastAPI(
    title= settings.app_name,
    description="API for uploading PDFs and querying them via RAG",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Document Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/env")
async def get_env():
    return {
        "app_name": settings.app_name,
        "port": settings.port,
        "debug_mode": settings.debug_mode,
        "openrouter_api_key": settings.openrouter_api_key
    }


