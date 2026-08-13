from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import chat
from app.api.v1.endpoints import document
from app.api.v1.endpoints import search
from app.api.v1.endpoints import embeddings

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(document.router)
api_router.include_router(search.router)
api_router.include_router(embeddings.router)