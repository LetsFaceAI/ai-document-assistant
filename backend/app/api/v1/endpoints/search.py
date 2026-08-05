from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import logging

from app.models.domain import RetrievalResult
from app.services.retrieval_service import RetrievalService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStore
from app.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: int

def get_retrieval_service() -> RetrievalService:
    embedding_service = EmbeddingService()
    vector_store = ChromaVectorStore()
    return RetrievalService(
        embedding_service=embedding_service, 
        vector_store=vector_store
    )

def get_visualization_service() -> VisualizationService:
    return VisualizationService()

def generate_plot_background(
    viz_service: VisualizationService,
    vector_store: ChromaVectorStore,
    query_embedding: list[float],
    retrieved_embeddings: list[list[float]],
    question: str
):
    """Background task to generate and save the UMAP plot without blocking the API."""
    try:
        background_embeddings = vector_store.get_background_embeddings(limit=1000)
        viz_service.generate_umap_plot(
            query_embedding=query_embedding,
            retrieved_embeddings=retrieved_embeddings,
            background_embeddings=background_embeddings,
            query_text=question
        )
        logger.info(f"Successfully generated UMAP plot for query: '{question}'")
    except Exception as e:
        logger.error(f"Background UMAP plot generation failed: {str(e)}")


@router.post("/search", response_model=RetrievalResult)
async def search_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Execute standard retrieval flow
        result = await retrieval_service.retrieve(request.question)
        
        # 2. Extract the vectors from the services for visualization
        # We access the embedded query by directly calling the embedding_service attached to retrieval_service
        query_embedding = retrieval_service.embedding_service.embed_query(request.question)
        
        # Pull out the embeddings from the retrieved chunks (which are now hidden from JSON)
        # We use getattr() in case the result object structure differs slightly
        chunks = getattr(result, "retrieved_chunks", getattr(result, "chunks", []))
        retrieved_embeddings = [
            chunk.embedding for chunk in chunks if chunk.embedding is not None
        ]
        
        # 3. Offload plot generation to a background thread
        background_tasks.add_task(
            generate_plot_background,
            viz_service=viz_service,
            vector_store=retrieval_service.vector_store,
            query_embedding=query_embedding,
            retrieved_embeddings=retrieved_embeddings,
            question=request.question
        )
        
        # 4. Return standard JSON immediately
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")