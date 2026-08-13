"""
File: backend/app/api/v1/endpoints/search.py
Description: Primary search endpoint. Delegates RAG retrieval pipeline to RetrievalService 
and schedules background UMAP plotting.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from app.clients.llm_client import NvidiaClient
from app.models.domain import RetrievalResult
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStore
from app.services.retrieval_service import RetrievalService
from app.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


# --- Background Task for UMAP Plotting ---
def generate_plot_background(
    viz_service: VisualizationService,
    vector_store: ChromaVectorStore,
    query_embedding: list[float],
    retrieved_embeddings: list[list[float]],
    question: str,
    filename: str,  # <--- NEW PARAMETER
):
    """Generates the UMAP evaluation plot in storage/debug/ without slowing down the API."""
    try:
        background_embeddings = vector_store.get_background_embeddings(limit=1000)
        viz_service.generate_umap_plot(
            query_embedding=query_embedding,
            retrieved_embeddings=retrieved_embeddings,
            background_embeddings=background_embeddings,
            query_text=question,
            filename=filename,  # <--- PASS FILENAME TO SERVICE
        )
    except Exception as e:
        logger.error(f"Background UMAP plot generation failed: {str(e)}")


@router.post("/search", response_model=RetrievalResult)
async def search_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    # Injected Dependencies
    embedding_service: EmbeddingService = Depends(lambda: EmbeddingService()),
    vector_store: ChromaVectorStore = Depends(lambda: ChromaVectorStore()),
    viz_service: VisualizationService = Depends(lambda: VisualizationService()),
    nvidia_client: NvidiaClient = Depends(lambda: NvidiaClient()),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Initialize RetrievalService with injected dependencies
        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_client=nvidia_client,
        )

        # Execute full multi-query + RRF pipeline
        retrieval_result, original_query_vec = await retrieval_service.retrieve(
            question=request.question, top_k=request.top_k
        )

        # Collect chunk vectors for background UMAP plot
        final_chunk_vectors = [
            c.embedding
            for c in retrieval_result.retrieved_chunks
            if getattr(c, "embedding", None) is not None
        ]

        # 1. Pre-generate the filename for the UMAP image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"umap_query_{timestamp}.png"

        # 2. Schedule background visualization with pre-generated filename
        background_tasks.add_task(
            generate_plot_background,
            viz_service=viz_service,
            vector_store=vector_store,
            query_embedding=original_query_vec,
            retrieved_embeddings=final_chunk_vectors,
            question=request.question,
            filename=debug_filename,  # <--- PASS FILENAME HERE
        )

        # 3. Attach the filename to the response model so React receives it
        retrieval_result.debug_image = debug_filename

        return retrieval_result

    except Exception as e:
        logger.error(f"Search endpoint failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")