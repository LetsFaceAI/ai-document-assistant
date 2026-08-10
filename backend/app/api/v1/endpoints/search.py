"""
File: backend/app/api/v1/endpoints/search.py
Description: The primary search endpoint. Orchestrates Query Expansion, 
Batch Embedding, Vector Search, and Reciprocal Rank Fusion (RRF).
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from app.clients.llm_client import NvidiaClient

from app.core.config import settings
from app.models.domain import RetrievalResult, RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStore
from app.services.query_expansion_service import QueryExpansionService
from app.services.fusion_service import FusionService
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
    question: str
):
    """Generates the UMAP evaluation plot in storage/debug/ without slowing down the API."""
    try:
        background_embeddings = vector_store.get_background_embeddings(limit=1000)
        viz_service.generate_umap_plot(
            query_embedding=query_embedding,
            retrieved_embeddings=retrieved_embeddings,
            background_embeddings=background_embeddings,
            query_text=question
        )
    except Exception as e:
        logger.error(f"Background UMAP plot generation failed: {str(e)}")

@router.post("/search", response_model=RetrievalResult)
async def search_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    # Injecting services
    embedding_service: EmbeddingService = Depends(lambda: EmbeddingService()),
    vector_store: ChromaVectorStore = Depends(lambda: ChromaVectorStore()),
    viz_service: VisualizationService = Depends(lambda: VisualizationService()),
    nvidia_client: NvidiaClient = Depends(lambda: NvidiaClient())
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    try:
        # --- LOGGING STEP 1: Original Query ---
        logger.info(f"[RAG Pipeline] Step 1: Original Query -> '{request.question}'")

        # --- STEP 2: Query Expansion ---
        # Note: Pass your LLM client instance here
        expansion_service = QueryExpansionService(
            llm_client=nvidia_client,
            model_name=settings.openrouter_base_model,
            query_count=settings.QUERY_EXPANSION_COUNT
        )
        expanded_queries = await expansion_service.generate_expanded_queries(request.question)
        
        # --- LOGGING STEP 2: Generated Search Queries ---
        logger.info(f"[RAG Pipeline] Step 2: Generated {len(expanded_queries)} Search Queries -> {expanded_queries}")

        # --- STEP 3: Batch Embeddings ---
        query_embeddings = embedding_service.embed_queries(expanded_queries)
        
        # --- LOGGING STEP 3: Embedded Queries ---
        logger.info(f"[RAG Pipeline] Step 3: Embedded {len(query_embeddings)} Queries.")

        # --- STEP 4: Batch Vector Search ---
        candidate_results = vector_store.search_batch(
            query_embeddings=query_embeddings, 
            top_k=request.top_k
        )
        
        total_candidates = sum(len(sublist) for sublist in candidate_results)
        
        # --- LOGGING STEP 4 & 5: Performed Batch Search & Retrieved Candidates ---
        logger.info(f"[RAG Pipeline] Step 4 & 5: Performed Batch Search. Retrieved {total_candidates} total candidate chunks.")

        # --- STEP 5: Reciprocal Rank Fusion ---
        fusion_service = FusionService(rrf_k=settings.RRF_K)
        final_ranked_chunks = fusion_service.fuse_results(
            multiple_result_lists=candidate_results, 
            top_k=request.top_k
        )
        
        # --- LOGGING STEP 6 & 7: RRF Applied & Final Top K Returned ---
        logger.info(f"[RAG Pipeline] Step 6 & 7: RRF Applied. Returning Top {len(final_ranked_chunks)} final ranked chunks.")

        # --- STEP 6: Asynchronous UMAP Plot Generation ---
        # We pass the original query vector (index 0) and the final chunk vectors to UMAP
        original_query_vec = query_embeddings[0]
        final_chunk_vectors = [c.embedding for c in final_ranked_chunks if c.embedding is not None]

        background_tasks.add_task(
            generate_plot_background,
            viz_service=viz_service,
            vector_store=vector_store,
            query_embedding=original_query_vec,
            retrieved_embeddings=final_chunk_vectors,
            question=request.question
        )

        # Calculate latency in milliseconds
        retrieval_time_ms = round((time.time() - start_time) * 1000, 2)

        # --- STEP 7: Return Response ---
        return RetrievalResult(
            question=request.question,
            top_k=request.top_k,
            retrieved_chunks=final_ranked_chunks,
            retrieval_time_ms=retrieval_time_ms,
            embedding_model=getattr(embedding_service, "model_name", "BAAI/bge-small-en-v1.5")
        )

    except Exception as e:
        logger.error(f"Search endpoint failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")