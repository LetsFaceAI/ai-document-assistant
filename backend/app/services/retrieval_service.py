import logging
import time
import anyio
from typing import Optional

from app.clients.llm_client import NvidiaClient
from app.core.config import settings
from app.models.domain import RetrievalResult
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStore
from app.services.query_expansion_service import QueryExpansionService
from app.services.fusion_service import FusionService

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        llm_client: Optional[NvidiaClient] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = llm_client or NvidiaClient()
        self.fusion_service = FusionService(rrf_k=settings.RRF_K)

    async def retrieve(
        self, question: str, top_k: Optional[int] = None
    ) -> tuple[RetrievalResult, list[float]]:
        """
        Executes the full RAG Retrieval Pipeline:
        1. Query Expansion (via LLM)
        2. Batch Query Embedding
        3. Batch Vector Search
        4. Reciprocal Rank Fusion (RRF)
        
        Returns:
            tuple[RetrievalResult, list[float]]: The domain retrieval result and the 
                                                original query embedding vector (for UMAP plotting).
        """
        effective_top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        start_time = time.time()

        # Step 1: Query Expansion
        logger.info(f"[RAG Pipeline] Step 1: Original Query -> '{question}'")
        expansion_service = QueryExpansionService(
            llm_client=self.llm_client,
            model_name=settings.openrouter_base_model,
            query_count=settings.QUERY_EXPANSION_COUNT,
        )
        expanded_queries = await expansion_service.generate_expanded_queries(question)
        logger.info(
            f"[RAG Pipeline] Step 2: Generated {len(expanded_queries)} Search Queries -> {expanded_queries}"
        )

        # Step 2: Batch Embeddings (Offloaded CPU inference to threadpool)
        query_embeddings = await anyio.to_thread.run_sync(
            self.embedding_service.embed_queries, expanded_queries
        )
        logger.info(f"[RAG Pipeline] Step 3: Embedded {len(query_embeddings)} Queries.")

        # Step 3: Batch Vector Search (Offloaded sync vector DB call to threadpool)
        candidate_results = await anyio.to_thread.run_sync(
            self.vector_store.search_batch, query_embeddings, effective_top_k
        )
        total_candidates = sum(len(sublist) for sublist in candidate_results)
        logger.info(
            f"[RAG Pipeline] Step 4 & 5: Batch Search completed. Retrieved {total_candidates} candidates."
        )

        # Step 4: Reciprocal Rank Fusion
        final_ranked_chunks = self.fusion_service.fuse_results(
            multiple_result_lists=candidate_results, top_k=effective_top_k
        )
        logger.info(
            f"[RAG Pipeline] Step 6 & 7: RRF Applied. Returning Top {len(final_ranked_chunks)} final chunks."
        )

        retrieval_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Primary query vector (index 0 is the original non-expanded prompt)
        original_query_vec = query_embeddings[0] if query_embeddings else []

        result = RetrievalResult(
            question=question,
            top_k=effective_top_k,
            retrieved_chunks=final_ranked_chunks,
            retrieval_time_ms=retrieval_time_ms,
            embedding_model=getattr(
                self.embedding_service, "model_name", "BAAI/bge-small-en-v1.5"
            ),
        )

        return result, original_query_vec