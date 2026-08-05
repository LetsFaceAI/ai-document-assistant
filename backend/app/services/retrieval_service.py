import logging
import time
import anyio
from app.core.config import settings
from app.models.domain import RetrievalResult
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStore

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
    async def retrieve(self, question: str) -> RetrievalResult:
        start_time = time.time()
        logger.info("Retrieval started")
        
        # Offload CPU-heavy embedding model inference to threadpool
        query_vector = await anyio.to_thread.run_sync(
            self.embedding_service.embed_query, question
        )
        logger.info("Embedding generated")
        
        # Offload synchronous ChromaDB vector store query to threadpool
        retrieved_chunks = await anyio.to_thread.run_sync(
            self.vector_store.search, query_vector, settings.RETRIEVAL_TOP_K
        )
        
        # Filter and construct result
        valid_chunks = [
            chunk for chunk in retrieved_chunks 
            if chunk.score >= settings.MIN_SIMILARITY_SCORE
        ]
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        return RetrievalResult(
            question=question,
            top_k=settings.RETRIEVAL_TOP_K,
            retrieved_chunks=valid_chunks,
            retrieval_time_ms=retrieval_time_ms,
            embedding_model=self.embedding_service.model_name
        )