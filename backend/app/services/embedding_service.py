import time
import logging
from sentence_transformers import SentenceTransformer
from app.models.domain import ChunkedDocument, EmbeddedDocument, EmbeddedChunk, EmbeddingSummary

logger = logging.getLogger(__name__)

class EmbeddingService:
    # Class-level variable to hold the model instance (Singleton)
    # This ensures we don't reload the 130MB model on every API request
    _model_instance = None 

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        
    @classmethod
    def _get_model(cls, model_name: str):
        """Lazy initialization: Loads the model only once when first requested."""
        if cls._model_instance is None:
            logger.info(f"Loading embedding model '{model_name}' into memory (CPU)...")
            # Force CPU execution as requested
            cls._model_instance = SentenceTransformer(model_name, device='cpu')
        return cls._model_instance

    def process(self, document: ChunkedDocument) -> EmbeddedDocument:
        start_time = time.time()
        
        # 1. Filter out empty or whitespace-only chunks and log warnings
        valid_chunks = []
        for chunk in document.chunks:
            if not chunk.text or not chunk.text.strip():
                logger.warning(
                    f"Skipping empty chunk (ID: {getattr(chunk, 'chunk_id', 'N/A')}, "
                    f"Index: {getattr(chunk, 'chunk_index', 'N/A')}). Check upstream extraction/chunking."
                )
                continue
            valid_chunks.append(chunk)

        chunks_count = len(valid_chunks)
        
        # Load the model
        model = self._get_model(self.model_name)
        dimension = model.get_sentence_embedding_dimension()
        
        logger.info("Embedding Started")
        logger.info(f"Chunks: {chunks_count}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dimension: {dimension}")

        # 2. Extract texts from VALID chunks only
        texts = [chunk.text for chunk in valid_chunks]
        
        # 3. Encode valid chunks (handle edge case where all chunks were empty)
        embeddings_array = (
            model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
            if texts else []
        )
        
        # 4. Map back to Domain Model
        embedded_chunks = []
        for i, (chunk, vector) in enumerate(zip(valid_chunks, embeddings_array)):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=str(getattr(chunk, 'chunk_id', i)), 
                    text=chunk.text,
                    embedding=vector.tolist(),
                    dimension=dimension,
                    model_name=self.model_name
                )
            )

        elapsed_time = time.time() - start_time
        
        logger.info("Embedding Completed")
        logger.info(f"Time: {elapsed_time:.1f} seconds")

        summary = EmbeddingSummary(
            model_name=self.model_name,
            dimension=dimension,
            total_chunks=chunks_count,
            processing_time_ms=elapsed_time * 1000
        )
        
        return EmbeddedDocument(
            file_hash=document.file_hash,
            metadata=document.metadata,
            chunks=embedded_chunks,
            embedding_stats=summary,
            chunking_stats=document.chunking_stats,
            processing_stats=document.processing_stats
        )