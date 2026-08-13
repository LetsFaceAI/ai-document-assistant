from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from app.models.chunk import DocumentChunk, ChunkingSummary
from app.models.document import PDFMetadataSchema, ExtractionSummary

# 1. State: After Text Extraction & Cleaning
class ProcessedDocument(BaseModel):
    file_hash: str
    metadata: PDFMetadataSchema
    raw_text: str
    clean_text: str
    processing_stats: ExtractionSummary

# 2. State: After Chunking
class ChunkedDocument(BaseModel):
    file_hash: str
    metadata: PDFMetadataSchema
    # Notice we drop raw_text and clean_text to save memory downstream
    chunks: List[DocumentChunk]
    chunking_stats: ChunkingSummary
    
    # Optional: Keep previous stats if needed for the final API response
    processing_stats: ExtractionSummary 

class EmbeddedChunk(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    dimension: int
    model_name: str

class EmbeddingSummary(BaseModel):
    model_name: str
    dimension: int
    total_chunks: int
    processing_time_ms: float

class EmbeddedDocument(BaseModel):
    file_hash: str
    metadata: PDFMetadataSchema
    chunks: List[EmbeddedChunk]
    embedding_stats: EmbeddingSummary
    
    # Carry forward previous stats
    chunking_stats: ChunkingSummary
    processing_stats: ExtractionSummary

class StorageSummary(BaseModel):
    provider: str
    collection_name: str
    total_chunks_stored: int
    persisted: bool

class VectorizedDocument(BaseModel):
    document_id: str
    collection_name: str
    stored_chunks: int
    storage_summary: StorageSummary

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    metadata: Dict[str, Any]
    # Store the vector for UMAP visualization
    embedding: Optional[List[float]] = Field(default=None, exclude=True)

class RetrievalResult(BaseModel):
    question: str
    top_k: int
    retrieved_chunks: List[RetrievedChunk]
    retrieval_time_ms: float
    embedding_model: str
    debug_image: Optional[str] = None