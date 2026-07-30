from pydantic import BaseModel, Field
import uuid

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: int
    text: str
    character_count: int
    estimated_token_count: int
    start_offset: int
    end_offset: int

class ChunkingSummary(BaseModel):
    total_chunks: int
    avg_chunk_size: float
    largest_chunk_size: int
    smallest_chunk_size: int
    total_estimated_tokens: int