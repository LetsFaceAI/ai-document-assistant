from pydantic import BaseModel, Field
from app.models.chunk import ChunkingSummary

class PDFMetadataSchema(BaseModel):
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    mod_date: str | None = None


class ExtractionSummary(BaseModel):
    page_count: int
    total_characters: int
    cleaned_characters: int
    metadata: PDFMetadataSchema
    # Optional additions:
    applied_rules_count: int | None = None
    processing_time_ms: float | None = None

class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Sanitized stored filename")
    original_filename: str = Field(..., description="Original name provided by user")
    content_type: str = Field(..., description="Validated MIME type")
    size_bytes: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 cryptographic hash of the file")
    storage_path: str = Field(..., description="Path relative to storage directory")
    message: str = Field(default="File uploaded and validated successfully.")
    extraction_summary: ExtractionSummary
    chunking_summary: ChunkingSummary
