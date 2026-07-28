from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Sanitized stored filename")
    original_filename: str = Field(..., description="Original name provided by user")
    content_type: str = Field(..., description="Validated MIME type")
    size_bytes: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 cryptographic hash of the file")
    storage_path: str = Field(..., description="Path relative to storage directory")
    message: str = Field(default="File uploaded and validated successfully.")