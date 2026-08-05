import logging
import shutil
import tempfile
import json
from app.core.config import settings
from fastapi import UploadFile
from app.services.pdf_service import PdfService
from pathlib import Path
from app.models.document import DocumentUploadResponse, ExtractionSummary, PDFMetadataSchema
from app.services.text_cleaning_service import TextCleaningService
from app.utils.file_utils import sanitize_filename, calculate_sha256
from app.services.chunking_service import ChunkingService
from app.models.domain import ProcessedDocument
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStore, ChromaVectorStore

logger = logging.getLogger(__name__)

class DocumentService:
    ALLOWED_MIME_TYPES = {"application/pdf"}
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB Limit

    def __init__(self):
        # Determine backend root path
        self.storage_dir = settings.STORAGE_DIR
        self.temp_dir = self.storage_dir / "temp"
        self.uploads_dir = self.storage_dir / "uploads"

        # Ensure storage directories exist at startup
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_service = PdfService()
        self.cleaner = TextCleaningService()
        self.chunker = ChunkingService()  # Configured via settings

        # 2. Instantiate EmbeddingService using settings
        self.embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=settings.EMBEDDING_BATCH_SIZE
        )

        # Inject the concrete implementation behind the Protocol
        self.vector_store: VectorStore = ChromaVectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            collection_name=settings.VECTOR_COLLECTION
        )

    async def process_pdf_upload(self, file: UploadFile) -> DocumentUploadResponse:
        # FIX 1: Catch None values immediately to prevent 500 errors
        if not file.filename or not file.content_type:
            raise ValueError("File name and content type must be provided.")

        logger.info(f"Initiating upload processing for file: '{file.filename}'")

        if file.content_type not in self.ALLOWED_MIME_TYPES:
            raise ValueError(f"Invalid file type '{file.content_type}'. Only PDF documents are allowed.")

        if file.size and file.size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds the maximum limit of 10MB.")

        clean_filename = sanitize_filename(file.filename)

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=False, suffix=".pdf") as tmp:
                temp_file_path = Path(tmp.name)
                total_bytes = 0
                
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > self.MAX_FILE_SIZE_BYTES:
                        raise ValueError("File size exceeds the maximum limit of 10MB.")
                    tmp.write(chunk)

            file_hash = calculate_sha256(temp_file_path)
            
            # FIX 2: Instant duplicate check using glob instead of re-hashing the whole directory
            hash_prefix = file_hash[:10]
            existing_matches = list(self.uploads_dir.glob(f"{hash_prefix}_*"))
            if existing_matches:
                logger.warning(f"Duplicate file detected matching hash prefix {hash_prefix}")
                raise FileExistsError(f"Duplicate file detected. This document already exists as '{existing_matches[0].name}'.")

            target_filename = f"{hash_prefix}_{clean_filename}"
            permanent_path = self.uploads_dir / target_filename
            
            shutil.move(str(temp_file_path), str(permanent_path))
            logger.info(f"Moved file to permanent storage: {permanent_path}")

            # 1. Extract PDF Content & Metadata
            extraction_result = self.pdf_service.extract(permanent_path)

            # 2. Clean Extracted Text
            cleaning_result = self.cleaner.clean_text(extraction_result.raw_text)

            # Dump metadata to dict and explicitly set the filename
            metadata_dict = extraction_result.metadata.model_dump()
            metadata_dict["filename"] = clean_filename

            # 3. Build Extraction Summary early for the domain model
            summary = ExtractionSummary(
                page_count=extraction_result.page_count,
                total_characters=extraction_result.total_characters,
                cleaned_characters=cleaning_result.cleaned_char_count,
                applied_rules_count=cleaning_result.applied_rules_count,  
                processing_time_ms=cleaning_result.processing_time_ms,
                metadata=PDFMetadataSchema(**metadata_dict),
            )

            # 4. STATE 1: Create ProcessedDocument
            processed_doc = ProcessedDocument(
                file_hash=file_hash,
                metadata=summary.metadata,
                raw_text=extraction_result.raw_text,
                clean_text=cleaning_result.cleaned_text,
                processing_stats=summary
            )

            # 5. STATE TRANSITION 1: ProcessedDocument -> ChunkedDocument
            chunked_doc = self.chunker.process(processed_doc)

            # 6. STATE TRANSITION 2: ChunkedDocument -> EmbeddedDocument (NEW!)
            embedded_doc = self.embedding_service.process(chunked_doc)

            # STATE TRANSITION 3: EmbeddedDocument -> VectorizedDocument
            vectorized_doc = self.vector_store.store(embedded_doc)

            # --------------------------------------------------------------------
            # --- DEBUG: Save Raw vs. Cleaned vs. Chunks using Domain Models   ---
            # --------------------------------------------------------------------
            debug_dir = self.storage_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            hash_prefix = file_hash[:10] if 'file_hash' in locals() else "debug_doc"

            # 1. Save Raw Text
            raw_path = debug_dir / f"{hash_prefix}_1_RAW.txt"
            raw_path.write_text(processed_doc.raw_text, encoding="utf-8")

            # 2. Save Cleaned Text
            cleaned_path = debug_dir / f"{hash_prefix}_2_CLEANED.txt"
            cleaned_path.write_text(processed_doc.clean_text, encoding="utf-8")

            # 3. Save Human-Readable Chunks
            chunks_txt_path = debug_dir / f"{hash_prefix}_3_CHUNKS.txt"
            
            readable_chunks = []
            for c in chunked_doc.chunks:
                header = (
                    f"=== CHUNK {c.chunk_index} "
                    f"| Length: {c.character_count} chars "
                    f"| Est. Tokens: ~{c.estimated_token_count} "
                    f"| Offsets: [{c.start_offset}:{c.end_offset}] ==="
                )
                readable_chunks.append(f"{header}\n{c.text}\n")

            chunks_txt_path.write_text("\n" + ("=" * 80) + "\n\n".join(readable_chunks), encoding="utf-8")

            # 4. Save Raw JSON Chunks
            chunks_json_path = debug_dir / f"{hash_prefix}_3_CHUNKS.json"
            chunks_data = [chunk.model_dump() for chunk in chunked_doc.chunks]
            chunks_json_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

            logger.info(f"Saved debug files (RAW, CLEANED, CHUNKS) to '{debug_dir}'")

            # Save Embeddings Metadata & Vectors for inspection
            embeddings_json_path = debug_dir / f"{hash_prefix}_4_EMBEDDINGS.json"
            embeddings_data = [chunk.model_dump() for chunk in embedded_doc.chunks]
            embeddings_json_path.write_text(json.dumps(embeddings_data, indent=2), encoding="utf-8")

            # NEW: Save Vectorized Document Summary
            vectorized_json_path = debug_dir / f"{hash_prefix}_5_VECTORIZED.json"
            vectorized_json_path.write_text(
                vectorized_doc.model_dump_json(indent=2), 
                encoding="utf-8"
            )
            # --------------------------------------------------------------------

            # Return response using the final pipeline state
            return DocumentUploadResponse(
                filename=target_filename,
                original_filename=file.filename,
                content_type=file.content_type,
                size_bytes=total_bytes,
                file_hash=chunked_doc.file_hash,
                storage_path=f"storage/uploads/{target_filename}",
                message="File uploaded, extracted, cleaned, and chunked successfully.",
                extraction_summary=chunked_doc.processing_stats,
                chunking_summary=chunked_doc.chunking_stats
            )

        finally:
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temporary file '{temp_file_path}': {cleanup_err}")