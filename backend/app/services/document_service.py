import logging
import shutil
import tempfile
from app.core.config import settings
from fastapi import UploadFile
from app.services.pdf_service import PdfService
from pathlib import Path
from app.models.document import DocumentUploadResponse, ExtractionSummary, PDFMetadataSchema
from app.services.text_cleaning_service import TextCleaningService
from app.utils.file_utils import sanitize_filename, calculate_sha256

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

            # Extract PDF Content & Metadata
            extraction_result = self.pdf_service.extract(permanent_path)

            # 5. Clean Extracted Text
            cleaned_text = self.cleaner.clean_text(extraction_result.raw_text)

            # --------------------------------------------------------------------
            # --- DEBUG: Save raw vs. cleaned text files for manual inspection ---
            debug_dir = self.storage_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            raw_file_path = debug_dir / f"{hash_prefix}_RAW.txt"
            cleaned_file_path = debug_dir / f"{hash_prefix}_CLEANED.txt"

            raw_file_path.write_text(cleaned_text.original_text, encoding="utf-8")
        
            # FIX THE CRASH: Must access .cleaned_text property now
            cleaned_file_path.write_text(cleaned_text.cleaned_text, encoding="utf-8")

            logger.info(f"Saved debug text files to '{debug_dir}'")
            logger.info(f"Cleaning stats: {cleaned_text.processing_time_ms}ms, applied {cleaned_text.applied_rules_count} rules")
            # --------------------------------------------------------------------

            # Build Summary for Client Response (Omitting raw_text)
            summary = ExtractionSummary(
                page_count=extraction_result.page_count,
                total_characters=extraction_result.total_characters,
                cleaned_characters=cleaned_text.cleaned_char_count,
                applied_rules_count=cleaned_text.applied_rules_count,  
                processing_time_ms=cleaned_text.processing_time_ms,
                metadata=PDFMetadataSchema(**extraction_result.metadata.model_dump()),
            )

            return DocumentUploadResponse(
                filename=target_filename,
                original_filename=file.filename,
                content_type=file.content_type,
                size_bytes=total_bytes,
                file_hash=file_hash,
                storage_path=f"storage/uploads/{target_filename}",
                message="File uploaded and extracted successfully.",
                extraction_summary=summary,
            )

        finally:
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temporary file '{temp_file_path}': {cleanup_err}")