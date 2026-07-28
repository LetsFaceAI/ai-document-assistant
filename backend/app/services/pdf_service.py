import fitz  # PyMuPDF
import logging
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PDFMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    mod_date: str | None = None


class PDFExtractionResult(BaseModel):
    raw_text: str
    page_count: int
    total_characters: int
    metadata: PDFMetadata

class PdfService:
    """Service responsible for extracting raw text and metadata from PDF files."""

    def extract(self, file_path: Path) -> PDFExtractionResult:
        logger.info(f"PDF extraction started for file: '{file_path.name}'")

        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)
            logger.info(f"PDF opened successfully. File: '{file_path.name}', Page count: {page_count}")

            extracted_pages: list[str] = []
            
            # Iterate through each page and extract raw, uncleaned text
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                raw_page_text = page.get_text("text")
                extracted_pages.append(raw_page_text)

            doc_metadata = doc.metadata or {}
            doc.close()

            # Join pages into a single raw text payload
            full_raw_text = "\n\n".join(extracted_pages)
            total_characters = len(full_raw_text)

            logger.info(
                f"PDF extraction completed for file: '{file_path.name}'. "
                f"Pages: {page_count}, Total characters extracted: {total_characters}"
            )

            metadata_obj = PDFMetadata(
                title=doc_metadata.get("title") or None,
                author=doc_metadata.get("author") or None,
                subject=doc_metadata.get("subject") or None,
                keywords=doc_metadata.get("keywords") or None,
                creator=doc_metadata.get("creator") or None,
                producer=doc_metadata.get("producer") or None,
                creation_date=doc_metadata.get("creationDate") or None,
                mod_date=doc_metadata.get("modDate") or None,
            )

            return PDFExtractionResult(
                raw_text=full_raw_text,
                page_count=page_count,
                total_characters=total_characters,
                metadata=metadata_obj,
            )

        except Exception as e:
            logger.error(f"Failed to extract PDF content for file '{file_path.name}': {str(e)}", exc_info=True)
            raise ValueError(f"Unable to extract text from PDF document: {str(e)}")