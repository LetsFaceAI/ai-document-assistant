from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.document import DocumentUploadResponse
from app.services.document_service import DocumentService
import logging

router = APIRouter(prefix="/documents", tags=["Documents"])
document_service = DocumentService()

logger = logging.getLogger(__name__)

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    try:
        return await document_service.process_pdf_upload(file)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except FileExistsError as fee:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(fee)
        )
    except Exception as e:
        # LOG THE FULL TRACEBACK TO YOUR TERMINAL
        logger.exception("Failed to process document upload:")
        
        # Temporarily return str(e) to Swagger so you can see the error in the browser
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Error: {str(e)}"
        )