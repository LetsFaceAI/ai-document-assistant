# backend/app/api/v1/endpoints/embeddings.py
import logging
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/debug-image/{filename}", status_code=status.HTTP_200_OK)
async def get_debug_image(filename: str):
    try:
        clean_filename = os.path.basename(filename)
        
        # Dynamically resolve absolute path relative to this file's location 
        # (Points to backend/storage/debug reliably)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent  # Adjust based on your depth to 'backend/' root
        # Alternatively, if backend root is known:
        image_path = Path("storage/debug") / clean_filename
        
        # Fallback absolute check if relative path fails
        if not image_path.exists():
            # Try finding it relative to the current working directory or backend root
            alt_path = Path.cwd() / "storage" / "debug" / clean_filename
            if alt_path.exists():
                image_path = alt_path

        if not image_path.exists():
            logger.warning(f"Debug image not found at path: {image_path.absolute()}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Debug image not found at {image_path.absolute()}"
            )
            
        return FileResponse(image_path, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve debug image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to retrieve debug image."
        )