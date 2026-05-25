from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.services import image_store
from backend.db import DATA_DIR

router = APIRouter()

TARGET_IMAGE_PATH = DATA_DIR / "H06_00.tiff"

@router.get("/images/target")
async def get_target_image():
    if not TARGET_IMAGE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Target image not found")
    return FileResponse(TARGET_IMAGE_PATH)

# image_id로 저장된 이미지 파일을 반환
@router.get("/images/{image_id}")
async def get_image(image_id: str, thumbnail: bool = False):
    try:
        path = image_store.get_image_path(image_id, thumbnail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(file_path)
