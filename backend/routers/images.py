from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from backend.services import image_store

router = APIRouter()

# image_id로 저장된 이미지 파일을 반환
@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    thumbnail: bool = False,
    download: bool = False,
    filename: str | None = None,
):
    try:
        path = image_store.get_image_path(image_id, thumbnail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    download_name = Path(filename).name if filename else file_path.name
    return FileResponse(
        file_path,
        filename=download_name if download else None,
    )
