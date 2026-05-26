import uuid
from pathlib import Path

from fastapi import UploadFile
import openslide
from PIL import Image
import tifffile

from backend.db import get_conn, DATA_DIR

TARGET_IMAGE_ID = "target"
TARGET_DIR = DATA_DIR / "target"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def _relative_to_data_dir(path: Path) -> Path:
    return path.resolve().relative_to(DATA_DIR.resolve())


def _find_target_image_path() -> Path:
    candidates = sorted(
        path
        for path in TARGET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No target image found in: {TARGET_DIR}")

    return candidates[0]


def ensure_target_image() -> str:
    target_path = _find_target_image_path()
    rel_path = _relative_to_data_dir(target_path)

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM images
            WHERE id = ?
            """,
            (TARGET_IMAGE_ID,),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO images (
                    id,
                    path,
                    original_filename,
                    has_thumbnail
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    TARGET_IMAGE_ID,
                    str(rel_path),
                    target_path.name,
                    0,
                ),
            )
            needs_thumbnail = True
        else:
            thumbnail_path = row["thumbnail_path"]
            thumbnail_file = DATA_DIR / thumbnail_path if thumbnail_path else None
            needs_thumbnail = (
                row["path"] != str(rel_path)
                or int(row["has_thumbnail"]) == 0
                or thumbnail_path is None
                or not thumbnail_file.is_file()
                or target_path.stat().st_mtime > thumbnail_file.stat().st_mtime
            )
            if row["path"] != str(rel_path):
                conn.execute(
                    """
                    UPDATE images
                    SET path = ?,
                        original_filename = ?,
                        has_thumbnail = ?,
                        thumbnail_path = NULL
                    WHERE id = ?
                    """,
                    (
                        str(rel_path),
                        target_path.name,
                        0,
                        TARGET_IMAGE_ID,
                    ),
                )

    if needs_thumbnail:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM images
                WHERE id = ?
                """,
                (TARGET_IMAGE_ID,),
            ).fetchone()
        make_thumbnail(row)

    return TARGET_IMAGE_ID


def get_target_image_path(thumbnail: bool = False) -> str:
    ensure_target_image()
    return get_image_path(TARGET_IMAGE_ID, thumbnail=thumbnail)


# 업로드된 파일을 data/uploads에 저장하고 image_id를 반환
async def save_image(file: UploadFile, image_id: str | None = None) -> str:
    if not image_id:
        image_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix if file.filename else ""
    path = Path("uploads") / f"{image_id}{ext}"

    with open(DATA_DIR / path, "wb") as f:
        f.write(await file.read())
    
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO images (
                id,
                path,
                has_thumbnail
            )
            VALUES (?, ?, ?)
            """,
            (
                image_id,
                str(path),
                0,
            ),
        )
    return image_id

def make_thumbnail(row: dict) -> str:
    image_id = row['id']
    path: Path = DATA_DIR / row['path']
    ext = path.suffix.lower()
    try:
        with Image.open(path) as img:
            img.thumbnail((1024, 1024))
            thumb = img.copy()
    except:
        if ext in {".tif", ".tiff"}:
            with tifffile.TiffFile(path) as slide:
                img = Image.fromarray(slide.asarray())
                img.thumbnail((1024, 1024))
                thumb = img.copy()
        else:
            with openslide.OpenSlide(path) as slide:
                thumb = slide.get_thumbnail((1024, 1024))
    
    if thumb is None:
        raise ValueError("Thumbnail is None")
    
    thumbnail_path = Path("thumbnails") / f"{image_id}.png"
    thumb.save(DATA_DIR / thumbnail_path)

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE images
            SET has_thumbnail = ?,
                thumbnail_path = ?
            WHERE id = ?
            """,
            (
                1,
                str(thumbnail_path),
                image_id,
            ),
        )

    return str(thumbnail_path)

# image_id로 저장된 파일 경로를 반환, 없으면 KeyError 발생
def get_image_path(image_id: str, thumbnail: bool = False) -> str:
    
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM images
            WHERE id = ?
            """,
            (image_id,),
        ).fetchone()
    
    if row is None:
        raise KeyError(f"Image not found for image_id: {image_id}")
    
    if not thumbnail:
        return DATA_DIR / row["path"]
    
    if int(row["has_thumbnail"]) == 0:
        thumbnail_path = make_thumbnail(row)
    else:
        thumbnail_path = row['thumbnail_path']
    
    return str(DATA_DIR / thumbnail_path)

def enroll_image(path: Path) -> str:
    if not path.exists():
        raise ValueError("Path doesn't exist!")
    
    image_id = str(uuid.uuid4())
    new_path = path.with_stem(image_id)
    path = path.rename(new_path)

    file_path = Path("results") / path.name

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO images (
                id,
                path,
                has_thumbnail
            )
            VALUES (?, ?, ?)
            """,
            (
                image_id,
                str(file_path),
                0,
            ),
        )

    return image_id
