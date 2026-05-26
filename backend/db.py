import sqlite3
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DPSN_DATA_DIR", "/mnt/Disk1/dpsn_data"))
DB_PATH = DATA_DIR / "app.db"

Path(DATA_DIR / "results").mkdir(parents=True, exist_ok=True)
Path(DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
Path(DATA_DIR / "target").mkdir(parents=True, exist_ok=True)
Path(DATA_DIR / "thumbnails").mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            original_filename TEXT,
            has_thumbnail INTEGER NOT NULL DEFAULT 0,
            thumbnail_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        try:
            conn.execute("ALTER TABLE images ADD COLUMN original_filename TEXT")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            image_id TEXT NOT NULL,
            wsi_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            progress REAL NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            error_detail TEXT NOT NULL DEFAULT '',
            cancelled INTEGER NOT NULL DEFAULT 0,
            result_image_id TEXT,
            metrics TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        """)
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN error_detail TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
