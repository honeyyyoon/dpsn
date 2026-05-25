from pathlib import Path
from backend.db import get_conn, DATA_DIR


def cleanup_old_data(days: int = 3) -> None:
    with get_conn() as conn:
        old_jobs = conn.execute(
            "SELECT id, image_id, result_image_id FROM jobs WHERE created_at < datetime('now', ?)",
            (f'-{days} days',)
        ).fetchall()
        if not old_jobs:
            return

        job_ids = [j['id'] for j in old_jobs]
        image_ids: set[str] = set()
        for job in old_jobs:
            image_ids.add(job['image_id'])
            if job['result_image_id']:
                image_ids.add(job['result_image_id'])

        placeholders = ','.join('?' * len(job_ids))
        conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", job_ids)

        for image_id in image_ids:
            still_used = conn.execute(
                "SELECT 1 FROM jobs WHERE image_id = ? OR result_image_id = ? LIMIT 1",
                (image_id, image_id)
            ).fetchone()
            if still_used:
                continue
            row = conn.execute(
                "SELECT path, thumbnail_path FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if row:
                for rel_path in [row['path'], row['thumbnail_path']]:
                    if rel_path:
                        (DATA_DIR / rel_path).unlink(missing_ok=True)
                conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
