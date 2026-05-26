import json
import uuid
import dataclasses
from pathlib import Path
import traceback

from fastapi import BackgroundTasks

from ai.runtime.task import Task
from ai.runtime.worker import Worker
from backend.services import image_store
from backend.db import DATA_DIR, get_conn

_worker = Worker()


class JobCancelledError(Exception):
    pass


# SQLite Row 객체를 일반 dict로 변환
def _row_to_dict(row) -> dict | None:
    return dict(row) if row else None


# job_id로 DB에서 단건 조회
def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


# 전체 job 목록을 최신순으로 조회
def get_all_jobs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# DB에 pending 상태로 job 삽입 후 백그라운드 작업으로 run_job 예약, job_id 반환
def create_job(model_id: int, image_id: str, background_tasks: BackgroundTasks,
               group_id: str, wsi_name: str) -> str:
    job_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, group_id, model_id, image_id, wsi_name, status, progress, message)
            VALUES (?, ?, ?, ?, ?, 'pending', 0, 'Job created. Waiting to start.')
            """,
            (job_id, group_id, model_id, image_id, wsi_name),
        )
    background_tasks.add_task(run_job, job_id, model_id, image_id)
    return job_id


# job 상태·진행률·메시지를 DB에 업데이트
def update_job(job_id: str, status: str, progress: int, message: str):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs SET status = ?, progress = ?, message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, progress, message, job_id),
        )


# AI 모델을 실행하고 결과(이미지·metrics)를 DB에 저장; 취소·오류 발생 시 상태 업데이트
def run_job(job_id: str, model_id: int, image_id: str):
    update_job(job_id, "running", 0, "Running...")

    def emit_event(status: str, progress: int, message: str):
        job = get_job(job_id)
        if job and job.get("cancelled"):
            raise JobCancelledError(f"Job {job_id} was cancelled")
        update_job(job_id, status, progress, message)

    try:
        src_path = image_store.get_image_path(image_id)
        tgt_path = DATA_DIR / "scc_01_cs2_level3.jpg"
        task = Task(
            src_img_path=Path(src_path),
            target_img_path=Path(tgt_path),
            result_path=DATA_DIR / "results",
            model_id=model_id
        )
        task_result = _worker.run(task, emit_event=emit_event)
        result_image_id = image_store.enroll_image(task_result.result_img_path)

        with get_conn() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = 'done', result_image_id = ?, metrics = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (result_image_id, json.dumps(dataclasses.asdict(task_result.metrics)), job_id),
            )
    except JobCancelledError:
        with get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,),
            )
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        with get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (str(e), job_id),
            )


# 실행 중이면 취소 플래그 설정, 완료·실패면 DB에서 삭제
def delete_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return
    if job["status"] in ("running", "pending"):
        with get_conn() as conn:
            conn.execute("UPDATE jobs SET cancelled = 1 WHERE id = ?", (job_id,))
    else:
        with get_conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
