import json
import uuid
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import traceback

from fastapi import BackgroundTasks
import torch

from ai.runtime.task import Task
from ai.runtime.worker import Worker
from backend.services import image_store
from backend.db import DATA_DIR, get_conn


def _available_inference_devices() -> list[str]:
    if torch.cuda.is_available():
        devices = [
            f"cuda:{gpu_index}"
            for gpu_index in (1, 2, 3)
            if torch.cuda.device_count() > gpu_index
        ]
        if devices:
            return devices
    return ["cpu"]


INFERENCE_DEVICES = _available_inference_devices()
_executor = ThreadPoolExecutor(max_workers=len(INFERENCE_DEVICES))
_device_queue: Queue[str] = Queue()
for _device in INFERENCE_DEVICES:
    _device_queue.put(_device)


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


# group_id로 그룹의 대표 정보(image_id, wsi_name) 조회
def get_group(group_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT image_id, wsi_name FROM jobs WHERE group_id = ? LIMIT 1",
            (group_id,)
        ).fetchone()
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
            VALUES (?, ?, ?, ?, ?, 'pending', 0, '작업이 생성되었습니다. 시작을 기다리는 중...')
            """,
            (job_id, group_id, model_id, image_id, wsi_name),
        )
    _executor.submit(run_job, job_id, model_id, image_id)
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
    device = _device_queue.get()
    try:
        job = get_job(job_id)
        if job and job.get("cancelled"):
            raise JobCancelledError(f"Job {job_id} was cancelled")

        update_job(job_id, "running", 0, f"{device}에서 실행 중...")

        def emit_event(status: str, progress: int, message: str):
            job = get_job(job_id)
            if job and job.get("cancelled"):
                raise JobCancelledError(f"Job {job_id} was cancelled")
            update_job(job_id, status, progress, f"[{device}] {message}")

        src_path = image_store.get_image_path(image_id)
        tgt_path = image_store.get_target_image_path()
        task = Task(
            src_img_path=Path(src_path),
            target_img_path=Path(tgt_path),
            result_path=DATA_DIR / "results",
            model_id=model_id
        )
        task_result = Worker().run(task, emit_event=emit_event, device=device)
        update_job(job_id, "running", 99, "Registering result image.")
        result_image_id = image_store.enroll_image(task_result.result_img_path)

        with get_conn() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = 'done', progress = 100, message = 'Job completed.',
                    result_image_id = ?, metrics = ?,
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
        message = str(e).strip() or type(e).__name__
        error_detail = traceback.format_exc()
        print("Error:", message)
        print(error_detail)
        with get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', message = ?, error_detail = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (message, error_detail, job_id),
            )
    finally:
        _device_queue.put(device)


# 그룹 전체를 취소(실행 중) 또는 삭제(완료·실패)
def delete_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return
    group_id = job["group_id"]
    if job["status"] in ("running", "pending"):
        with get_conn() as conn:
            conn.execute("UPDATE jobs SET cancelled = 1 WHERE group_id = ?", (group_id,))
    else:
        with get_conn() as conn:
            conn.execute("DELETE FROM jobs WHERE group_id = ?", (group_id,))
