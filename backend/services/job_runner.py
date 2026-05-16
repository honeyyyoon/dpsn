import uuid
import dataclasses
from pathlib import Path
import traceback

from fastapi import BackgroundTasks

from ai.runtime.task import Task
from ai.runtime.worker import Worker
from backend.services import image_store
from backend.db import DATA_DIR

jobs: dict = {}
_worker = Worker()


class JobCancelledError(Exception):
    pass


# job 상태를 running으로 전환하고 Worker를 실행해 결과를 jobs dict에 저장
def run_job(job_id: str, model_id: int, image_id: str):
    jobs[job_id]["status"] = "running"

    def emit_event(status: str, progress: int, message: str):
        if jobs.get(job_id, {}).get("cancelled"):
            raise JobCancelledError(f"Job {job_id} was cancelled")
        update_job(job_id, status=status, progress=progress, message=message)

    try:
        src_path = image_store.get_image_path(image_id)
        tgt_path = DATA_DIR / "H06_00.tiff"
        task = Task(
            src_img_path=Path(src_path),
            target_img_path=Path(tgt_path),
            result_path=DATA_DIR / "results",
            model_id=model_id
        )
        task_result = _worker.run(task, emit_event=emit_event)

        result_image_id = image_store.enroll_image(task_result.result_img_path)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result_image_id"] = result_image_id
        jobs[job_id]["result"] = dataclasses.asdict(task_result.metrics)
    except JobCancelledError:
        jobs[job_id]["status"] = "cancelled"
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(e)


# job을 생성하고 background task로 run_job을 스케줄링
def create_job(model_id: int, image_id: str, background_tasks: BackgroundTasks) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "model_id": model_id,
        "image_id": image_id,
        "progress": 0,
        "cancelled": False,
        "message": "Job created. Waiting to start.",
        "result": None
    }
    background_tasks.add_task(run_job, job_id, model_id, image_id)
    return job_id


def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return
    if job["status"] in ("running", "pending"):
        job["cancelled"] = True
    else:
        jobs.pop(job_id)


def update_job(
    job_id: str,
    status: str,
    progress: int,
    message: str,
):
    jobs[job_id].update({
        "status": status,
        "progress": progress,
        "message": message,
    })
