import json
from datetime import datetime
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from backend.schemas import JobResponse, JobStatusResponse, JobResultResponse, JobGroupResponse, JobListItem, AddModelsRequest
from backend.services import job_runner, image_store

router = APIRouter()


# 전체 job을 group_id 기준으로 묶어 JobGroupResponse 목록으로 반환
@router.get("/jobs", response_model=list[JobGroupResponse])
async def list_jobs():
    rows = job_runner.get_all_jobs()
    groups: dict[str, JobGroupResponse] = {}
    for row in rows:
        gid = row["group_id"]
        metrics = None
        if row["metrics"]:
            try:
                metrics = json.loads(row["metrics"])
            except Exception:
                pass
        elapsed = None
        if row["status"] == "done":
            try:
                elapsed = (
                    datetime.fromisoformat(row["updated_at"]) - datetime.fromisoformat(row["created_at"])
                ).total_seconds()
            except Exception:
                pass
        item = JobListItem(
            id=row["id"],
            model_id=row["model_id"],
            status=row["status"],
            progress=row["progress"],
            message=row.get("message"),
            error_detail=row.get("error_detail"),
            result_image_id=row.get("result_image_id"),
            metrics=metrics,
            elapsed_seconds=elapsed,
        )
        if gid not in groups:
            groups[gid] = JobGroupResponse(
                group_id=gid,
                wsi_name=row["wsi_name"],
                image_id=row["image_id"],
                created_at=row["created_at"],
                jobs=[],
            )
        groups[gid].jobs.append(item)
    return list(groups.values())


# 이미지와 모델 ID 목록을 받아 각 모델에 대한 job을 생성하고 job_id 목록 반환
@router.post("/jobs", response_model=list[JobResponse])
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    model_ids: str = Form(...),
    wsi_name: str = Form(default=""),
):
    tokens = [x.strip() for x in model_ids.split(",")]
    filtered_tokens = [x for x in tokens if x]
    if not filtered_tokens:
        raise HTTPException(status_code=400, detail="model_ids must contain at least one valid integer")
    try:
        model_id_list = [int(x) for x in filtered_tokens]
    except ValueError:
        raise HTTPException(status_code=400, detail="model_ids must be a comma-separated list of integers")

    image_id = await image_store.save_image(image)
    derived_wsi = wsi_name or (image.filename.rsplit(".", 1)[0] if image.filename else "unknown")

    group_id = None
    responses = []
    for mid in model_id_list:
        job_id = job_runner.create_job(
            model_id=mid,
            image_id=image_id,
            background_tasks=background_tasks,
            group_id=group_id or "placeholder",
            wsi_name=derived_wsi,
        )
        if group_id is None:
            group_id = job_id
            from backend.db import get_conn
            with get_conn() as conn:
                conn.execute("UPDATE jobs SET group_id = ? WHERE id = ?", (group_id, job_id))
        responses.append(JobResponse(job_id=job_id, image_id=image_id))
    return responses


# 기존 그룹에 모델 추가
@router.post("/jobs/add", response_model=list[JobResponse])
async def add_models_to_group(background_tasks: BackgroundTasks, req: AddModelsRequest):
    if not req.model_ids:
        raise HTTPException(status_code=400, detail="model_ids must not be empty")
    responses = []
    for mid in req.model_ids:
        job_id = job_runner.create_job(
            model_id=mid,
            image_id=req.image_id,
            background_tasks=background_tasks,
            group_id=req.group_id,
            wsi_name=req.wsi_name,
        )
        responses.append(JobResponse(job_id=job_id, image_id=req.image_id))
    return responses


# job_id로 현재 상태(status, progress, message)를 조회해 반환
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message", ""),
        error_detail=job.get("error_detail", ""),
    )


# 실행 중이면 취소, 완료·실패 상태면 DB에서 삭제
@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_runner.delete_job(job_id)


# 완료된 job의 결과(결과 이미지 ID, metrics)를 반환; 미완료면 400 에러
@router.get("/jobs/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not done yet: {job['status']}")
    metrics = {}
    raw_metrics = job.get("metrics")
    if raw_metrics:
        try:
            parsed = json.loads(raw_metrics)
            if isinstance(parsed, dict):
                metrics = parsed
        except Exception:
            pass
    try:
        elapsed = (
            datetime.fromisoformat(job["updated_at"]) - datetime.fromisoformat(job["created_at"])
        ).total_seconds()
    except Exception:
        elapsed = 0.0
    return {
        "job_id": job_id,
        "status": "done",
        "result_image_id": job["result_image_id"],
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
