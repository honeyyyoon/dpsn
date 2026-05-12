from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from backend.schemas import JobResponse, JobStatusResponse, JobResultResponse
from backend.services import job_runner, image_store

router = APIRouter()

# 이미지와 model_id 목록을 받아 각 모델에 대한 job을 생성하고 job_id 목록을 반환
@router.post("/jobs", response_model=list[JobResponse])
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    model_ids: str = Form(...)
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
    return [JobResponse(job_id=job_runner.create_job(mid, image_id, background_tasks), image_id=image_id) for mid in model_id_list]

# job_id로 job을 조회하고 현재 status를 반환
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    if job_id not in job_runner.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = job_runner.jobs[job_id]
    return JobStatusResponse(job_id=job_id, status=job["status"])

# job이 done 상태일 때 job_id로 결과(metrics)를 조회해 반환
@router.get("/jobs/{job_id}/results", response_model= JobResultResponse)
async def get_job_results(job_id: str):
    # TODO: metrics 스키마 구체화 (AI 모델 연결 후 dict → Pydantic 모델로 교체)
    if job_id not in job_runner.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_runner.jobs[job_id]
    
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not done yet: {job['status']}")
    
    return {
        "job_id": job_id,
        "status": "done",
        "result_image_id": job["result_image_id"],
        "metrics": job["result"]
    }