from typing import Any
from pydantic import BaseModel

class ModelResponse(BaseModel):
    id: int
    name: str
    category: str
    description: str

class JobResponse(BaseModel):
    job_id: str
    image_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    error_detail: str = ""

class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result_image_id: str
    metrics: dict
    elapsed_seconds: float

class JobListItem(BaseModel):
    id: str
    model_id: int
    status: str
    progress: float
    message: str | None = None
    error_detail: str | None = None
    result_image_id: str | None
    metrics: Any | None
    elapsed_seconds: float | None

class JobGroupResponse(BaseModel):
    group_id: str
    wsi_name: str
    image_id: str
    created_at: str
    jobs: list[JobListItem]