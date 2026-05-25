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

class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result_image_id: str
    metrics: dict

class JobListItem(BaseModel):
    id: str
    model_id: int
    status: str
    progress: float
    result_image_id: str | None
    metrics: Any | None

class JobGroupResponse(BaseModel):
    group_id: str
    wsi_name: str
    image_id: str
    created_at: str
    jobs: list[JobListItem]