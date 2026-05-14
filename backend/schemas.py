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
    status: str  # "pending" / "running" / "done" / "failed"
    progress: int  # 0~100
    message: str  # 진행 상황에 대한 메시지
    
class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result_image_id: str
    metrics: dict  # 나중에 AI 연결 후 구체적인 스키마로 교체