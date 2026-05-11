from dataclasses import dataclass
from pathlib import Path

@dataclass
class Metrics:
    ssim: float
    psnr: float
    fid: float

@dataclass
class Task:
    src_img_path: Path
    target_img_path: Path | None
    result_path: Path
    model_id: int 

@dataclass
class TaskResult:
    result_img_path: Path
    metrics: Metrics
    thumbnail_path: Path | None = None
