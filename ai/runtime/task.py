from dataclasses import dataclass
from pathlib import Path

@dataclass
class Metrics:
    ssim: float
    psnr: float
    fid: float
    stain_preservation_corr: float | None = None
    normalized_target_stain_angle_deg: float | None = None
    source_target_stain_angle_deg: float | None = None
    stain_angle_improvement_deg: float | None = None
    custom_structure_score: float | None = None
    custom_color_score: float | None = None
    source_stain_rank: float | None = None
    normalized_stain_rank: float | None = None
    target_stain_rank: float | None = None

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
