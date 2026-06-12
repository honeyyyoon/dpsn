from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

@dataclass
class Metrics:
    ssim: float
    psnr: float
    fid: float
    gaussian_color_dist: float = 0.0
    gaussian_color_gain: float = 0.0

@dataclass
class Task:
    src_img_path: Path
    target_img_path: Path | None
    result_path: Path
    model_id: int 
    target_img_paths: Sequence[Path] | None = None

@dataclass
class TaskResult:
    result_img_path: Path
    metrics: Metrics
    thumbnail_path: Path | None = None
