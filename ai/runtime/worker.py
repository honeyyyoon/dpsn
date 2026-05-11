from __future__ import annotations

from importlib import import_module
import logging
from pathlib import Path
from typing import Any

from ai.metrics.ssim import SSIM
from ai.metrics.psnr import PSNR
from ai.metrics.fid import FID
from ai.pipelines.base import ModelPipeline
from ai.runtime.task import Metrics, Task, TaskResult


class WorkerError(RuntimeError):
    """Base class for worker errors."""


class UnknownModelError(WorkerError):
    """Raised when a task asks for a model_id that is not available."""


class InvalidPipelineResultError(WorkerError):
    """Raised when a pipeline does not return the expected output path."""


PIPELINE_MAP: dict[int, str] = {
    1: "ai.pipelines.reinhard:Reinhard",
    # 2: Macenko(),  
    # 3: Vahadane(),  
    # 4: StainGAN(),  
    5: "ai.pipelines.stainnet:StainNetPipeline",
    # 6: StainSWIN(), 
}

class Worker:
    """Simple runtime coordinator for one normalization task."""

    def run(self, task: Task, emit_event=None) -> TaskResult:
        pipeline = self._create_pipeline(task.model_id)
        pipeline_result = pipeline.run(
            task.src_img_path, 
            task.result_path,
            task.target_img_path,
            {
                "ssim": SSIM(),
                "psnr": PSNR(),
                "fid": FID(),
            }
        )
        # Metrics are still placeholder-level in the current project.
        # TODO: ai/metrics/ 구현 후 실제 metrics 계산으로 교체
        metrics = Metrics(ssim=0.95, psnr=32.4, fid=60)

        return TaskResult(
            result_img_path=pipeline_result.output_path,
            metrics=metrics,
            thumbnail_path=pipeline_result.thumbnail_path,
        )

    def _create_pipeline(self, model_id: int) -> ModelPipeline:
        pipeline_path = PIPELINE_MAP.get(model_id)
        if pipeline_path is None:
            raise UnknownModelError(
                f"model_id {model_id} does not have a registered pipeline."
            )

        module_path, class_name = pipeline_path.split(":", maxsplit=1)
        module = import_module(module_path)
        pipeline_class = getattr(module, class_name)
        return pipeline_class(self._build_logger(Path("result/log.txt")))

    def _get_result_img_path(self, pipeline_result: Any) -> Path:
        output_path = getattr(pipeline_result, "output_path", None)
        if not output_path:
            raise InvalidPipelineResultError(
                "Pipeline result must contain a non-empty output_path."
            )

        return Path(output_path)
    
    def _build_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger_name = f"Worker:{log_path.stem}:{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # avoid duplicated handlers if recreated
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger
