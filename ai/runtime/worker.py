from __future__ import annotations

from importlib import import_module
import logging
from pathlib import Path
from typing import Any

from ai.metrics.metric import Metric
from ai.pipelines.base import ModelPipeline
from ai.runtime.task import Metrics, Task, TaskResult
from ai.samplers.evaluation_sampler import EvaluationSampler, EvaluationSamples


class WorkerError(RuntimeError):
    """Base class for worker errors."""


class UnknownModelError(WorkerError):
    """Raised when a task asks for a model_id that is not available."""


class InvalidPipelineResultError(WorkerError):
    """Raised when a pipeline does not return the expected output path."""


class PipelineImportError(WorkerError):
    """Raised when a registered pipeline module or class cannot be loaded."""


PIPELINE_MAP: dict[int, str] = {
    1: "ai.pipelines.reinhard:Reinhard",
    2: "ai.pipelines.macenko:Macenko",  
    3: "ai.pipelines.vahadane:Vahadane",
    4: "ai.pipelines.staingan:StainGANPipeline",  
    5: "ai.pipelines.stainnet:StainNetPipeline",
    6: "ai.pipelines.stainswin:StainSWINPipeline",
    7: "ai.pipelines.multistain_cyclegan:MultiStainCycleGANPipeline",
}

PIPELINE_CONFIG_MAP: dict[int, str] = {
    4: "StainGANInferenceConfig",
    5: "StainNetInferenceConfig",
    6: "StainSWINInferenceConfig",
    7: "MultiStainCycleGANInferenceConfig",
}

class Worker:
    """Simple runtime coordinator for one normalization task."""

    def run(self, task: Task, emit_event) -> TaskResult:
        evaluation_samples = self._sample_for_evaluation(task, emit_event)

        emit_event(status="running", progress=1, message="Loading pipeline.")
        pipeline = self._create_pipeline(task.model_id)
        pipeline_result = pipeline.run(
            task.src_img_path, 
            task.result_path,
            task.target_img_path,
            ["ssim", "psnr"],
            emit_event=emit_event
        )
        sampled_scores = self._evaluate_sampled_metrics(
            evaluation_samples=evaluation_samples,
            output_path=pipeline_result.output_path,
            emit_event=emit_event,
        )
        metrics = Metrics(
            ssim=self._score_or_zero(pipeline_result.scores.get("ssim")),
            psnr=self._score_or_zero(pipeline_result.scores.get("psnr")),
            fid=self._score_or_zero(sampled_scores.get("fid")),
            gaussian_color_dist=self._score_or_zero(
                sampled_scores.get("gaussian_color_dist")
            ),
            gaussian_color_gain=self._score_or_zero(
                sampled_scores.get("gaussian_color_gain")
            ),
        )

        return TaskResult(
            result_img_path=pipeline_result.output_path,
            metrics=metrics,
            thumbnail_path=pipeline_result.thumbnail_path,
        )

    def _sample_for_evaluation(
        self,
        task: Task,
        emit_event,
    ) -> EvaluationSamples | None:
        if task.target_img_path is None:
            return None

        emit_event(
            status="running",
            progress=1,
            message="Sampling fixed evaluation patches.",
        )
        sampler = EvaluationSampler()
        return sampler.sample(task.src_img_path, task.target_img_path)

    def _evaluate_sampled_metrics(
        self,
        evaluation_samples: EvaluationSamples | None,
        output_path: Path,
        emit_event,
    ) -> dict[str, float | None]:
        if evaluation_samples is None:
            return {
                "fid": None,
                "gaussian_color_dist": None,
                "gaussian_color_gain": None,
            }

        emit_event(
            status="running",
            progress=100,
            message="Evaluating sampled color metrics.",
        )
        sampler = EvaluationSampler()
        output_patches = sampler.load_output_patches(evaluation_samples, output_path)
        metric = Metric(
            use_ssim=False,
            use_psnr=False,
            use_fid=True,
            use_gaussian_color_dist=True,
            target_patch=evaluation_samples.target_patches,
        )
        metric.evaluate(evaluation_samples.source_patches, output_patches)
        return metric.finalize()

    def _create_pipeline(self, model_id: int) -> ModelPipeline:
        pipeline_path = PIPELINE_MAP.get(model_id)
        if pipeline_path is None:
            raise UnknownModelError(
                f"model_id {model_id}에 등록된 파이프라인이 없습니다."
            )

        module_path, class_name = pipeline_path.split(":", maxsplit=1)
        try:
            module = import_module(module_path)
            pipeline_class = getattr(module, class_name)
        except (ImportError, AttributeError) as error:
            raise PipelineImportError(
                f"model_id {model_id}의 파이프라인 '{pipeline_path}'을 불러오지 못했습니다: {error}"
            ) from error
        return pipeline_class(self._build_logger(Path("result/log.txt")))

    def _score_or_zero(self, value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    def _get_result_img_path(self, pipeline_result: Any) -> Path:
        output_path = getattr(pipeline_result, "output_path", None)
        if not output_path:
            raise InvalidPipelineResultError(
                "파이프라인 결과에는 비어 있지 않은 output_path가 필요합니다."
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
