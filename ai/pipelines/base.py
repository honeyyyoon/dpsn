from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging
from pathlib import Path

from ai.pipelines.result import PipelineResult


class PipelineError(RuntimeError):
    """Base class for pipeline errors."""


class PipelineInputShapeError(PipelineError):
    """Raised when a pipeline receives tensors or arrays with invalid shapes."""


class ModelPipeline(ABC):
    logger: logging.Logger
    
    def __init__(self, logger: logging.Logger | None):
        self.logger = logger
    
    @abstractmethod
    def run(
        self, 
        src_img_path: Path,
        result_path: Path,
        target_img_path: Path | Sequence[Path] | None,
        metrics: list[str],
        emit_event=None
    ) -> PipelineResult: ...
