from abc import ABC, abstractmethod
import logging
from pathlib import Path

from ai.metrics.base import Metric
from ai.pipelines.result import PipelineResult

class ModelPipeline(ABC):
    logger: logging.Logger
    
    def __init__(self, logger: logging.Logger | None):
        self.logger = logger
    
    @abstractmethod
    def run(
        self, 
        src_img_path: Path,
        result_path: Path,
        target_img_path: Path | None,
        metrics: dict[str, Metric],
    ) -> PipelineResult: ...
