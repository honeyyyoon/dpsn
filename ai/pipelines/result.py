from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PipelineResult:
    output_path: Path
    scores: dict = field(default_factory=dict)
    thumbnail_path: Path | None = None
