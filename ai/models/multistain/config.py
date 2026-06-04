"""
config.py defines the scanner IDs, MPP values, canonical target domain, dataset/checkpoint paths, 
training defaults, GPU IDs, and validation rules
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_DATASET_DIR = Path("/mnt/Disk1/dpsn_datasets/multiscanner_dataset")
DEFAULT_CHECKPOINTS_DIR = Path("/mnt/Disk1/dpsn_outputs/checkpoints/multistain")
DEFAULT_PATCH_CACHE_DIR = Path("/mnt/Disk1/dpsn_patch_cache/multistain_patch_cache")


SCANNER_MPP: dict[str, float] = {
    "cs2": 0.25,
    "nz210": 0.22,
    "nz20": 0.23,
    "p1000": 0.25,
    "gt450": 0.26,
}

SCANNER_NAMES: dict[str, str] = {
    "cs2": "Aperio ScanScope CS2",
    "nz210": "NanoZoomer S210",
    "nz20": "NanoZoomer 2.0-HT",
    "p1000": "Pannoramic 1000",
    "gt450": "Aperio GT 450",
}


@dataclass(slots=True)
class MultiStainCycleGANConfig:
    """Configuration for many-source-to-one MultiStain-CycleGAN training."""

    dataset_dir: Path = DEFAULT_DATASET_DIR
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    patch_cache_dir: Path = DEFAULT_PATCH_CACHE_DIR
    experiment_name: str = "multistain_many_to_nz210"

    scanner_mpp: dict[str, float] = field(default_factory=lambda: dict(SCANNER_MPP))
    scanner_names: dict[str, str] = field(default_factory=lambda: dict(SCANNER_NAMES))
    canonical_domain: str = "nz210"
    source_domains: tuple[str, ...] = ("cs2", "nz20", "p1000", "gt450")

    image_size: int = 256
    target_mpp: float | None = None
    read_level: int = 0
    patches_per_source_slide: int = 128
    patches_per_target_slide: int = 128
    mask_longest_side: int = 1024
    strict_mpp_check: bool = True
    recursive: bool = False
    use_patch_cache: bool = True

    train_sample_count: int = 36
    val_sample_count: int = 8
    split_seed: int = 0

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    generator_blocks: int = 9
    discriminator_layers: int = 3

    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 40
    checkpoint_interval: int = 2
    lr: float = 0.0002
    beta1: float = 0.5
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_content: float = 0.0
    pool_size: int = 50

    device: str = "auto"
    gpu_ids: tuple[int, ...] = (1, 2, 3)
    verbose: bool = False

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        self.patch_cache_dir = Path(self.patch_cache_dir)

        if self.target_mpp is None:
            self.target_mpp = self.scanner_mpp[self.canonical_domain]

        self.source_domains = tuple(self.source_domains)
        self._validate()

    @property
    def all_domains(self) -> tuple[str, ...]:
        return (*self.source_domains, self.canonical_domain)

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"{self.experiment_name}_latest.pth"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"{self.experiment_name}_best.pth"

    def epoch_checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoints_dir / f"{self.experiment_name}_epoch_{epoch:04d}.pth"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("dataset_dir", "checkpoints_dir", "patch_cache_dir"):
            payload[key] = str(payload[key])
        return payload

    def _validate(self) -> None:
        if self.canonical_domain not in self.scanner_mpp:
            raise ValueError(f"Unknown canonical_domain: {self.canonical_domain!r}")
        if self.canonical_domain in self.source_domains:
            raise ValueError("canonical_domain must not be included in source_domains")

        missing = sorted(set(self.source_domains) - set(self.scanner_mpp))
        if missing:
            raise ValueError(f"source_domains contain unknown scanner ids: {missing}")

        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if self.target_mpp is None or self.target_mpp <= 0:
            raise ValueError(f"target_mpp must be > 0, got {self.target_mpp}")
        if self.read_level < 0:
            raise ValueError(f"read_level must be >= 0, got {self.read_level}")
        if self.patches_per_source_slide <= 0:
            raise ValueError(
                f"patches_per_source_slide must be > 0, got {self.patches_per_source_slide}"
            )
        if self.patches_per_target_slide <= 0:
            raise ValueError(
                f"patches_per_target_slide must be > 0, got {self.patches_per_target_slide}"
            )
        if self.train_sample_count <= 0:
            raise ValueError(f"train_sample_count must be > 0, got {self.train_sample_count}")
        if self.val_sample_count <= 0:
            raise ValueError(f"val_sample_count must be > 0, got {self.val_sample_count}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be > 0, got {self.checkpoint_interval}"
            )
        if self.pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {self.pool_size}")
        if not self.gpu_ids:
            raise ValueError("gpu_ids must not be empty")
