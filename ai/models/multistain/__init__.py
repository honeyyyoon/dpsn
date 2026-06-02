from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "GANLoss": "ai.models.multistain.networks",
    "ImagePool": "ai.models.multistain.networks",
    "MultiStainCycleGANConfig": "ai.models.multistain.config",
    "MultiStainCycleGANModel": "ai.models.multistain.multistain_model",
    "MultiStainPatchDataset": "ai.models.multistain.dataset",
    "NLayerDiscriminator": "ai.models.multistain.networks",
    "ResnetGenerator": "ai.models.multistain.networks",
    "SCANNER_MPP": "ai.models.multistain.config",
    "SCANNER_NAMES": "ai.models.multistain.config",
    "SlideRecord": "ai.models.multistain.dataset",
    "create_datasets": "ai.models.multistain.dataset",
    "discover_multiscanner_records": "ai.models.multistain.dataset",
    "split_sample_ids": "ai.models.multistain.dataset",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
