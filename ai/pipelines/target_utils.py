from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ai.samplers.grid_sampler import GridSampler
from ai.wsi.loader import load_patch, open_wsi_handle


def as_target_paths(target_img_path: Path | Sequence[Path] | None) -> tuple[Path, ...]:
    if target_img_path is None:
        return ()
    if isinstance(target_img_path, Path):
        return (target_img_path,)
    return tuple(Path(path) for path in target_img_path)


def load_grid_target_patches(
    target_img_path: Path | Sequence[Path],
    grid_sampler: GridSampler,
) -> np.ndarray:
    patches = []
    for path in as_target_paths(target_img_path):
        handle = open_wsi_handle(path)
        refs = grid_sampler.sample(handle)
        patches.extend(load_patch(ref).img for ref in refs)

    if not patches:
        raise ValueError("No target patches were sampled from the target image set.")

    return np.stack(patches, axis=0)
