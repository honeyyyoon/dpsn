from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.patch_ref import PatchRef


class EvaluationSamplerError(RuntimeError):
    """Base class for evaluation sampler errors."""


@dataclass(frozen=True)
class EvaluationSamples:
    source_patches: np.ndarray
    target_patches: np.ndarray
    source_refs: tuple[PatchRef, ...]
    target_refs: tuple[PatchRef, ...]
    source_dimensions: tuple[int, int]
    source_read_level: int
    target_read_levels: tuple[int, ...]


class EvaluationSampler:
    """Deterministic sampler for model-comparison metrics."""

    def __init__(
        self,
        patch_size: int = 256,
        max_source_patches: int = 64,
        max_target_patches: int = 512,
        seed: int = 0,
        training_tissue_threshold: float = 0.5,
        quality_tissue_threshold: float = 0.2,
        max_black_fraction: float = 0.1,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size는 0보다 커야 합니다. 입력값: {patch_size}")
        if max_source_patches <= 0:
            raise ValueError(
                f"max_source_patches는 0보다 커야 합니다. 입력값: {max_source_patches}"
            )
        if max_target_patches <= 0:
            raise ValueError(
                f"max_target_patches는 0보다 커야 합니다. 입력값: {max_target_patches}"
            )

        self.patch_size = int(patch_size)
        self.max_source_patches = int(max_source_patches)
        self.max_target_patches = int(max_target_patches)
        self.seed = int(seed)
        self.training_tissue_threshold = float(training_tissue_threshold)
        self.quality_tissue_threshold = float(quality_tissue_threshold)
        self.max_black_fraction = float(max_black_fraction)

    def sample(
        self,
        source_img_path: Path,
        target_img_path: Path | Sequence[Path],
    ) -> EvaluationSamples:
        target_img_paths = self._as_path_tuple(target_img_path)
        if not target_img_paths:
            raise EvaluationSamplerError("At least one target image is required.")
        if not 0.0 <= self.quality_tissue_threshold <= 1.0:
            raise ValueError("quality_tissue_threshold must be between 0 and 1.")
        if not 0.0 <= self.max_black_fraction <= 1.0:
            raise ValueError("max_black_fraction must be between 0 and 1.")

        source_handle = open_wsi_handle(source_img_path)
        target_handles = [open_wsi_handle(path) for path in target_img_paths]

        source_level = self._select_source_read_level(source_handle)
        source_refs = self._sample_refs(
            source_handle,
            read_level=source_level,
            max_patches=self.max_source_patches,
        )

        target_levels: list[int] = []
        target_refs: list[PatchRef] = []
        per_target_limit = max(1, math.ceil(self.max_target_patches / len(target_handles)))
        for target_handle in target_handles:
            target_level = self._select_matching_target_read_level(
                source_handle=source_handle,
                target_handle=target_handle,
                source_level=source_level,
            )
            target_levels.append(target_level)
            remaining = self.max_target_patches - len(target_refs)
            if remaining <= 0:
                break
            target_refs.extend(
                self._sample_refs(
                    target_handle,
                    read_level=target_level,
                    max_patches=min(per_target_limit, remaining),
                )
            )

        return EvaluationSamples(
            source_patches=self._load_refs(source_refs),
            target_patches=self._load_refs(target_refs),
            source_refs=tuple(source_refs),
            target_refs=tuple(target_refs),
            source_dimensions=source_handle.dim,
            source_read_level=source_level,
            target_read_levels=tuple(target_levels),
        )

    def load_output_patches(
        self,
        samples: EvaluationSamples,
        output_img_path: Path,
    ) -> np.ndarray:
        output_handle = open_wsi_handle(output_img_path)
        scale_x = samples.source_dimensions[0] / output_handle.dim[0]
        scale_y = samples.source_dimensions[1] / output_handle.dim[1]

        patches = []
        for source_ref in samples.source_refs:
            output_ref = self._source_ref_to_output_ref(
                source_ref=source_ref,
                output_handle=output_handle,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            patch = load_patch(output_ref).img
            patch = self._resize_chw(patch, height=source_ref.height, width=source_ref.width)
            patches.append(patch)

        return np.stack(patches, axis=0)

    def _select_source_read_level(self, source_handle: WSIHandle) -> int:
        for level, (width, height) in enumerate(source_handle.level_dimensions):
            if width >= self.patch_size and height >= self.patch_size:
                return level
        return 0

    def _select_matching_target_read_level(
        self,
        source_handle: WSIHandle,
        target_handle: WSIHandle,
        source_level: int,
    ) -> int:
        candidate_levels = [
            level
            for level, (width, height) in enumerate(target_handle.level_dimensions)
            if width >= self.patch_size and height >= self.patch_size
        ]
        if not candidate_levels:
            candidate_levels = [0]

        if self._has_valid_mpp(source_handle) and self._has_valid_mpp(target_handle):
            source_mpp = self._effective_mpp(source_handle, source_level)
            return min(
                candidate_levels,
                key=lambda level: abs(
                    math.log(self._effective_mpp(target_handle, level) / source_mpp)
                ),
            )

        source_longest_side = max(source_handle.level_dimensions[source_level])
        return min(
            candidate_levels,
            key=lambda level: abs(
                math.log(max(target_handle.level_dimensions[level]) / source_longest_side)
            ),
        )

    def _sample_refs(
        self,
        wsi_handle: WSIHandle,
        read_level: int,
        max_patches: int,
    ) -> list[PatchRef]:
        sampler = PatchSampler(
            patch_size=self.patch_size,
            read_level=read_level,
            training_tissue_threshold=self.training_tissue_threshold,
            strict_mpp_check=False,
        )
        candidate_limit = max_patches * 3
        candidate_refs = sampler.sample(
            wsi_handle,
            mode="training",
            max_patches=candidate_limit,
            seed=self.seed,
            save_debug=False,
        )
        filtered_refs = [
            ref for ref in candidate_refs
            if self._is_quality_metric_patch(load_patch(ref).img)
        ]
        if not filtered_refs:
            return candidate_refs[:max_patches]
        return filtered_refs[:max_patches]

    def _source_ref_to_output_ref(
        self,
        source_ref: PatchRef,
        output_handle: WSIHandle,
        scale_x: float,
        scale_y: float,
    ) -> PatchRef:
        out_x = int(round(source_ref.x / scale_x))
        out_y = int(round(source_ref.y / scale_y))
        out_w = max(1, int(round(source_ref.width * source_ref.downsample / scale_x)))
        out_h = max(1, int(round(source_ref.height * source_ref.downsample / scale_y)))

        out_x = min(max(out_x, 0), max(0, output_handle.dim[0] - 1))
        out_y = min(max(out_y, 0), max(0, output_handle.dim[1] - 1))
        out_w = min(out_w, output_handle.dim[0] - out_x)
        out_h = min(out_h, output_handle.dim[1] - out_y)

        return output_handle.make_ref(
            pos=(out_x, out_y),
            level=0,
            dim=(out_w, out_h),
        )

    def _load_refs(self, refs: list[PatchRef]) -> np.ndarray:
        return np.stack([load_patch(ref).img for ref in refs], axis=0)

    def _is_quality_metric_patch(self, patch: np.ndarray) -> bool:
        rgb = np.transpose(patch, (1, 2, 0)).astype(np.float32) / 255.0
        max_rgb = rgb.max(axis=2)
        min_rgb = rgb.min(axis=2)
        saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)

        black_fraction = float((max_rgb < 0.06).mean())
        tissue = (
            (max_rgb > 0.08)
            & (max_rgb < 0.92)
            & (saturation > 0.05)
        )
        tissue_fraction = float(tissue.mean())
        return (
            black_fraction <= self.max_black_fraction
            and tissue_fraction >= self.quality_tissue_threshold
        )

    def _as_path_tuple(self, path_or_paths: Path | Sequence[Path]) -> tuple[Path, ...]:
        if isinstance(path_or_paths, Path):
            return (path_or_paths,)
        return tuple(Path(path) for path in path_or_paths)

    def _resize_chw(self, patch: np.ndarray, height: int, width: int) -> np.ndarray:
        if patch.shape[1] == height and patch.shape[2] == width:
            return patch

        image = Image.fromarray(np.transpose(patch, (1, 2, 0)), mode="RGB")
        image = image.resize((width, height), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

    def _has_valid_mpp(self, wsi_handle: WSIHandle) -> bool:
        mpp_x, mpp_y = wsi_handle.mpp
        return mpp_x > 0 and mpp_y > 0

    def _effective_mpp(self, wsi_handle: WSIHandle, level: int) -> float:
        mpp_x, mpp_y = wsi_handle.mpp
        return ((float(mpp_x) + float(mpp_y)) / 2.0) * float(
            wsi_handle.level_downsamples[level]
        )
