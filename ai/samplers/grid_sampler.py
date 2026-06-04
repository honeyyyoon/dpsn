from __future__ import annotations

from ai.wsi.handle import WSIHandle
from ai.wsi.patch_ref import PatchRef


class GridSamplerError(RuntimeError):
    """Base class for grid sampler errors."""


class InvalidReadLevelError(GridSamplerError):
    """Raised when the configured read level is not available in the WSI."""


class GridSampler:
    """
    Deterministic full-grid sampler for WSI inference.

    This sampler covers the whole selected read level and also includes the
    right/bottom edge when stride does not land exactly on the last valid
    top-left coordinate.
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int | None = None,
        read_level: int = 0,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size는 0보다 커야 합니다. 입력값: {patch_size}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride는 0보다 커야 합니다. 입력값: {stride}")
        if read_level < 0:
            raise ValueError(f"read_level은 0 이상이어야 합니다. 입력값: {read_level}")

        self.patch_size = patch_size
        self.stride = stride
        self.read_level = read_level

    def sample(
        self,
        wsi_handle: WSIHandle,
    ) -> list[PatchRef]:
        level_count = len(wsi_handle.level_dimensions)
        if not (0 <= self.read_level < level_count):
            raise InvalidReadLevelError(
                f"read_level {self.read_level}은 0 이상 {level_count - 1} 이하이어야 합니다."
            )

        read_w, read_h = wsi_handle.level_dimensions[self.read_level]
        patch_w = min(self.patch_size, read_w)
        patch_h = min(self.patch_size, read_h)
        stride = self.patch_size if self.stride is None else self.stride
        level = self.read_level

        xs = self._grid_positions(read_w, patch_w, stride)
        ys = self._grid_positions(read_h, patch_h, stride)

        patch_refs: list[PatchRef] = []
        for y in ys:
            for x in xs:
                patch_ref = wsi_handle.make_ref(
                    pos=(x, y),
                    level=self.read_level,
                    dim=(patch_w, patch_h),
                )
                patch_refs.append(patch_ref)

        return patch_refs

    def _grid_positions(
        self,
        length: int,
        patch_length: int,
        stride: int,
    ) -> list[int]:
        max_start = length - patch_length
        positions = list(range(0, max_start + 1, stride))

        if not positions or positions[-1] != max_start:
            positions.append(max_start)

        return positions
