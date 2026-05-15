from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import zarr

from ai.wsi.patch_ref import PatchRef


class PatchWriter(ABC):
    """Common interface for patch-wise WSI output writing."""

    @abstractmethod
    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        """Write one CHW uint8 RGB patch into the output canvas."""

    @abstractmethod
    def finalize(self) -> Path:
        """Flush all staged data and return the final output path."""


class ZarrWSIWriter(PatchWriter):
    """Single-level Zarr writer with PNG thumbnail extraction."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        level_downsample: float,
        channels: int = 3,
        tile_size: int = 512,
        overwrite: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {tile_size}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample must be > 0, got {level_downsample}"
            )
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.level_downsample = float(level_downsample)
        self.thumbnail_path = self.output_path.parent / "out_image.png"
        self.thumbnail_max_size = int(2048)

        if self.output_path.exists() and overwrite:
            shutil.rmtree(self.output_path)

        self.root = zarr.open_group(str(self.output_path), mode="w")
        self.image = self._create_zarr_array(
            root=self.root,
            name="image",
            shape=(self.height, self.width, self.channels),
        )
        self.root.attrs.update(
            {
                "writer_type": "zarr",
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "tile_size": self.tile_size,
                "level_downsample": self.level_downsample,
            }
        )

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1 = int(round(ref.x / self.level_downsample))
        y1 = int(round(ref.y / self.level_downsample))
        if x1 < 0 or y1 < 0:
            raise ValueError(f"Negative write position: {(x1, y1)}")

        img_hwc = self._to_hwc_uint8(img)
        patch_h, patch_w = img_hwc.shape[:2]
        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)
        write_w = x2 - x1
        write_h = y2 - y1
        if write_w <= 0 or write_h <= 0:
            return

        self.image[y1:y2, x1:x2, :] = img_hwc[:write_h, :write_w, :]

    def finalize(self) -> Path:
        self._write_thumbnail()
        return self.output_path

    def close(self) -> None:
        # Kept for pipeline compatibility. The finalized Zarr output should
        # remain on disk, so this is intentionally a no-op.
        return None
    
    def _write_thumbnail(self) -> Path:
        if self.thumbnail_path is None:
            raise ValueError("thumbnail_path is not set.")

        self.thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

        max_size = self.thumbnail_max_size
        stride = max(1, int(np.ceil(max(self.height / max_size, self.width / max_size))))

        thumb = self.image[::stride, ::stride, :]

        thumb_arr = np.asarray(thumb, dtype=np.uint8)

        img = Image.fromarray(thumb_arr, mode="RGB")
        img.thumbnail((max_size, max_size))
        img.save(self.thumbnail_path)

        return self.thumbnail_path

    def _to_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be a numpy.ndarray, got {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img must have shape [C, H, W], got {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img must have {self.channels} channels in CHW format, got {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img must be uint8, got {img.dtype}")

        return np.transpose(img, (1, 2, 0))

    def _create_zarr_array(self, root, name: str, shape: tuple[int, int, int]):
        kwargs = {
            "name": name,
            "shape": shape,
            "chunks": (self.tile_size, self.tile_size, self.channels),
            "dtype": np.uint8,
            "fill_value": 0,
        }
        if hasattr(root, "create_array"):
            return root.create_array(**kwargs)
        if hasattr(root, "create_dataset"):
            return root.create_dataset(**kwargs)
        raise AttributeError(
            "Zarr group does not support create_array or create_dataset."
        )


class MultiZarrWSIWriter(PatchWriter):
    """
    Multiscale Zarr writer using the layout image/<level>.

    Level 0 is written patch-by-patch during inference. Lower-resolution pyramid
    levels are derived from level 0 during finalize and stored as additional
    arrays in the same Zarr group.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        level_downsample: float,
        channels: int = 3,
        tile_size: int = 512,
        overwrite: bool = True,
        pyramid_levels: int = 3,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {tile_size}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample must be > 0, got {level_downsample}"
            )
        if pyramid_levels < 0:
            raise ValueError(f"pyramid_levels must be >= 0, got {pyramid_levels}")

        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.level_downsample = float(level_downsample)
        self.pyramid_levels = int(pyramid_levels)
        self.thumbnail_path = self.output_path.parent / "out_image.png"
        self.thumbnail_max_size = int(2048)

        if self.output_path.exists() and overwrite:
            shutil.rmtree(self.output_path)

        self.root = zarr.open_group(str(self.output_path), mode="w")
        self.image_group = self.root.require_group("image")
        self.level0 = self._create_zarr_array(
            root=self.image_group,
            name="0",
            shape=(self.height, self.width, self.channels),
        )
        self.root.attrs.update(
            {
                "writer_type": "multizarr",
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "tile_size": self.tile_size,
                "level_downsample": self.level_downsample,
                "pyramid_levels": self.pyramid_levels,
                "layout": "image/<level>",
            }
        )

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1 = int(round(ref.x / self.level_downsample))
        y1 = int(round(ref.y / self.level_downsample))
        if x1 < 0 or y1 < 0:
            raise ValueError(f"Negative write position: {(x1, y1)}")

        img_hwc = self._to_hwc_uint8(img)
        patch_h, patch_w = img_hwc.shape[:2]
        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)
        write_w = x2 - x1
        write_h = y2 - y1
        if write_w <= 0 or write_h <= 0:
            return

        self.level0[y1:y2, x1:x2, :] = img_hwc[:write_h, :write_w, :]

    def finalize(self) -> Path:
        self._write_pyramid_levels()
        self._write_thumbnail()
        return self.output_path

    def _write_pyramid_levels(self) -> None:
        current = self.level0
        current_h = self.height
        current_w = self.width

        for level in range(1, self.pyramid_levels + 1):
            next_h = max(1, (current_h + 1) // 2)
            next_w = max(1, (current_w + 1) // 2)

            downsampled = np.asarray(current[::2, ::2, :], dtype=np.uint8)
            downsampled = downsampled[:next_h, :next_w, :]

            array = self._create_zarr_array(
                root=self.image_group,
                name=str(level),
                shape=(next_h, next_w, self.channels),
            )
            array[:, :, :] = downsampled

            current = array
            current_h = next_h
            current_w = next_w

    def _write_thumbnail(self) -> Path:
        self.thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

        level_key = str(self.pyramid_levels)
        source = self.image_group[level_key] if level_key in self.image_group else self.level0
        thumb_arr = np.asarray(source, dtype=np.uint8)

        img = Image.fromarray(thumb_arr, mode="RGB")
        img.thumbnail((self.thumbnail_max_size, self.thumbnail_max_size))
        img.save(self.thumbnail_path)

        return self.thumbnail_path

    def _to_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be a numpy.ndarray, got {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img must have shape [C, H, W], got {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img must have {self.channels} channels in CHW format, got {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img must be uint8, got {img.dtype}")

        return np.transpose(img, (1, 2, 0))

    def _create_zarr_array(self, root, name: str, shape: tuple[int, int, int]):
        kwargs = {
            "name": name,
            "shape": shape,
            "chunks": (min(self.tile_size, shape[0]), min(self.tile_size, shape[1]), self.channels),
            "dtype": np.uint8,
            "fill_value": 0,
        }
        if hasattr(root, "create_array"):
            return root.create_array(**kwargs)
        if hasattr(root, "create_dataset"):
            return root.create_dataset(**kwargs)
        raise AttributeError(
            "Zarr group does not support create_array or create_dataset."
        )
