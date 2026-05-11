from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
import uuid

from abc import ABC, abstractmethod
from pathlib import Path
import shutil

import numpy as np
from tifffile import TiffWriter
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


class TiffWSIWriter(PatchWriter):
    """
    Stage patch-wise output to chunked on-disk storage and export as TIFF.

    The write path does not depend on holding the full output image in memory.
    Patches are written into a Zarr-backed canvas first, then exported at
    finalize time.

    Pyramid TIFF export is implemented using tiled SubIFDs via tifffile.
    This is a practical writer for the current codebase, but cross-viewer
    compatibility of generated pyramid TIFFs may still vary by viewer.
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
        pyramid_levels: int = 2,
        compression: str | None = None,
        mpp_x: float | None = None,
        mpp_y: float | None = None,
        keep_store: bool = False,
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
        self.compression = compression
        self.keep_store = keep_store
        self.mpp_x = mpp_x
        self.mpp_y = mpp_y

        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.store_path = self.output_path.with_suffix(self.output_path.suffix + ".zarr")
        if self.store_path.exists() and overwrite:
            shutil.rmtree(self.store_path)

        self.root = zarr.open_group(str(self.store_path), mode="w")
        self.image = self._create_zarr_image(self.root)

        self.root.attrs.update(
            {
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "tile_size": self.tile_size,
                "level_downsample": self.level_downsample,
                "pyramid_levels": self.pyramid_levels,
            }
        )

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1, y1 = self._read_level_pos(ref)

        if x1 >= self.width or y1 >= self.height:
            return
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
        base_kwargs = {
            "data": self._iter_tiles(level=0),
            "shape": (self.height, self.width, self.channels),
            "dtype": np.uint8,
            "photometric": "rgb",
            "tile": (self.tile_size, self.tile_size),
            "compression": self.compression,
            "metadata": None,
        }
        if self.pyramid_levels > 0:
            base_kwargs["subifds"] = self.pyramid_levels
        if self._has_valid_mpp():
            base_kwargs["resolution"] = self._resolution_for_level(0)
            base_kwargs["resolutionunit"] = "CENTIMETER"

        with TiffWriter(str(self.output_path), bigtiff=True) as tif:
            tif.write(**base_kwargs)

            for level in range(1, self.pyramid_levels + 1):
                level_h, level_w = self._level_shape(level)
                level_kwargs = {
                    "data": self._iter_tiles(level=level),
                    "shape": (level_h, level_w, self.channels),
                    "dtype": np.uint8,
                    "photometric": "rgb",
                    "tile": (self.tile_size, self.tile_size),
                    "compression": self.compression,
                    "subfiletype": 1,
                    "metadata": None,
                }
                if self._has_valid_mpp():
                    level_kwargs["resolution"] = self._resolution_for_level(level)
                    level_kwargs["resolutionunit"] = "CENTIMETER"
                tif.write(**level_kwargs)

        if not self.keep_store and self.store_path.exists():
            shutil.rmtree(self.store_path)

        return self.output_path

    def _read_level_pos(self, ref: PatchRef) -> tuple[int, int]:
        x = int(round(ref.x / self.level_downsample))
        y = int(round(ref.y / self.level_downsample))
        return (x, y)

    def _create_zarr_image(self, root):
        kwargs = {
            "name": "image",
            "shape": (self.height, self.width, self.channels),
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

    def _level_shape(self, level: int) -> tuple[int, int]:
        scale = 2**level
        return (
            (self.height + scale - 1) // scale,
            (self.width + scale - 1) // scale,
        )

    def _iter_tiles(self, level: int):
        scale = 2**level
        level_h, level_w = self._level_shape(level)

        for y in range(0, level_h, self.tile_size):
            for x in range(0, level_w, self.tile_size):
                tile_h = min(self.tile_size, level_h - y)
                tile_w = min(self.tile_size, level_w - x)

                tile = self.image[
                    y * scale:(y + tile_h) * scale:scale,
                    x * scale:(x + tile_w) * scale:scale,
                    :,
                ]

                padded = np.zeros(
                    (self.tile_size, self.tile_size, self.channels),
                    dtype=np.uint8,
                )
                padded[:tile_h, :tile_w, :] = np.asarray(tile, dtype=np.uint8)
                yield padded

    def _has_valid_mpp(self) -> bool:
        return (
            self.mpp_x is not None
            and self.mpp_y is not None
            and self.mpp_x > 0
            and self.mpp_y > 0
        )

    def _resolution_for_level(self, level: int) -> tuple[float, float] | None:
        if not self._has_valid_mpp():
            return None

        scale = 2**level
        return (
            1e4 / (self.mpp_x * scale),
            1e4 / (self.mpp_y * scale),
        )


class RegularTiffWSIWriter(TiffWSIWriter):
    """
    Backward-compatible name for the default TIFF writer.

    This now uses the same chunked staging path as TiffWSIWriter instead of
    building the whole output image in RAM.
    """


class ZarrWSIWriter(PatchWriter):
    """Optional writer that keeps the staged output as Zarr only."""

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
        self.temp_id = str(uuid.uuid4())

        self.output_path = Path(output_path) / self.temp_id
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.level_downsample = float(level_downsample)
        self.thumbnail_path = Path(output_path) / f"{self.temp_id}.png"
        self.thumbnail_max_size = int(2048)

        if self.output_path.exists() and overwrite:
            shutil.rmtree(self.output_path)

        self.root = zarr.open_group(str(self.output_path / "temp"), mode="w")
        self.image = self._create_zarr_image(self.root)

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
        return self.thumbnail_path
    
    def close(self):
        shutil.rmtree(self.output_path)
    
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

    def _create_zarr_image(self, root):
        kwargs = {
            "name": "image",
            "shape": (self.height, self.width, self.channels),
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
