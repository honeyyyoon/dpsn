from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image
import tifffile
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

class MultiZarrWSIWriter(PatchWriter):
    """
    Writer that stages patch output in Zarr and emits a pyramid WSI TIFF.

    The public behavior mirrors ZarrWSIWriter: patches are written to a staged
    Zarr array and finalize returns the PNG thumbnail path. In addition, this
    writer leaves a same-stem .tiff WSI next to that thumbnail.
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
        write_levels: tuple[int, ...] = (0,),
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
        if not write_levels:
            raise ValueError("write_levels must contain at least one level")
        if any(level < 0 for level in write_levels):
            raise ValueError(f"write_levels must be >= 0, got {write_levels}")

        self.temp_id = str(uuid.uuid4())

        self.output_path = Path(output_path) / self.temp_id
        self.wsi_path = Path(output_path) / f"{self.temp_id}.tiff"
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.tile_size = int(tile_size)
        self.level_downsample = float(level_downsample)
        self.pyramid_levels = int(pyramid_levels)
        self.write_levels = tuple(sorted(set(write_levels)))
        self.thumbnail_path = Path(output_path) / f"{self.temp_id}.png"
        self.thumbnail_max_size = int(2048)

        if self.output_path.exists() and overwrite:
            shutil.rmtree(self.output_path)
        if self.wsi_path.exists() and overwrite:
            self.wsi_path.unlink()
        if self.thumbnail_path.exists() and overwrite:
            self.thumbnail_path.unlink()

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
        self._write_wsi_tiff()
        # Thumbnail generation is handled by backend image_store when needed.
        # self._write_thumbnail()
        return self.wsi_path
    
    def close(self):
        shutil.rmtree(self.output_path)

    def _write_wsi_tiff(self) -> Path:
        self.wsi_path.parent.mkdir(parents=True, exist_ok=True)
        levels = [self._read_level(level) for level in self.write_levels]
        subifd_count = max(len(levels) - 1, 0)

        with tifffile.TiffWriter(self.wsi_path, bigtiff=True) as tif:
            tif.write(
                levels[0],
                photometric="rgb",
                tile=(self.tile_size, self.tile_size),
                subifds=subifd_count,
                metadata={
                    "axes": "YXS",
                    "writer_type": "multizarr_wsi_tiff",
                    "source_level_downsample": self.level_downsample,
                    "write_levels": self.write_levels,
                },
            )

            for level_img in levels[1:]:
                tif.write(
                    level_img,
                    photometric="rgb",
                    tile=(min(self.tile_size, level_img.shape[0]), min(self.tile_size, level_img.shape[1])),
                    subfiletype=1,
                    metadata={"axes": "YXS"},
                )
        return self.wsi_path
    
    def _read_level(self, level: int) -> np.ndarray:
        if level == 0:
            return np.asarray(self.image[:, :, :], dtype=np.uint8)
        
        stride = 2 ** level
        return np.asarray(self.image[::stride, ::stride, :], dtype=np.uint8)

    # def _write_thumbnail(self) -> Path:
    #     self.thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     max_size = self.thumbnail_max_size
    #     stride = max(1, int(np.ceil(max(self.height / max_size, self.width / max_size))))
    #     thumb_arr = np.asarray(self.image[::stride, ::stride, :], dtype=np.uint8)
    #
    #     img = Image.fromarray(thumb_arr, mode="RGB")
    #     img.thumbnail((max_size, max_size))
    #     img.save(self.thumbnail_path)
    #
    #     return self.thumbnail_path

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
