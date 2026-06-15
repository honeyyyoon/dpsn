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
            raise ValueError(f"width와 height는 0보다 커야 합니다. 입력값: {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels는 0보다 커야 합니다. 입력값: {channels}")
        if tile_size <= 0:
            raise ValueError(f"tile_size는 0보다 커야 합니다. 입력값: {tile_size}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample은 0보다 커야 합니다. 입력값: {level_downsample}"
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
            raise ValueError(f"write 위치는 음수일 수 없습니다: {(x1, y1)}")

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
            raise ValueError("thumbnail_path가 설정되지 않았습니다.")

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
            raise TypeError(f"img는 numpy.ndarray 타입이어야 합니다. 입력 타입: {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img는 [C, H, W] shape이어야 합니다. 입력 shape: {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img는 CHW 형식의 {self.channels}채널이어야 합니다. 입력 shape: {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img dtype은 uint8이어야 합니다. 입력 dtype: {img.dtype}")

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
            "Zarr group이 create_array 또는 create_dataset을 지원하지 않습니다."
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
            raise ValueError(f"width와 height는 0보다 커야 합니다. 입력값: {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels는 0보다 커야 합니다. 입력값: {channels}")
        if tile_size <= 0:
            raise ValueError(f"tile_size는 0보다 커야 합니다. 입력값: {tile_size}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample은 0보다 커야 합니다. 입력값: {level_downsample}"
            )
        if pyramid_levels < 0:
            raise ValueError(f"pyramid_levels는 0 이상이어야 합니다. 입력값: {pyramid_levels}")
        if not write_levels:
            raise ValueError("write_levels에는 최소 하나 이상의 level이 필요합니다.")
        if any(level < 0 for level in write_levels):
            raise ValueError(f"write_levels는 모두 0 이상이어야 합니다. 입력값: {write_levels}")

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
            raise ValueError(f"write 위치는 음수일 수 없습니다: {(x1, y1)}")

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
            raise TypeError(f"img는 numpy.ndarray 타입이어야 합니다. 입력 타입: {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img는 [C, H, W] shape이어야 합니다. 입력 shape: {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img는 CHW 형식의 {self.channels}채널이어야 합니다. 입력 shape: {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img dtype은 uint8이어야 합니다. 입력 dtype: {img.dtype}")

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
            "Zarr group이 create_array 또는 create_dataset을 지원하지 않습니다."
        )


class BlendedMultiZarrWSIWriter(MultiZarrWSIWriter):
    """MultiZarr writer that blends overlapping inference patches smoothly."""

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
        blend_margin: int | None = None,
        min_weight: float = 0.05,
    ) -> None:
        super().__init__(
            output_path=output_path,
            width=width,
            height=height,
            level_downsample=level_downsample,
            channels=channels,
            tile_size=tile_size,
            overwrite=overwrite,
            pyramid_levels=pyramid_levels,
            write_levels=write_levels,
        )
        if min_weight <= 0 or min_weight > 1:
            raise ValueError(f"min_weight는 (0, 1] 범위여야 합니다. 입력값: {min_weight}")

        self.blend_margin = blend_margin
        self.min_weight = float(min_weight)
        self.image_sum = self._create_zarr_float_image("image_sum")
        self.weight_sum = self._create_zarr_weight_image("weight_sum")

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1 = int(round(ref.x / self.level_downsample))
        y1 = int(round(ref.y / self.level_downsample))
        if x1 < 0 or y1 < 0:
            raise ValueError(f"write 위치는 음수일 수 없습니다: {(x1, y1)}")

        img_hwc = self._to_hwc_uint8(img)
        patch_h, patch_w = img_hwc.shape[:2]
        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)
        write_w = x2 - x1
        write_h = y2 - y1
        if write_w <= 0 or write_h <= 0:
            return

        weights = self._blend_weights(
            write_h=write_h,
            write_w=write_w,
            touches_left=x1 == 0,
            touches_top=y1 == 0,
            touches_right=x2 == self.width,
            touches_bottom=y2 == self.height,
        )
        patch = img_hwc[:write_h, :write_w, :].astype(np.float32)
        image_sum = np.asarray(self.image_sum[y1:y2, x1:x2, :], dtype=np.float32)
        weight_sum = np.asarray(self.weight_sum[y1:y2, x1:x2], dtype=np.float32)
        self.image_sum[y1:y2, x1:x2, :] = image_sum + patch * weights[:, :, None]
        self.weight_sum[y1:y2, x1:x2] = weight_sum + weights

    def finalize(self) -> Path:
        self._flush_blended_image()
        return super().finalize()

    def _flush_blended_image(self) -> None:
        row_chunk = max(1, int(self.tile_size))
        for y1 in range(0, self.height, row_chunk):
            y2 = min(self.height, y1 + row_chunk)
            image_sum = np.asarray(self.image_sum[y1:y2, :, :], dtype=np.float32)
            weight_sum = np.asarray(self.weight_sum[y1:y2, :], dtype=np.float32)

            empty = weight_sum <= 0
            safe_weight = np.where(empty, 1.0, weight_sum)
            image = image_sum / safe_weight[:, :, None]
            image[empty, :] = 0
            self.image[y1:y2, :, :] = np.rint(image).clip(0, 255).astype(np.uint8)

    def _blend_weights(
        self,
        write_h: int,
        write_w: int,
        touches_left: bool,
        touches_top: bool,
        touches_right: bool,
        touches_bottom: bool,
    ) -> np.ndarray:
        wy = self._axis_weights(write_h, touches_top, touches_bottom)
        wx = self._axis_weights(write_w, touches_left, touches_right)
        return np.outer(wy, wx).astype(np.float32)

    def _axis_weights(
        self,
        length: int,
        touches_start: bool,
        touches_end: bool,
    ) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        if length <= 1:
            return weights

        margin = self._effective_blend_margin(length)
        if margin <= 0:
            return weights

        ramp = np.linspace(self.min_weight, 1.0, margin, dtype=np.float32)
        if not touches_start:
            weights[:margin] = np.minimum(weights[:margin], ramp)
        if not touches_end:
            weights[-margin:] = np.minimum(weights[-margin:], ramp[::-1])
        return weights

    def _effective_blend_margin(self, length: int) -> int:
        if self.blend_margin is not None:
            margin = int(self.blend_margin)
        else:
            margin = min(128, max(1, length // 4))
        return max(0, min(margin, length // 2))

    def _create_zarr_float_image(self, name: str):
        kwargs = {
            "name": name,
            "shape": (self.height, self.width, self.channels),
            "chunks": (self.tile_size, self.tile_size, self.channels),
            "dtype": np.float32,
            "fill_value": 0.0,
        }
        if hasattr(self.root, "create_array"):
            return self.root.create_array(**kwargs)
        if hasattr(self.root, "create_dataset"):
            return self.root.create_dataset(**kwargs)
        raise AttributeError(
            "Zarr group이 create_array 또는 create_dataset을 지원하지 않습니다."
        )

    def _create_zarr_weight_image(self, name: str):
        kwargs = {
            "name": name,
            "shape": (self.height, self.width),
            "chunks": (self.tile_size, self.tile_size),
            "dtype": np.float32,
            "fill_value": 0.0,
        }
        if hasattr(self.root, "create_array"):
            return self.root.create_array(**kwargs)
        if hasattr(self.root, "create_dataset"):
            return self.root.create_dataset(**kwargs)
        raise AttributeError(
            "Zarr group이 create_array 또는 create_dataset을 지원하지 않습니다."
        )
