from collections import defaultdict
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch

from ai.metrics.metric import Metric
from ai.pipelines.base import ModelPipeline, PipelineInputShapeError
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter


class ReinhardError(RuntimeError):
    """Base class for Reinhard pipeline errors."""


class MissingTargetImageError(ReinhardError):
    """Raised when Reinhard normalization is run without a target image."""


class ReinhardImageTooLargeError(ReinhardError):
    """Raised when the selected WSI would exceed the configured iteration limit."""


@dataclass(frozen=True)
class ReinhardStats:
    means: torch.Tensor
    stds: torch.Tensor


class Reinhard(ModelPipeline):
    target_sampler: PatchSampler

    def __init__(
        self, 
        logger: logging.Logger,
        batch_size: int = 64,
        patch_size: int = 256,
        max_sample_patches: int = 16,
        max_iteration: int = 128,
        device: str | torch.device | None = None,
    ):
        super().__init__(logger=logger)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.max_sample_patches = int(max_sample_patches)
        self.max_iteration = int(max_iteration)
        self.device = self._select_device(device)
        self._color_constants_cache: dict[
            tuple[torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._validate_config()

    def run(
        self,
        src_img_path: Path,
        result_path: Path, 
        target_img_path: Path | None,
        metrics: list[str],
        emit_event=None
    ) -> PipelineResult:
        self.logger.info("Run Reinhard")
        self.logger.info(f"Use Reinhard device: {self.device}")
        
        if target_img_path is None:
            raise MissingTargetImageError("Reinhard 정규화에는 타겟 이미지가 필요합니다.")
        
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)
        self._log_wsi_info("source", src_wsi_handle)
        self._log_wsi_info("target", target_wsi_handle)

        patch_sampler = PatchSampler(
            training_tissue_threshold=0.3,
            strict_mpp_check=False
        )
        src_images = self._sample_patch_images(
            patch_sampler=patch_sampler,
            wsi_handle=src_wsi_handle,
            label="source",
        )
        tgt_images = self._sample_patch_images(
            patch_sampler=patch_sampler,
            wsi_handle=target_wsi_handle,
            label="target",
        )

        target_stats = self._fit_stats(tgt_images, label="target")
        source_stats = self._fit_stats(src_images, label="source")
        metric = self._build_metric(metrics, tgt_images)

        level = self._select_read_level(src_wsi_handle)
        self.logger.info(
            "Grid Sample from Source Image: "
            f"patch_size={self.patch_size}, read_level={level}, "
            f"downsamples={src_wsi_handle.level_downsamples[level]}"
        )
        grid_sampler = GridSampler(patch_size=self.patch_size, read_level=level)
        src_refs = grid_sampler.sample(src_wsi_handle)
        self.logger.info(f"Sampled: {len(src_refs)}")

        writer = MultiZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[level][0], 
            src_wsi_handle.level_dimensions[level][1],
            level_downsample=src_wsi_handle.level_downsamples[level],
            tile_size = src_refs[0].width
        )
        timer = self._process_batches(
            src_refs=src_refs,
            writer=writer,
            metric=metric,
            source_stats=source_stats,
            target_stats=target_stats,
            emit_event=emit_event,
        )
        
        self.logger.info("Finish normalize")
        self.logger.info(
            f"Elapsed time: load({timer['load']:.4f}s), "
            f"transform({timer['transform']:.4f}s), "
            f"metric({timer['metric']:.4f}s), writer({timer['writer']:.4f}s)"
        )
        
        output_path = writer.finalize()
        writer.close()
        # self.logger.info(f"Save Normalized Image: {image.width} x {image.height}")
        

        return PipelineResult(
            output_path=output_path,
            scores=metric.finalize(),
            thumbnail_path=None,
        )

    def _validate_config(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size는 0보다 커야 합니다. 입력값: {self.batch_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size는 0보다 커야 합니다. 입력값: {self.patch_size}")
        if self.max_sample_patches <= 0:
            raise ValueError(
                f"max_sample_patches는 0보다 커야 합니다. 입력값: {self.max_sample_patches}"
            )
        if self.max_iteration <= 0:
            raise ValueError(f"max_iteration은 0보다 커야 합니다. 입력값: {self.max_iteration}")

    def _select_device(self, device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _log_wsi_info(self, label: str, wsi_handle) -> None:
        self.logger.info(f"Read {label} image")
        self.logger.info(f"Size: {wsi_handle.dim[0]} x {wsi_handle.dim[1]}")
        self.logger.info(f"Level: 0 - {wsi_handle.max_level}")
        self.logger.info(f"Mpp: {wsi_handle.mpp}")

    def _sample_patch_images(
        self,
        patch_sampler: PatchSampler,
        wsi_handle,
        label: str,
    ) -> np.ndarray:
        self.logger.info(f"Sample {label} image")
        refs = patch_sampler.sample(
            wsi_handle,
            max_patches=self.max_sample_patches,
            mode="training",
            save_debug=False,
        )
        images = np.stack([load_patch(ref).img for ref in refs], axis=0)
        self.logger.info(f"Patches: {len(images)}")

        return images

    def _fit_stats(self, images: np.ndarray, label: str) -> ReinhardStats:
        self.logger.info(f"Get {label} Reinhard Stats")
        stats = self._get_reinhard_stats_tensor(self._to_chw_float_tensor(images))
        means = stats.means.detach().cpu().numpy()
        stds = stats.stds.detach().cpu().numpy()
        self.logger.info(
            f"{label.capitalize()} stat: "
            f"means={means.round(2)}, stds={stds.round(2)}"
        )

        return stats

    def _build_metric(self, metrics: list[str], target_images: np.ndarray) -> Metric:
        return Metric(
            use_ssim="ssim" in metrics,
            use_psnr="psnr" in metrics,
            use_fid="fid" in metrics,
            use_custom="custom" in metrics,
            target_patch=target_images,
        )

    def _select_read_level(self, wsi_handle) -> int:
        level = 0
        expected_iterations = self._expected_iterations(
            wsi_handle.level_dimensions[level]
        )

        while (
            expected_iterations > self.max_iteration
            and level < wsi_handle.max_level
        ):
            level += 1
            expected_iterations = self._expected_iterations(
                wsi_handle.level_dimensions[level]
            )

        if expected_iterations > self.max_iteration:
            raise ReinhardImageTooLargeError(
                "이미지가 너무 큽니다. "
                f"예상 반복 횟수: {expected_iterations}, "
                f"최대 반복 횟수: {self.max_iteration}"
            )

        return level

    def _expected_iterations(self, size: tuple[int, int]) -> int:
        width, height = size
        columns = max(1, math.ceil(width / self.patch_size))
        rows = max(1, math.ceil(height / self.patch_size))

        return math.ceil(columns * rows / self.batch_size)

    def _build_writer(
        self,
        result_path: Path,
        src_wsi_handle,
        src_refs,
        level: int,
    ) -> MultiZarrWSIWriter:
        return MultiZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[0][0],
            src_wsi_handle.level_dimensions[0][1],
            level_downsample=src_wsi_handle.level_downsamples[level],
            tile_size=src_refs[0].width,
        )

    def _process_batches(
        self,
        src_refs,
        writer: MultiZarrWSIWriter,
        metric: Metric,
        source_stats: ReinhardStats,
        target_stats: ReinhardStats,
        emit_event=None,
    ) -> defaultdict[str, float]:
        timer = defaultdict(float)
        total_steps = math.ceil(len(src_refs) / self.batch_size)

        step = 0
        for idx in range(0, len(src_refs), self.batch_size):
            step += 1
            t0 = time.time()
            batch_ref = src_refs[idx:idx + self.batch_size]
            patches = np.stack([load_patch(ref).img for ref in batch_ref], axis=0)
            timer['load'] += time.time() - t0

            t0 = time.time()
            patch_tensor = self._to_chw_float_tensor(patches)
            new_patch_tensor = self._transform_with_stats_tensor(
                patch_tensor,
                source_stats=source_stats,
                target_stats=target_stats,
            )
            timer['transform'] += time.time() - t0

            t0 = time.time()
            metric.evaluate_torch(patch_tensor, new_patch_tensor)
            timer['metric'] += time.time() - t0

            t0 = time.time()
            new_patches = self._to_chw_uint8_numpy(new_patch_tensor)
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i])
            timer['writer'] += time.time() - t0

            if emit_event:
                emit_event(
                    status="running",
                    progress=int(step / total_steps * 100),
                    message=(
                        f"Processing {idx} ~ "
                        f"{min(idx + self.batch_size, len(src_refs))} / {len(src_refs)}"
                    ),
                )

        return timer
        
    def get_reinhard_stats(self, image: np.ndarray):
        """
        image: RGB image, shape (B, C, H, W), uint8 or float
        return:
            means: [L_mean, a_mean, b_mean]
            stds:  [L_std,  a_std,  b_std]
        """
        stats = self._get_reinhard_stats_tensor(self._to_chw_float_tensor(image))

        return (
            stats.means.detach().cpu().numpy(),
            stats.stds.detach().cpu().numpy(),
        )

    def _get_reinhard_stats_tensor(self, image: torch.Tensor) -> ReinhardStats:
        lab = self._rgb_to_lab(image)
        flattened = lab.reshape(-1, 3)

        return ReinhardStats(
            means=flattened.mean(dim=0),
            stds=flattened.std(dim=0, unbiased=False),
        )

    def _transform_with_stats_tensor(
        self,
        image: torch.Tensor,
        source_stats: ReinhardStats,
        target_stats: ReinhardStats,
    ) -> torch.Tensor:
        lab = self._rgb_to_lab(image)
        lab = (
            (lab - source_stats.means)
            / self._safe_stds(source_stats.stds)
            * target_stats.stds
            + target_stats.means
        )

        lab[..., 0] = lab[..., 0].clamp(0, 100)
        lab[..., 1:] = lab[..., 1:].clamp(-128, 127)

        return self._lab_to_rgb(lab)
    
    def transform_image(
        self, 
        image: np.ndarray, 
        target_means: np.ndarray,
        target_stds: np.ndarray,
        src_means: np.ndarray,
        src_stds: np.ndarray, 
    ) -> np.ndarray:
        image = self._to_chw_float_tensor(image)
        source_stats = ReinhardStats(
            means=self._stats_tensor(src_means, "src_means"),
            stds=self._stats_tensor(src_stds, "src_stds"),
        )
        target_stats = ReinhardStats(
            means=self._stats_tensor(target_means, "target_means"),
            stds=self._stats_tensor(target_stds, "target_stds"),
        )

        output = self._transform_with_stats_tensor(
            image,
            source_stats=source_stats,
            target_stats=target_stats,
        )

        return output.detach().cpu().numpy()

    def _to_chw_float_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if image.ndim != 4:
            raise PipelineInputShapeError(f"이미지는 4차원이어야 합니다. 입력 차원: {image.ndim}D")
        if image.shape[1] != 3:
            raise PipelineInputShapeError(
                f"이미지는 CHW 형식의 3채널이어야 합니다. 입력 shape: {image.shape}"
            )

        should_scale = np.issubdtype(image.dtype, np.integer) or image.max() > 1.0
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
            device=self.device,
            dtype=torch.float32,
        )
        if should_scale:
            tensor = tensor / 255.0

        return tensor

    def _stats_tensor(self, value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.shape != (3,):
            raise PipelineInputShapeError(
                f"{name}은 shape (3,)이어야 합니다. 입력 shape: {tuple(tensor.shape)}"
            )

        return tensor

    def _safe_stds(self, stds: torch.Tensor) -> torch.Tensor:
        return torch.where(stds.abs() < 1e-6, torch.ones_like(stds), stds)

    def _to_chw_uint8_numpy(self, image: torch.Tensor) -> np.ndarray:
        return (
            image.round()
            .clamp(0, 255)
            .to(dtype=torch.uint8)
            .detach()
            .cpu()
            .numpy()
        )

    def _color_constants(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if key not in self._color_constants_cache:
            rgb_to_xyz = torch.tensor(
                [
                    [0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041],
                ],
                device=device,
                dtype=dtype,
            )
            xyz_to_rgb = torch.tensor(
                [
                    [3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252],
                ],
                device=device,
                dtype=dtype,
            )
            white_point = torch.tensor(
                [0.95047, 1.0, 1.08883],
                device=device,
                dtype=dtype,
            )
            self._color_constants_cache[key] = (
                rgb_to_xyz,
                xyz_to_rgb,
                white_point,
            )

        return self._color_constants_cache[key]

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=torch.float32)
        rgb = image.clamp(0, 1)
        linear_rgb = torch.where(
            rgb > 0.04045,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92,
        )

        rgb_to_xyz, _, white_point = self._color_constants(image.device, image.dtype)
        xyz = torch.matmul(linear_rgb.permute(0, 2, 3, 1), rgb_to_xyz.T)
        xyz = xyz / white_point

        delta = 6.0 / 29.0
        xyz_f = torch.where(
            xyz > delta ** 3,
            torch.pow(xyz, 1.0 / 3.0),
            xyz / (3 * delta ** 2) + 4.0 / 29.0,
        )

        x, y, z = xyz_f.unbind(dim=-1)

        return torch.stack(
            [
                116.0 * y - 16.0,
                500.0 * (x - y),
                200.0 * (y - z),
            ],
            dim=-1,
        )

    def _lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        lab = lab.to(dtype=torch.float32)
        l_channel, a_channel, b_channel = lab.unbind(dim=-1)
        fy = (l_channel + 16.0) / 116.0
        fx = fy + a_channel / 500.0
        fz = fy - b_channel / 200.0

        delta = 6.0 / 29.0
        fxyz = torch.stack([fx, fy, fz], dim=-1)
        xyz = torch.where(
            fxyz > delta,
            torch.pow(fxyz, 3.0),
            3 * delta ** 2 * (fxyz - 4.0 / 29.0),
        )
        _, xyz_to_rgb, white_point = self._color_constants(lab.device, lab.dtype)
        xyz = xyz * white_point
        linear_rgb = torch.matmul(xyz, xyz_to_rgb.T)

        rgb = torch.where(
            linear_rgb > 0.0031308,
            1.055 * torch.pow(linear_rgb.clamp_min(0), 1.0 / 2.4) - 0.055,
            12.92 * linear_rgb,
        )

        return rgb.clamp(0, 1).permute(0, 3, 1, 2) * 255.0
