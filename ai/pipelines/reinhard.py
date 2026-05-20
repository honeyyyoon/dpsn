from collections import defaultdict
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time

import numpy as np
from skimage import color

from ai.metrics.metric import Metric
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import ZarrWSIWriter


@dataclass(frozen=True)
class ReinhardStats:
    means: np.ndarray
    stds: np.ndarray


class Reinhard(ModelPipeline):
    target_sampler: PatchSampler

    def __init__(
        self, 
        logger: logging.Logger,
        batch_size: int = 64,
        patch_size: int = 256,
        max_sample_patches: int = 16,
        max_iteration: int = 16
    ):
        super().__init__(logger=logger)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.max_sample_patches = int(max_sample_patches)
        self.max_iteration = int(max_iteration)
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
        
        if target_img_path is None:
            raise ValueError("Reinhard needs a target image.")
        
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)
        self._log_wsi_info("source", src_wsi_handle)
        self._log_wsi_info("target", target_wsi_handle)

        patch_sampler = PatchSampler(strict_mpp_check=False)
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

        writer = self._build_writer(result_path, src_wsi_handle, src_refs, level)
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

        return PipelineResult(
            output_path=output_path,
            scores=metric.finalize(),
            thumbnail_path=output_path,
        )

    def _validate_config(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.max_sample_patches <= 0:
            raise ValueError(
                f"max_sample_patches must be > 0, got {self.max_sample_patches}"
            )
        if self.max_iteration <= 0:
            raise ValueError(f"max_iteration must be > 0, got {self.max_iteration}")

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
        means, stds = self.get_reinhard_stats(images)
        self.logger.info(
            f"{label.capitalize()} stat: "
            f"means={means.round(2)}, stds={stds.round(2)}"
        )

        return ReinhardStats(means=means, stds=stds)

    def _build_metric(self, metrics: list[str], target_images: np.ndarray) -> Metric:
        return Metric(
            use_ssim="ssim" in metrics,
            use_psnr="psnr" in metrics,
            use_fid="fid" in metrics,
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
            raise ValueError(
                "Image is too big! "
                f"Expected iteration: {expected_iterations}, "
                f"Max iteration: {self.max_iteration}"
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
    ) -> ZarrWSIWriter:
        return ZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[0][0],
            src_wsi_handle.level_dimensions[0][1],
            level_downsample=src_wsi_handle.level_downsamples[level],
            tile_size=src_refs[0].width,
        )

    def _process_batches(
        self,
        src_refs,
        writer: ZarrWSIWriter,
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
            new_patches = self._transform_with_stats(
                patches,
                source_stats=source_stats,
                target_stats=target_stats,
            )
            timer['transform'] += time.time() - t0

            t0 = time.time()
            metric.evaluate(patches, new_patches)
            timer['metric'] += time.time() - t0

            t0 = time.time()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i].astype(np.uint8))
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
        image = self._to_hwc_batch(image)

        lab = color.rgb2lab(image / 255.0)  # L: [0,100], a/b: roughly [-128,127]

        means = lab.reshape(-1, 3).mean(axis=0)
        stds = lab.reshape(-1, 3).std(axis=0)

        return means, stds

    def _transform_with_stats(
        self,
        image: np.ndarray,
        source_stats: ReinhardStats,
        target_stats: ReinhardStats,
    ) -> np.ndarray:
        return self.transform_image(
            image,
            target_means=target_stats.means,
            target_stds=target_stats.stds,
            src_means=source_stats.means,
            src_stds=source_stats.stds,
        )
    
    def transform_image(
        self, 
        image: np.ndarray, 
        target_means: np.ndarray,
        target_stds: np.ndarray,
        src_means: np.ndarray,
        src_stds: np.ndarray, 
    ) -> np.ndarray:
        image = self._to_hwc_batch(image)
        target_means = self._validate_stats_vector(target_means, "target_means")
        target_stds = self._validate_stats_vector(target_stds, "target_stds")
        src_means = self._validate_stats_vector(src_means, "src_means")
        src_stds = self._validate_stats_vector(src_stds, "src_stds")

        lab = color.rgb2lab(image / 255.0)
        lab = (lab - src_means) / self._safe_stds(src_stds) * target_stds + target_means
        
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        lab[..., 1:] = np.clip(lab[..., 1:], -128, 127)

        image = np.clip(color.lab2rgb(lab) * 255.0, 0, 255)
        image = image.transpose([0, 3, 1, 2])

        return image

    def _to_hwc_batch(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if image.ndim != 4:
            raise ValueError(f"Image should be 4D, but got {image.ndim}D")
        if image.shape[1] != 3:
            raise ValueError(f"Image should have 3 channels in CHW format, got {image.shape}")

        return image.transpose([0, 2, 3, 1])

    def _validate_stats_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"{name} should have shape (3,), got {value.shape}")

        return value

    def _safe_stds(self, stds: np.ndarray) -> np.ndarray:
        return np.where(stds == 0, 1.0, stds)
