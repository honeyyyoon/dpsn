from collections import defaultdict
import logging
import math
from pathlib import Path
import time

import numpy as np

from ai.metrics.metric import Metric
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter


class VahadaneNormalizer:
    def __init__(
        self,
        beta: float = 0.15,
        Io: int = 240,
        sparsity_lambda: float = 0.01,
        max_iter: int = 100,
        eps: float = 1e-8,
        max_fit_pixels: int = 200_000,
        random_seed: int = 0,
    ):
        self.beta = beta
        self.Io = Io
        self.sparsity_lambda = sparsity_lambda
        self.max_iter = max_iter
        self.eps = eps
        self.max_fit_pixels = max_fit_pixels
        self.random_seed = random_seed

        self.source_stain_matrix: np.ndarray | None = None
        self.source_max_conc: np.ndarray | None = None
        self.target_stain_matrix: np.ndarray | None = None
        self.target_max_conc: np.ndarray | None = None

    def fit(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> None:
        self._validate_rgb(source_rgb)
        self._validate_rgb(target_rgb)

        self.source_stain_matrix = self._estimate_stain_matrix(source_rgb)
        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)
        self.source_max_conc = np.percentile(source_conc, 99, axis=1)

        self.target_stain_matrix = self._estimate_stain_matrix(target_rgb)
        target_conc = self._estimate_concentrations(target_rgb, self.target_stain_matrix)
        self.target_max_conc = np.percentile(target_conc, 99, axis=1)

    def normalize(self, source_rgb: np.ndarray) -> np.ndarray:
        if (
            self.source_stain_matrix is None
            or self.source_max_conc is None
            or self.target_stain_matrix is None
            or self.target_max_conc is None
        ):
            raise RuntimeError("Call fit(source_patches, target_patches) before normalize().")

        self._validate_rgb(source_rgb)
        original_shape = source_rgb.shape
        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)

        scale = self.target_max_conc / (self.source_max_conc + self.eps)
        normalized_conc = source_conc * scale[:, None]

        normalized_od = self.target_stain_matrix @ normalized_conc
        normalized_rgb = self.Io * np.exp(-normalized_od)
        normalized_rgb = normalized_rgb.T.reshape(original_shape)

        return np.clip(normalized_rgb, 0, 255).astype(np.uint8)

    def _validate_rgb(self, rgb: np.ndarray) -> None:
        if rgb.ndim not in {3, 4}:
            raise ValueError(f"Expected RGB image or batch, got {rgb.ndim}D array")
        if rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB data with channels-last shape, got {rgb.shape}")

    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        rgb = np.clip(rgb, 1, self.Io)
        return -np.log((rgb + self.eps) / self.Io)

    def _prepare_od(self, rgb: np.ndarray) -> np.ndarray:
        self._validate_rgb(rgb)

        od = self._rgb_to_od(rgb).reshape(-1, 3)
        od_norm = np.linalg.norm(od, axis=1)
        od = od[od_norm > self.beta]

        if len(od) == 0:
            raise ValueError("No valid tissue pixels found. Try lowering beta.")

        if len(od) > self.max_fit_pixels:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(len(od), size=self.max_fit_pixels, replace=False)
            od = od[indices]

        return od.T

    def _estimate_stain_matrix(self, rgb: np.ndarray) -> np.ndarray:
        od = self._prepare_od(rgb)
        stain_matrix, concentrations = self._initialize_nmf(od)

        for _ in range(self.max_iter):
            numerator = stain_matrix.T @ od
            denominator = stain_matrix.T @ stain_matrix @ concentrations
            denominator = denominator + self.sparsity_lambda + self.eps
            concentrations *= numerator / denominator

            numerator = od @ concentrations.T
            denominator = stain_matrix @ concentrations @ concentrations.T + self.eps
            stain_matrix *= numerator / denominator
            stain_matrix = self._normalize_columns(stain_matrix)

        return self._order_stains(stain_matrix)

    def _initialize_nmf(self, od: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, vh = np.linalg.svd(od.T, full_matrices=False)
        stain_matrix = np.abs(vh[:2].T)

        if stain_matrix.shape != (3, 2) or np.linalg.matrix_rank(stain_matrix) < 2:
            stain_matrix = np.array(
                [
                    [0.65, 0.07],
                    [0.70, 0.99],
                    [0.29, 0.11],
                ],
                dtype=np.float32,
            )

        stain_matrix = self._normalize_columns(stain_matrix)
        concentrations = self._estimate_concentrations_from_od(od, stain_matrix)
        concentrations = np.maximum(concentrations, self.eps)
        return stain_matrix, concentrations

    def _estimate_concentrations(
        self,
        rgb: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        od = self._rgb_to_od(rgb).reshape(-1, 3).T
        return self._estimate_concentrations_from_od(od, stain_matrix)

    def _estimate_concentrations_from_od(
        self,
        od: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od, rcond=None)
        return np.maximum(concentrations, 0)

    def _normalize_columns(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + self.eps)

    def _order_stains(self, stain_matrix: np.ndarray) -> np.ndarray:
        if stain_matrix[0, 0] >= stain_matrix[0, 1]:
            return stain_matrix
        return stain_matrix[:, [1, 0]]


class Vahadane(ModelPipeline):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        batch_size: int = 64,
        patch_size: int = 256,
        max_sample_patches: int = 64,
        max_iteration: int = 128,
    ):
        super().__init__(logger or logging.getLogger(__name__))
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
        emit_event=None,
    ) -> PipelineResult:
        if target_img_path is None:
            raise ValueError("Vahadane pipeline requires a target image.")

        self.logger.info("Run Vahadane")
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)
        self._log_wsi_info("source", src_wsi_handle)
        self._log_wsi_info("target", target_wsi_handle)

        normalizer = VahadaneNormalizer()

        fit_patch_sampler = PatchSampler(patch_size=self.patch_size, strict_mpp_check=False)

        source_patches = self._sample_patch_images(
            patch_sampler=fit_patch_sampler,
            wsi_handle=src_wsi_handle,
            label="source",
        )
        target_patches = self._sample_patch_images(
            patch_sampler=fit_patch_sampler,
            wsi_handle=target_wsi_handle,
            label="target",
        )

        metric = self._build_metric(metrics, target_patches)

        self.logger.info("Fit Vahadane normalizer")
        normalizer.fit(
            self._to_hwc_batch(source_patches),
            self._to_hwc_batch(target_patches),
        )

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
            normalizer=normalizer,
            emit_event=emit_event,
        )

        self.logger.info("Finish Vahadane normalize")
        self.logger.info(
            f"Elapsed time: load({timer['load']:.4f}s), "
            f"transform({timer['transform']:.4f}s), "
            f"metric({timer['metric']:.4f}s), writer({timer['writer']:.4f}s)"
        )

        output_path = writer.finalize()
        writer.close()

        return PipelineResult(
            output_path=output_path,
            scores=metric.finalize(),
            thumbnail_path=None,
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
            mode="training",
            max_patches=self.max_sample_patches,
            save_debug=False,
        )
        images = np.stack([load_patch(ref).img for ref in refs], axis=0)
        self.logger.info(f"Patches: {len(images)}")

        return images

    def _build_metric(self, metrics: list[str], target_patches: np.ndarray) -> Metric:
        return Metric(
            use_ssim="ssim" in metrics,
            use_psnr="psnr" in metrics,
            use_fid="fid" in metrics,
            target_patch=target_patches,
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
    ) -> MultiZarrWSIWriter:
        return MultiZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[level][0],
            src_wsi_handle.level_dimensions[level][1],
            level_downsample=src_wsi_handle.level_downsamples[level],
            tile_size=src_refs[0].width,
        )

    def _process_batches(
        self,
        src_refs,
        writer: MultiZarrWSIWriter,
        metric: Metric,
        normalizer: VahadaneNormalizer,
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
            timer["load"] += time.time() - t0

            t0 = time.time()
            new_patches = self._normalize_batch(normalizer, patches)
            timer["transform"] += time.time() - t0

            t0 = time.time()
            metric.evaluate(patches, new_patches)
            timer["metric"] += time.time() - t0

            t0 = time.time()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i].astype(np.uint8))
            timer["writer"] += time.time() - t0

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

    def _normalize_batch(
        self,
        normalizer: VahadaneNormalizer,
        patches: np.ndarray,
    ) -> np.ndarray:
        normalized = normalizer.normalize(self._to_hwc_batch(patches))

        return self._to_chw_batch(normalized)

    def _to_hwc_batch(self, patches: np.ndarray) -> np.ndarray:
        if patches.ndim == 3:
            patches = patches[np.newaxis, ...]
        if patches.ndim != 4:
            raise ValueError(f"Expected patch batch with 4D shape, got {patches.ndim}D")
        if patches.shape[1] != 3:
            raise ValueError(f"Expected CHW patch batch with 3 channels, got {patches.shape}")

        return patches.transpose(0, 2, 3, 1)

    def _to_chw_batch(self, patches: np.ndarray) -> np.ndarray:
        if patches.ndim == 3:
            patches = patches[np.newaxis, ...]
        if patches.ndim != 4:
            raise ValueError(f"Expected patch batch with 4D shape, got {patches.ndim}D")
        if patches.shape[-1] != 3:
            raise ValueError(f"Expected HWC patch batch with 3 channels, got {patches.shape}")

        return patches.transpose(0, 3, 1, 2)
