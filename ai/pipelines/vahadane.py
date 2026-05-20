from collections import defaultdict
import logging
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

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

        h, w, _ = source_rgb.shape
        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)

        scale = self.target_max_conc / (self.source_max_conc + self.eps)
        normalized_conc = source_conc * scale[:, None]

        normalized_od = self.target_stain_matrix @ normalized_conc
        normalized_rgb = self.Io * np.exp(-normalized_od)
        normalized_rgb = normalized_rgb.T.reshape(h, w, 3)

        return np.clip(normalized_rgb, 0, 255).astype(np.uint8)

    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        rgb = np.clip(rgb, 1, self.Io)
        return -np.log((rgb + self.eps) / self.Io)

    def _prepare_od(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 3:
            if rgb.shape[2] != 3:
                raise ValueError(f"Expected RGB image with shape [H, W, 3], got {rgb.shape}")
        elif rgb.ndim == 4:
            if rgb.shape[-1] != 3:
                raise ValueError(f"Expected RGB patches with shape [N, H, W, 3], got {rgb.shape}")
        else:
            raise ValueError(f"Expected RGB image or patches, got {rgb.ndim}D array")

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
    ):
        super().__init__(logger)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.max_sample_patches = int(max_sample_patches)

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
        normalizer = VahadaneNormalizer()

        fit_patch_sampler = PatchSampler(patch_size=self.patch_size, strict_mpp_check=False)

        source_refs = fit_patch_sampler.sample(
            src_wsi_handle,
            mode="training",
            max_patches=self.max_sample_patches,
            save_debug=False,
        )
        target_refs = fit_patch_sampler.sample(
            target_wsi_handle,
            mode="training",
            max_patches=self.max_sample_patches,
            save_debug=False,
        )

        source_patches = np.stack([load_patch(ref).img for ref in source_refs])
        target_patches = np.stack([load_patch(ref).img for ref in target_refs])

        metric = Metric(
            use_ssim="ssim" in metrics,
            use_psnr="psnr" in metrics,
            use_fid="fid" in metrics,
            target_patch=target_patches,
        )

        normalizer.fit(
            source_patches.transpose(0, 2, 3, 1),
            target_patches.transpose(0, 2, 3, 1),
        )

        grid_sampler = GridSampler(patch_size=self.patch_size, read_level=0)
        src_refs = grid_sampler.sample(src_wsi_handle)

        writer = MultiZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[0][0],
            src_wsi_handle.level_dimensions[0][1],
            level_downsample=src_wsi_handle.level_downsamples[0],
            tile_size=src_refs[0].width,
        )

        timer = defaultdict(float)
        total_steps = len(range(0, len(src_refs), self.batch_size))
        step = 0

        for idx in tqdm(range(0, len(src_refs), self.batch_size)):
            step += 1
            t0 = time.time()
            batch_ref = src_refs[idx:idx + self.batch_size]
            patches = [load_patch(ref) for ref in batch_ref]
            timer["load"] += time.time() - t0

            patches = np.stack([patch.img for patch in patches], axis=0)
            patches_hwc = patches.transpose(0, 2, 3, 1)

            t0 = time.time()
            new_patches = np.stack([
                normalizer.normalize(patch).transpose(2, 0, 1)
                for patch in patches_hwc
            ])
            timer["transform"] += time.time() - t0

            metric.evaluate(patches, new_patches)

            t0 = time.time()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i].astype(np.uint8))
            timer["writer"] += time.time() - t0

            if emit_event:
                emit_event(
                    status="running",
                    progress=int(step / total_steps * 100),
                    message=f"Processing {idx} ~ {idx + self.batch_size} / {len(src_refs)}",
                )

        self.logger.info("Finish Vahadane normalize")
        self.logger.info(
            f"Elapsed time: load({timer['load']:.4f}s), "
            f"transform({timer['transform']:.4f}s), writer({timer['writer']:.4f}s)"
        )

        output_path = writer.finalize()
        writer.close()

        return PipelineResult(
            output_path=output_path,
            scores=metric.finalize(),
            thumbnail_path=None,
        )
