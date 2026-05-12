from collections import defaultdict
import logging
from pathlib import Path
import time

import numpy as np
from tqdm import  tqdm

from ai.metrics.base import Metric
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import open_wsi_handle, load_patch
from ai.wsi.writer import ZarrWSIWriter

class MacenkoNormalizer:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        Io: int = 255,
        eps: float = 1e-8,
    ):
        self.alpha = alpha
        self.beta = beta
        self.Io = Io
        self.eps = eps

        self.target_stain_matrix: np.ndarray | None = None
        self.target_max_conc: np.ndarray | None = None

    def fit(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> None:
        stain_matrix = self._estimate_stain_matrix(target_rgb)
        concentrations = self._estimate_concentrations(target_rgb, stain_matrix)

        self.target_stain_matrix = stain_matrix
        self.target_max_conc = np.percentile(concentrations, 99, axis=1)
        self.source_stain_matrix = self._estimate_stain_matrix(source_rgb)

    def normalize(self, source_rgb: np.ndarray) -> np.ndarray:
        if self.target_stain_matrix is None or self.target_max_conc is None:
            raise RuntimeError("Call fit(target_rgb) before normalize(source_rgb).")

        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)

        source_max_conc = np.percentile(source_conc, 99, axis=1)
        scale = self.target_max_conc / (source_max_conc + self.eps)

        normalized_conc = source_conc * scale[:, None]

        h, w, _ = source_rgb.shape

        normalized_od = self.target_stain_matrix @ normalized_conc
        normalized_rgb = self.Io * np.exp(-normalized_od)
        normalized_rgb = normalized_rgb.T.reshape(h, w, 3)

        return np.clip(normalized_rgb, 0, 255).astype(np.uint8)

    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        rgb = np.maximum(rgb, 1)
        return -np.log((rgb + self.eps) / self.Io)

    def _estimate_stain_matrix(self, rgb: np.ndarray) -> np.ndarray:
        od = self._rgb_to_od(rgb).reshape(-1, 3)

        # 배경 제거
        od = self._rgb_to_od(rgb).reshape(-1, 3)

        od_norm = np.linalg.norm(od, axis=1)
        od = od[od_norm > self.beta]

        if len(od) == 0:
            raise ValueError("No valid tissue pixels found. Try lowering beta.")

        # OD covariance의 principal directions
        _, _, vh = np.linalg.svd(od, full_matrices=False)
        top_vectors = vh[:2].T  # [3, 2]

        projected = od @ top_vectors

        angles = np.arctan2(projected[:, 1], projected[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        v1 = top_vectors @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v2 = top_vectors @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # Hematoxylin / Eosin 순서 안정화
        if v1[0] > v2[0]:
            stain_matrix = np.stack([v1, v2], axis=1)
        else:
            stain_matrix = np.stack([v2, v1], axis=1)

        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0, keepdims=True) + self.eps)

        return stain_matrix  # [3, 2]

    def _estimate_concentrations(
        self,
        rgb: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        od = self._rgb_to_od(rgb).reshape(-1, 3).T  # [3, N]

        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od, rcond=None)

        return concentrations  # [2, N]


class Macenko(ModelPipeline):
    def __init__(self, logger: logging.Logger | None = None):
        super().__init__(logger)
    def run(
        self, 
        src_img_path: Path,
        result_path: Path,
        target_img_path: Path | None,
        metrics: dict[str, Metric],
        emit_event=None
    ) -> PipelineResult:

        if target_img_path is None:
            raise ValueError("Macenko pipeline requires a target image.")
        
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)
        macenko = MacenkoNormalizer()

        patch_sampler = PatchSampler(patch_size=64, strict_mpp_check=False)
        source_refs = patch_sampler.sample(src_wsi_handle,mode="training", max_patches=16, save_debug=False)
        target_refs = patch_sampler.sample(target_wsi_handle,mode="training", max_patches=16, save_debug=False)

        source_patches = [load_patch(ref).img.transpose(1, 2, 0) for ref in source_refs]
        target_patches = [load_patch(ref).img.transpose(1, 2, 0) for ref in target_refs]
        for idx in range(len(source_patches)):
            macenko.fit(source_patches[idx], target_patches[idx])

        grid_sampler = GridSampler(patch_size=64, read_level=0)
        src_refs = grid_sampler.sample(src_wsi_handle)

        writer = ZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[0][0], 
            src_wsi_handle.level_dimensions[0][1],
            level_downsample=src_wsi_handle.level_downsamples[0],
            tile_size = src_refs[0].width
        )

        timer = defaultdict(float)
        scores = defaultdict(float)
        self.batch_size = 64

        iter = len(range(0, len(src_refs), self.batch_size))
        step = 0
        for idx in tqdm(range(0, len(src_refs), self.batch_size)):
            step += 1
            t0 = time.time()
            batch_ref = src_refs[idx:idx + self.batch_size]
            patches = [load_patch(ref) for ref in batch_ref]
            timer['load'] += time.time() - t0

            patches = [patch.img.transpose(1, 2, 0) for patch in patches]

            t0 = time.time()
            new_patches = [macenko.normalize(patch).transpose(2, 0, 1) for patch in patches]
            timer['transform'] += time.time() - t0

            patches = np.stack(patches, axis=0).transpose(0, 3, 1, 2)
            new_patches = np.stack(new_patches, axis=0)

            t0 = time.time()
            for key, metric in metrics.items():
                scores[key] += metric.evaluate(patches, new_patches)

            t0 = time.time()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i].astype(np.uint8))
            timer['writer'] += time.time() - t0
            if emit_event:
                print(step, iter)
                emit_event(status="running", progress=int(step / iter * 100), message=f"Processing {idx} ~ {idx + self.batch_size} / {len(src_refs)}")

        for key, score in scores.items():
            scores[key] /= iter
        scores = dict(scores)

        output_path = writer.finalize()
        writer.close()

        return PipelineResult(
            output_path=output_path,
            scores=scores,
            thumbnail_path=output_path,
        )
