from collections import defaultdict
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


class VahadaneError(RuntimeError):
    """Base class for Vahadane pipeline errors."""


class VahadaneNotFittedError(VahadaneError):
    """Raised when Vahadane normalization is used before fit."""


class MissingTargetImageError(VahadaneError):
    """Raised when Vahadane normalization is run without a target image."""


class VahadaneImageTooLargeError(VahadaneError):
    """Raised when the selected WSI would exceed the configured iteration limit."""


class NoTissuePixelsError(VahadaneError):
    """Raised when stain fitting cannot find valid tissue pixels."""


class VahadaneNormalizer:
    DEFAULT_STAIN_MATRIX = torch.tensor(
        [
            [0.65, 0.07],
            [0.70, 0.99],
            [0.29, 0.11],
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        beta: float = 0.15,
        Io: int = 240,
        sparsity_lambda: float = 0.01,
        max_iter: int = 100,
        eps: float = 1e-8,
        max_fit_pixels: int = 200_000,
        random_seed: int = 0,
        device: str | torch.device | None = None,
    ):
        self.beta = beta
        self.Io = Io
        self.sparsity_lambda = sparsity_lambda
        self.max_iter = max_iter
        self.eps = eps
        self.max_fit_pixels = max_fit_pixels
        self.random_seed = random_seed
        self.device = self._select_device(device)

        self.source_stain_matrix: torch.Tensor | None = None
        self.source_max_conc: torch.Tensor | None = None
        self.target_stain_matrix: torch.Tensor | None = None
        self.target_max_conc: torch.Tensor | None = None

    def fit(
        self,
        source_rgb: np.ndarray | torch.Tensor,
        target_rgb: np.ndarray | torch.Tensor,
    ) -> None:
        source_rgb = self._to_hwc_float_tensor(source_rgb)
        target_rgb = self._to_hwc_float_tensor(target_rgb)

        self.source_stain_matrix = self._estimate_stain_matrix(source_rgb)
        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)
        self.source_max_conc = torch.quantile(source_conc, 0.99, dim=1)

        self.target_stain_matrix = self._estimate_stain_matrix(target_rgb)
        target_conc = self._estimate_concentrations(target_rgb, self.target_stain_matrix)
        self.target_max_conc = torch.quantile(target_conc, 0.99, dim=1)

    def normalize(self, source_rgb: np.ndarray | torch.Tensor) -> np.ndarray:
        normalized = self.normalize_tensor(source_rgb)
        return normalized.detach().cpu().numpy()

    def normalize_tensor(self, source_rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
        if (
            self.source_stain_matrix is None
            or self.source_max_conc is None
            or self.target_stain_matrix is None
            or self.target_max_conc is None
        ):
            raise VahadaneNotFittedError(
                "normalize()를 호출하기 전에 fit(source_patches, target_patches)를 먼저 호출해야 합니다."
            )

        source_rgb = self._to_hwc_float_tensor(source_rgb)
        original_shape = source_rgb.shape
        source_conc = self._estimate_concentrations(source_rgb, self.source_stain_matrix)

        scale = self.target_max_conc / (self.source_max_conc + self.eps)
        normalized_conc = source_conc * scale[:, None]

        normalized_od = self.target_stain_matrix @ normalized_conc
        normalized_rgb = self.Io * torch.exp(-normalized_od)
        normalized_rgb = normalized_rgb.T.reshape(original_shape)

        return normalized_rgb.clamp(0, 255).round().to(dtype=torch.uint8)

    def _select_device(self, device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _validate_rgb(self, rgb: np.ndarray) -> None:
        if rgb.ndim not in {3, 4}:
            raise PipelineInputShapeError(
                f"RGB 이미지 또는 batch가 필요합니다. 입력 배열 차원: {rgb.ndim}D"
            )
        if rgb.shape[-1] != 3:
            raise PipelineInputShapeError(
                f"RGB 데이터는 channels-last shape이어야 합니다. 입력 shape: {rgb.shape}"
            )

    def _to_hwc_float_tensor(self, rgb: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(rgb, np.ndarray):
            self._validate_rgb(rgb)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb))
        elif isinstance(rgb, torch.Tensor):
            tensor = rgb
            if tensor.ndim not in {3, 4}:
                raise PipelineInputShapeError(
                    f"RGB 이미지 또는 batch가 필요합니다. 입력 tensor 차원: {tensor.ndim}D"
                )
            if tensor.shape[-1] != 3:
                raise PipelineInputShapeError(
                    f"RGB 데이터는 channels-last shape이어야 합니다. 입력 shape: {tuple(tensor.shape)}"
                )
        else:
            raise PipelineInputShapeError(
                f"rgb는 numpy.ndarray 또는 torch.Tensor 타입이어야 합니다. 입력 타입: {type(rgb).__name__}"
            )

        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if float(tensor.max().detach().cpu().item()) <= 1.0:
            tensor = tensor * 255.0

        return tensor

    def _rgb_to_od(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(device=self.device, dtype=torch.float32)
        rgb = rgb.clamp(1, self.Io)
        return -torch.log((rgb + self.eps) / self.Io)

    def _prepare_od(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = self._to_hwc_float_tensor(rgb)

        od = self._rgb_to_od(rgb).reshape(-1, 3)
        od_norm = torch.linalg.vector_norm(od, dim=1)
        od = od[od_norm > self.beta]

        if len(od) == 0:
            raise NoTissuePixelsError("유효한 조직 픽셀을 찾지 못했습니다. beta 값을 낮춰보세요.")

        if len(od) > self.max_fit_pixels:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(len(od), size=self.max_fit_pixels, replace=False)
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            od = od[indices]

        return od.T

    def _estimate_stain_matrix(self, rgb: torch.Tensor) -> torch.Tensor:
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

    def _initialize_nmf(self, od: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, vh = torch.linalg.svd(od.T, full_matrices=False)
        stain_matrix = torch.abs(vh[:2].T)

        if stain_matrix.shape != (3, 2) or not torch.isfinite(stain_matrix).all():
            stain_matrix = self.DEFAULT_STAIN_MATRIX.to(self.device)

        stain_matrix = self._normalize_columns(stain_matrix)
        concentrations = self._estimate_concentrations_from_od(od, stain_matrix)
        return stain_matrix, concentrations.clamp_min(self.eps)

    def _estimate_concentrations(
        self,
        rgb: torch.Tensor,
        stain_matrix: torch.Tensor,
    ) -> torch.Tensor:
        od = self._rgb_to_od(rgb).reshape(-1, 3).T
        return self._estimate_concentrations_from_od(od, stain_matrix)

    def _estimate_concentrations_from_od(
        self,
        od: torch.Tensor,
        stain_matrix: torch.Tensor,
    ) -> torch.Tensor:
        gram = stain_matrix.T @ stain_matrix
        det = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
        det = torch.clamp(det, min=self.eps)
        gram_inv = torch.stack(
            [
                torch.stack([gram[1, 1], -gram[0, 1]]),
                torch.stack([-gram[1, 0], gram[0, 0]]),
            ],
        ) / det
        concentrations = gram_inv @ stain_matrix.T @ od
        return concentrations.clamp_min(0)

    def _normalize_columns(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix / (torch.linalg.vector_norm(matrix, dim=0, keepdim=True) + self.eps)

    def _order_stains(self, stain_matrix: torch.Tensor) -> torch.Tensor:
        if float(stain_matrix[0, 0].detach().cpu().item()) >= float(
            stain_matrix[0, 1].detach().cpu().item()
        ):
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
        device: str | torch.device | None = None,
    ):
        super().__init__(logger or logging.getLogger(__name__))
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.max_sample_patches = int(max_sample_patches)
        self.max_iteration = int(max_iteration)
        self.device = self._select_device(device)
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
            raise MissingTargetImageError("Vahadane 파이프라인에는 타겟 이미지가 필요합니다.")

        self.logger.info("Run Vahadane")
        self.logger.info(f"Use Vahadane device: {self.device}")
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)
        self._log_wsi_info("source", src_wsi_handle)
        self._log_wsi_info("target", target_wsi_handle)

        normalizer = VahadaneNormalizer(device=self.device)

        fit_patch_sampler = PatchSampler(
            patch_size=self.patch_size,
            training_tissue_threshold=0.3,
            strict_mpp_check=False
        )

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
            self._to_hwc_float_tensor(source_patches),
            self._to_hwc_float_tensor(target_patches),
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
            raise VahadaneImageTooLargeError(
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
            patch_tensor = self._to_chw_tensor(patches)
            new_patch_tensor = self._normalize_batch(normalizer, patch_tensor)
            timer["transform"] += time.time() - t0

            t0 = time.time()
            metric.evaluate_torch(patch_tensor, new_patch_tensor)
            timer["metric"] += time.time() - t0

            t0 = time.time()
            new_patches = new_patch_tensor.detach().cpu().numpy()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i])
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
        patches: torch.Tensor,
    ) -> torch.Tensor:
        normalized = normalizer.normalize_tensor(patches.permute(0, 2, 3, 1))

        return normalized.permute(0, 3, 1, 2)

    def _to_chw_tensor(self, patches: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(patches, np.ndarray):
            if patches.ndim == 3:
                patches = patches[np.newaxis, ...]
            if patches.ndim != 4:
                raise PipelineInputShapeError(
                    f"patch batch는 4차원 shape이어야 합니다. 입력 차원: {patches.ndim}D"
                )
            if patches.shape[1] != 3:
                raise PipelineInputShapeError(
                    f"CHW patch batch는 3채널이어야 합니다. 입력 shape: {patches.shape}"
                )
            tensor = torch.from_numpy(np.ascontiguousarray(patches))
        elif isinstance(patches, torch.Tensor):
            tensor = patches
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise PipelineInputShapeError(
                    f"patch batch는 4차원 shape이어야 합니다. 입력 차원: {tensor.ndim}D"
                )
            if tensor.shape[1] != 3:
                raise PipelineInputShapeError(
                    f"CHW patch batch는 3채널이어야 합니다. 입력 shape: {tuple(tensor.shape)}"
                )
        else:
            raise PipelineInputShapeError(
                f"patches는 numpy.ndarray 또는 torch.Tensor 타입이어야 합니다. 입력 타입: {type(patches).__name__}"
            )

        return tensor.to(device=self.device)

    def _to_hwc_float_tensor(self, patches: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = self._to_chw_tensor(patches).permute(0, 2, 3, 1)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if float(tensor.max().detach().cpu().item()) <= 1.0:
            tensor = tensor * 255.0

        return tensor

    def _to_hwc_batch(self, patches: np.ndarray) -> np.ndarray:
        if patches.ndim == 3:
            patches = patches[np.newaxis, ...]
        if patches.ndim != 4:
            raise PipelineInputShapeError(
                f"patch batch는 4차원 shape이어야 합니다. 입력 차원: {patches.ndim}D"
            )
        if patches.shape[1] != 3:
            raise PipelineInputShapeError(
                f"CHW patch batch는 3채널이어야 합니다. 입력 shape: {patches.shape}"
            )

        return patches.transpose(0, 2, 3, 1)

    def _to_chw_batch(self, patches: np.ndarray) -> np.ndarray:
        if patches.ndim == 3:
            patches = patches[np.newaxis, ...]
        if patches.ndim != 4:
            raise PipelineInputShapeError(
                f"patch batch는 4차원 shape이어야 합니다. 입력 차원: {patches.ndim}D"
            )
        if patches.shape[-1] != 3:
            raise PipelineInputShapeError(
                f"HWC patch batch는 3채널이어야 합니다. 입력 shape: {patches.shape}"
            )

        return patches.transpose(0, 3, 1, 2)
