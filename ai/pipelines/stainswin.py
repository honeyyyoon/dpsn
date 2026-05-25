from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from pathlib import PosixPath, WindowsPath
import time
from typing import Any

import numpy as np
import torch

from ai.metrics.metric import Metric
from ai.models.stainswin.stainswin_model import StainSWIN as StainSWINModel
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter


@dataclass(slots=True)
class StainSWINInferenceConfig:
    checkpoint_path: Path | None = None
    checkpoint_dir: Path = Path(__file__).resolve().parents[1] / "checkpoints" / "stainswin"

    input_nc: int = 3
    output_nc: int = 3
    embed_dim: int = 30
    num_heads: int = 6
    num_res_blocks: int = 4
    stbs_per_block: int = 6
    window_size: int = 8
    mlp_ratio: float = 4.0
    conv_kernel_size: int = 3
    reconstruction_channels: int | None = None
    use_image_residual: bool = True

    patch_size: int = 512
    stride: int = 512
    read_level: int = 0
    batch_size: int = 4
    fallback_batch_sizes: tuple[int, ...] = (2, 1)
    tile_size: int = 512
    device: str = "auto"
    keep_store: bool = False
    verbose: bool = True
    log_every_batches: int = 5
    compute_ssim: bool = True


class StainSWINPipeline(ModelPipeline):
    """
    WSI inference pipeline for a trained StainSWIN model.

    This follows the current StainNet pipeline structure closely and swaps in
    the transformer-based StainSWIN model.
    """

    def __init__(
        self,
        logger: logging.Logger | None,
        config: StainSWINInferenceConfig | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.config = config or StainSWINInferenceConfig()
        self._validate_config()

        self.device = self._select_device(self.config.device)
        self.grid_sampler = GridSampler(
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            read_level=self.config.read_level,
        )
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def run(
        self,
        src_img_path: Path,
        result_path: Path,
        target_img_path: Path | None = None,
        metrics: list[str] = [],
        emit_event = None
    ) -> PipelineResult:
        
        tgt_images = None
        if "fid" in metrics:
            if target_img_path is None:
                raise ValueError("FID needs target image")
            tgt_wsi_handle = open_wsi_handle(target_img_path)
            tgt_refs = self.grid_sampler.sample(tgt_wsi_handle)
            tgt_images = np.stack([load_patch(ref).img for ref in tgt_refs], axis=0)
        del target_img_path

        src_wsi_handle = open_wsi_handle(src_img_path)
        level_count = len(src_wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise ValueError(
                f"read_level {self.config.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        level_downsample = float(
            src_wsi_handle.level_downsamples[self.config.read_level]
        )
        refs = self.grid_sampler.sample(src_wsi_handle)
        total_refs = len(refs)
        total_batches = (total_refs + self.config.batch_size - 1) // self.config.batch_size

        checkpoint_path = self._resolve_checkpoint_path()
        self._log_run_summary(
            src_img_path=src_img_path,
            checkpoint_path=checkpoint_path,
            output_path=Path("result"),
            wsi_handle=src_wsi_handle,
            total_refs=total_refs,
            total_batches=total_batches,
        )

        self._log(
            f"Loaded WSI metadata: read_level={self.config.read_level}, "
            f"level_shape=({read_w}, {read_h}), total_patches={total_refs}, "
            f"batch_size={self.config.batch_size}, total_batches={total_batches}"
        )

        writer = MultiZarrWSIWriter(
            output_path=result_path,
            width=read_w,
            height=read_h,
            level_downsample=level_downsample,
            channels=3,
            tile_size=self.config.tile_size,
            overwrite=True,
        )

        run_start = time.time()
        metric = Metric(
            use_ssim = "ssim" in metrics,
            use_psnr = "psnr" in metrics,
            use_fid = "fid" in metrics,
            target_patch = tgt_images
        )

        start = 0
        processed_batches = 0
        current_batch_size = self.config.batch_size
        while start < len(refs):
            batch_refs = refs[start:start + current_batch_size]
            batch_patches = [load_patch(ref).img for ref in batch_refs]
            normalized_batch, current_batch_size = self._normalize_batch_with_fallback(
                batch_patches,
                current_batch_size,
            )
            batch_input = np.stack(batch_patches, axis=0)
            batch_output = np.stack(normalized_batch, axis=0)

            metric.evaluate(batch_input, batch_output)

            for ref, normalized_patch in zip(batch_refs, normalized_batch):
                writer.write_patch(ref, normalized_patch)

            processed_batches += 1
            start += len(batch_refs)
            processed = min(start, total_refs)
            if emit_event:
                emit_event(
                    status="running",
                    progress=int(processed / total_refs * 100),
                    message=(
                        f"Processing {start - len(batch_refs)} ~ "
                        f"{processed} / {total_refs}"
                    ),
                )

            if (
                processed_batches == 1
                or start >= total_refs
                or processed_batches % max(self.config.log_every_batches, 1) == 0
            ):
                elapsed = time.time() - run_start
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = total_refs - processed
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_text = (
                    f"{eta_seconds:.1f}s"
                    if np.isfinite(eta_seconds)
                    else "unknown"
                )
                self._log(
                    f"Processed batch {processed_batches} "
                    f"({processed}/{total_refs} patches, "
                    f"batch_size={current_batch_size}, "
                    f"{rate:.2f} patches/s, eta {eta_text})"
                )

        self._log("Finalizing MultiZarr writer and writing WSI TIFF...")
        final_output_path = writer.finalize()
        writer.close()
        total_elapsed = time.time() - run_start
        normalized_scores = metric.finalize()

        self._log(
            f"Finished inference in {total_elapsed:.1f}s. "
            f"Output written to {final_output_path}"
        )
        self._log(f"WSI TIFF written to {writer.wsi_path}")
        for key, value in normalized_scores.items():
            self._log(f"{key.upper()}: {value:.6f}")
        return PipelineResult(
            output_path=final_output_path,
            scores=normalized_scores,
            thumbnail_path=None,
        )

    def _validate_config(self) -> None:
        if self.config.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.config.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.config.read_level < 0:
            raise ValueError("read_level must be >= 0")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if any(batch_size <= 0 for batch_size in self.config.fallback_batch_sizes):
            raise ValueError("fallback_batch_sizes must all be > 0")
        if any(batch_size >= self.config.batch_size for batch_size in self.config.fallback_batch_sizes):
            raise ValueError("fallback_batch_sizes must be smaller than batch_size")
        if self.config.tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if self.config.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.config.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.config.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be > 0")
        if self.config.stbs_per_block <= 0:
            raise ValueError("stbs_per_block must be > 0")

    def _load_model(self) -> StainSWINModel:
        model = StainSWINModel(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_res_blocks=self.config.num_res_blocks,
            stbs_per_block=self.config.stbs_per_block,
            window_size=self.config.window_size,
            mlp_ratio=self.config.mlp_ratio,
            conv_kernel_size=self.config.conv_kernel_size,
            reconstruction_channels=self.config.reconstruction_channels,
            use_image_residual=self.config.use_image_residual,
        )

        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        return model

    def _load_checkpoint(self, checkpoint_path: Path) -> Any:
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except pickle.UnpicklingError as exc:
            torch.serialization.add_safe_globals([Path, PosixPath, WindowsPath])
            try:
                return torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except pickle.UnpicklingError:
                raise ValueError(
                    "Failed to load the StainSWIN checkpoint in weights-only mode. "
                    "If this checkpoint was produced outside this project, inspect "
                    "its contents before relaxing the loader further."
                ) from exc

    def _resolve_checkpoint_path(self) -> Path:
        if self.config.checkpoint_path is not None:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"StainSWIN checkpoint not found: {checkpoint_path}")
            return checkpoint_path

        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"StainSWIN checkpoint directory not found: {checkpoint_dir}"
            )

        latest_candidates = sorted(
            [
                *checkpoint_dir.glob("*latest*.pth"),
                *checkpoint_dir.glob("*latest*.pt"),
            ]
        )
        if len(latest_candidates) == 1:
            return latest_candidates[0]
        if len(latest_candidates) > 1:
            names = ", ".join(str(path) for path in latest_candidates)
            raise ValueError(
                "Multiple StainSWIN latest checkpoint files found. "
                f"Pass checkpoint_path explicitly: {names}"
            )

        candidates = sorted(
            [
                *checkpoint_dir.glob("*.pth"),
                *checkpoint_dir.glob("*.pt"),
            ]
        )
        if not candidates:
            raise FileNotFoundError(
                f"No StainSWIN checkpoint found in: {checkpoint_dir}"
            )
        if len(candidates) > 1:
            names = ", ".join(str(path) for path in candidates)
            raise ValueError(
                "Multiple checkpoint files found. Pass checkpoint_path explicitly: "
                f"{names}"
            )
        return candidates[0]

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "net", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint

        raise ValueError("Checkpoint does not contain a valid state_dict.")

    def _strip_module_prefix(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prefix = "module."
        if not any(key.startswith(prefix) for key in state_dict):
            return state_dict

        return {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }

    def _normalize_batch(self, patches_chw: list[np.ndarray]) -> list[np.ndarray]:
        batch = np.stack(patches_chw, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).to(
            device=self.device,
            dtype=torch.float32,
        )
        tensor = (tensor - 0.5) * 2.0

        with torch.inference_mode():
            output = self.model(tensor)

        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0.0, 1.0)

        output_np = output.detach().cpu().numpy()
        output_np = np.rint(output_np * 255.0).astype(np.uint8)
        return [output_np[i] for i in range(output_np.shape[0])]
    
    def _normalize_batch_with_fallback(
        self,
        patches_chw: list[np.ndarray],
        batch_size: int,
    ) -> tuple[list[np.ndarray], int]:
        candidate_sizes = self._candidate_batch_sizes(batch_size)
        last_error: RuntimeError | None = None

        for candidate_size in candidate_sizes:
            try:
                if candidate_size != batch_size:
                    self._log(
                        "Retrying StainSWIN inference with reduced "
                        f"batch_size={candidate_size}."
                    )
                normalized: list[np.ndarray] = []
                for start in range(0, len(patches_chw), candidate_size):
                    normalized.extend(
                        self._normalize_batch(patches_chw[start:start + candidate_size])
                    )
                return normalized, candidate_size
            except RuntimeError as error:
                if not self._is_cuda_oom(error):
                    raise
                last_error = error
                self._log(
                    f"CUDA OOM encountered with batch_size={candidate_size}. "
                    "Trying a smaller batch size..."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        assert last_error is not None
        raise last_error
    
    def _candidate_batch_sizes(self, current_batch_size: int) -> list[int]:
        candidates = [self.config.batch_size, *self.config.fallback_batch_sizes]
        candidates = sorted(set(candidates), reverse=True)
        return [size for size in candidates if size <= current_batch_size]
    
    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _select_device(self, device: str) -> torch.device:
        if device != "auto":
            resolved = torch.device(device)
            if resolved.type == "cuda" and (resolved.index is None or resolved.index == 0):
                raise ValueError(
                    "GPU 0 is disabled for this project. Please use cuda:1, cuda:2, or cuda:3."
                )
            return resolved

        if torch.cuda.is_available():
            for gpu_index in (1, 2, 3):
                if torch.cuda.device_count() > gpu_index:
                    return torch.device(f"cuda:{gpu_index}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _log_run_summary(
        self,
        src_img_path: Path,
        checkpoint_path: Path,
        output_path: Path,
        wsi_handle: WSIHandle,
        total_refs: int,
        total_batches: int,
    ) -> None:
        self._log("Run configuration:")
        self._log(f"  input_path={src_img_path}")
        self._log(f"  checkpoint_path={checkpoint_path}")
        self._log(f"  output_path={output_path}")
        self._log(f"  read_level={self.config.read_level}")
        self._log(f"  patch_size={self.config.patch_size}")
        self._log(f"  stride={self.config.stride}")
        self._log(f"  batch_size={self.config.batch_size}")
        self._log(f"  tile_size={self.config.tile_size}")
        self._log(f"  device={self.device}")
        self._log(f"  compute_ssim={self.config.compute_ssim}")
        self._log(f"  level_dimensions={wsi_handle.level_dimensions}")
        self._log(f"  total_patches={total_refs}")
        self._log(f"  total_batches={total_batches}")

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[StainSWINPipeline] {message}", flush=True)
        if self.logger is not None:
            self.logger.info(message)


StainSWIN = StainSWINPipeline
StainSWINConfig = StainSWINInferenceConfig
