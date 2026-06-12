from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from pathlib import PosixPath, WindowsPath
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

from ai.metrics.metric import Metric
from ai.models.multistain.networks import ResnetGenerator
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.pipelines.target_utils import load_grid_target_patches
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter


@dataclass(slots=True)
class MultiStainCycleGANInferenceConfig:
    checkpoint_path: Path | None = None
    checkpoint_dir: Path = Path("/home/snu_2026/dpsn/ai/checkpoints/multistain")

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    generator_blocks: int = 9
    generator_key: str = "net_g_source_to_target"

    patch_size: int = 512
    stride: int = 512
    read_level: int = 0
    batch_size: int = 8
    tile_size: int = 512
    pyramid_levels: int = 3
    device: str = "auto"
    verbose: bool = True
    log_every_batches: int = 5
    compute_ssim: bool = True
    preserve_background: bool = True


class MultiStainCycleGANPipeline(ModelPipeline):
    def __init__(
        self,
        logger: logging.Logger | None,
        config: MultiStainCycleGANInferenceConfig | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.config = config or MultiStainCycleGANInferenceConfig()
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
        target_img_path: Path | Sequence[Path] | None = None,
        metrics: list[str] = [],
        emit_event=None,
    ) -> PipelineResult:
        self._emit_progress(emit_event, 1, "Preparing MultiStain-CycleGAN inference.")

        tgt_images = None
        if "fid" in metrics or "gaussian_color_dist" in metrics:
            if target_img_path is None:
                raise ValueError("Target-dependent metrics need target image")
            self._emit_progress(emit_event, 3, "Loading target patches for FID.")
            tgt_images = load_grid_target_patches(target_img_path, self.grid_sampler)
            self._emit_progress(emit_event, 6, "Loaded target patches for FID.")

        del target_img_path

        self._emit_progress(emit_event, 8, "Sampling source WSI patches.")
        src_wsi_handle = open_wsi_handle(src_img_path)
        level_count = len(src_wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise ValueError(
                f"read_level {self.config.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        level_downsample = float(src_wsi_handle.level_downsamples[self.config.read_level])
        refs = self.grid_sampler.sample(src_wsi_handle)
        total_refs = len(refs)
        total_batches = (total_refs + self.config.batch_size - 1) // self.config.batch_size

        checkpoint_path = self._resolve_checkpoint_path()
        self._log_run_summary(
            src_img_path=src_img_path,
            checkpoint_path=checkpoint_path,
            output_path=result_path,
            wsi_handle=src_wsi_handle,
            total_refs=total_refs,
            total_batches=total_batches,
        )

        writer = MultiZarrWSIWriter(
            output_path=result_path,
            width=read_w,
            height=read_h,
            level_downsample=level_downsample,
            channels=3,
            tile_size=self.config.tile_size,
            overwrite=True,
            pyramid_levels=self.config.pyramid_levels,
        )

        run_start = time.time()
        metric = Metric(
            use_ssim="ssim" in metrics,
            use_psnr="psnr" in metrics,
            use_fid="fid" in metrics,
            use_gaussian_color_dist="gaussian_color_dist" in metrics,
            target_patch=tgt_images,
        )
        self._emit_progress(emit_event, 10, f"Starting inference on {total_refs} patches.")

        for start in range(0, len(refs), self.config.batch_size):
            batch_refs = refs[start:start + self.config.batch_size]
            batch_patches = [load_patch(ref).img for ref in batch_refs]
            normalized_batch = self._normalize_batch(batch_patches)
            batch_input = np.stack(batch_patches, axis=0)
            batch_output = np.stack(normalized_batch, axis=0)

            for ref, normalized_patch in zip(batch_refs, normalized_batch):
                writer.write_patch(ref, normalized_patch)

            batch_index = (start // self.config.batch_size) + 1
            processed = min(start + len(batch_refs), total_refs)
            self._emit_progress(
                emit_event,
                10 + int(processed / max(total_refs, 1) * 75),
                f"Processing {start} ~ {processed} / {total_refs}",
            )

            metric.evaluate(batch_input, batch_output)

            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % max(self.config.log_every_batches, 1) == 0
            ):
                elapsed = time.time() - run_start
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = total_refs - processed
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "unknown"
                self._log(
                    f"Processed batch {batch_index}/{total_batches} "
                    f"({processed}/{total_refs} patches, {rate:.2f} patches/s, eta {eta_text})"
                )

        self._log("Finalizing MultiZarr writer and writing WSI TIFF...")
        self._emit_progress(emit_event, 88, "Writing output WSI TIFF.")
        final_output_path = writer.finalize()
        writer.close()
        total_elapsed = time.time() - run_start
        self._emit_progress(emit_event, 95, "Computing final metrics.")
        normalized_scores = metric.finalize()
        self._emit_progress(emit_event, 98, "Finalizing MultiStain-CycleGAN result.")

        self._log(
            f"Finished inference in {total_elapsed:.1f}s. "
            f"Output written to {final_output_path}"
        )
        self._log(f"WSI TIFF written to {writer.wsi_path}")
        for key, value in normalized_scores.items():
            self._log(f"{key.upper()}: {value}")
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
        if self.config.tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if self.config.pyramid_levels < 0:
            raise ValueError("pyramid_levels must be >= 0")

    def _load_model(self) -> ResnetGenerator:
        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = self._load_checkpoint(checkpoint_path)
        self._apply_checkpoint_model_config(checkpoint)

        model = ResnetGenerator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            ngf=self.config.ngf,
            n_blocks=self.config.generator_blocks,
        )

        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        return model

    def _apply_checkpoint_model_config(self, checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict):
            return

        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            return

        for field_name in ("input_nc", "output_nc", "ngf", "generator_blocks"):
            value = checkpoint_config.get(field_name)
            if value is not None:
                setattr(self.config, field_name, value)

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
                    "Failed to load the MultiStain-CycleGAN checkpoint in weights-only mode. "
                    "If this checkpoint was produced outside this project, inspect "
                    "its contents before relaxing the loader further."
                ) from exc

    def _resolve_checkpoint_path(self) -> Path:
        if self.config.checkpoint_path is not None:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"MultiStain-CycleGAN checkpoint not found: {checkpoint_path}"
                )
            return checkpoint_path

        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"MultiStain-CycleGAN checkpoint directory not found: {checkpoint_dir}"
            )

        best_candidates = sorted(
            [
                *checkpoint_dir.glob("*best*.pth"),
                *checkpoint_dir.glob("*best*.pt"),
            ]
        )
        if len(best_candidates) == 1:
            return best_candidates[0]
        if len(best_candidates) > 1:
            names = ", ".join(str(path) for path in best_candidates)
            raise ValueError(
                "Multiple MultiStain-CycleGAN best checkpoint files found. "
                f"Pass checkpoint_path explicitly: {names}"
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
                "Multiple MultiStain-CycleGAN latest checkpoint files found. "
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
                f"No MultiStain-CycleGAN checkpoint found in: {checkpoint_dir}"
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
            preferred_keys = (
                self.config.generator_key,
                "net_g_source_to_target",
                "g_source_to_target_state_dict",
                "g_source_to_canonical_state_dict",
                "g_a2b_state_dict",
                "state_dict",
                "model_state_dict",
                "net",
                "model",
            )
            for key in preferred_keys:
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint

        raise ValueError(
            "Checkpoint does not contain a valid MultiStain-CycleGAN generator state_dict."
        )

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
        source_batch = np.stack(patches_chw, axis=0).astype(np.uint8)
        batch = source_batch.astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).to(
            device=self.device,
            dtype=torch.float32,
        )
        tensor = (tensor - 0.5) * 2.0
        input_h, input_w = tensor.shape[-2:]
        pad_h = (-input_h) % 4
        pad_w = (-input_w) % 4
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.inference_mode():
            output = self.model(tensor)

        output = output[..., :input_h, :input_w]
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0.0, 1.0)
        output_np = output.detach().cpu().numpy()
        output_np = np.rint(output_np * 255.0).astype(np.uint8)
        if self.config.preserve_background:
            output_np = self._preserve_background(source_batch, output_np)
        return [output_np[i] for i in range(output_np.shape[0])]

    def _preserve_background(
        self,
        source_batch: np.ndarray,
        output_batch: np.ndarray,
    ) -> np.ndarray:
        source_rgb = source_batch.astype(np.float32) / 255.0
        max_rgb = source_rgb.max(axis=1)
        min_rgb = source_rgb.min(axis=1)
        saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
        tissue_mask = (max_rgb > 0.08) & (max_rgb < 0.92) & (saturation > 0.04)

        if ndi is not None:
            soft_masks = []
            for mask in tissue_mask:
                expanded = ndi.binary_dilation(mask, iterations=8)
                soft = ndi.gaussian_filter(expanded.astype(np.float32), sigma=3.0)
                soft_masks.append(np.clip(soft, 0.0, 1.0))
            alpha = np.stack(soft_masks, axis=0)[:, None, :, :]
        else:
            alpha = tissue_mask.astype(np.float32)[:, None, :, :]

        blended = (
            alpha * output_batch.astype(np.float32)
            + (1.0 - alpha) * source_batch.astype(np.float32)
        )
        return np.rint(blended).clip(0, 255).astype(np.uint8)

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
        self._log(f"  generator_key={self.config.generator_key}")
        self._log(f"  read_level={self.config.read_level}")
        self._log(f"  patch_size={self.config.patch_size}")
        self._log(f"  stride={self.config.stride}")
        self._log(f"  batch_size={self.config.batch_size}")
        self._log(f"  tile_size={self.config.tile_size}")
        self._log(f"  pyramid_levels={self.config.pyramid_levels}")
        self._log(f"  device={self.device}")
        self._log(f"  compute_ssim={self.config.compute_ssim}")
        self._log(f"  level_dimensions={wsi_handle.level_dimensions}")
        self._log(f"  total_patches={total_refs}")
        self._log(f"  total_batches={total_batches}")

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

    def _emit_progress(self, emit_event, progress: int, message: str) -> None:
        if emit_event:
            emit_event(
                status="running",
                progress=max(0, min(99, int(progress))),
                message=message,
            )

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[MultiStainCycleGANPipeline] {message}", flush=True)
        if self.logger is not None:
            self.logger.info(message)


MultiStainCycleGAN = MultiStainCycleGANPipeline
MultiStainCycleGANConfig = MultiStainCycleGANInferenceConfig
