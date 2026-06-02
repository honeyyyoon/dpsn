from __future__ import annotations

from dataclasses import dataclass
import pickle
import logging
from pathlib import Path
from pathlib import PosixPath, WindowsPath
import time
from typing import Any

import numpy as np
import torch

from ai.metrics.metric import Metric
from ai.models.stainnet.stainnet_model import StainNet as StainNetModel
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter


class StainNetError(RuntimeError):
    """Base class for StainNet pipeline errors."""


class MissingTargetImageError(StainNetError):
    """Raised when target-dependent metrics are requested without a target image."""


class StainNetReadLevelError(StainNetError):
    """Raised when the configured read level is not available in the WSI."""


class StainNetCheckpointError(StainNetError):
    """Base class for StainNet checkpoint errors."""


class StainNetCheckpointNotFoundError(StainNetCheckpointError):
    """Raised when a required StainNet checkpoint cannot be found."""


class StainNetCheckpointLoadError(StainNetCheckpointError):
    """Raised when a StainNet checkpoint cannot be loaded safely."""


class StainNetCheckpointSelectionError(StainNetCheckpointError):
    """Raised when checkpoint discovery finds ambiguous candidates."""


class StainNetInvalidCheckpointError(StainNetCheckpointError):
    """Raised when a StainNet checkpoint does not contain usable model weights."""


class StainNetDeviceError(StainNetError):
    """Raised when the requested device is not allowed for StainNet inference."""


# class that stores all settings needed for StainNet WSI inference
@dataclass(slots=True)
class StainNetInferenceConfig:
    """
    Configuration for patch-wise WSI inference with a trained StainNet model.
    """

    checkpoint_path: Path | None = None
    checkpoint_dir: Path = Path(__file__).resolve().parents[1] / "checkpoints" / "stainnet"
    # output_dir: Path = Path("/mnt/Disk1/dpsn_datasets/inf_result_stainnet")

    input_nc: int = 3
    output_nc: int = 3
    channels: int = 32
    n_layer: int = 3
    kernel_size: int = 1

    patch_size: int = 512
    stride: int = 512
    read_level: int = 0
    batch_size: int = 8
    tile_size: int = 512 # small rectangular chunks called that a WSI is divided and saved in
    device: str = "auto"
    keep_store: bool = False # whether to keep the writer’s intermediate storage after output is finalized
    verbose: bool = True
    log_every_batches: int = 5
    compute_ssim: bool = True

# Class that performs the whole inference procedure on a WSI
class StainNetPipeline(ModelPipeline):
    """
    WSI inference pipeline for a trained StainNet model.

    This path is for inference only. Training uses paired aligned image/patch
    folders and lives in separate dataset/training modules.
    """

    def __init__(self, logger: logging.Logger | None, config: StainNetInferenceConfig | None = None) -> None:
        super().__init__(logger=logger)
        self.config = config or StainNetInferenceConfig()
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
        emit_event=None
    ) -> PipelineResult:
        self._emit_progress(emit_event, 1, "Preparing StainNet inference.")

        tgt_imgs = None
        if "fid" in metrics or "custom" in metrics:
            if target_img_path is None:
                raise MissingTargetImageError("타겟 의존 메트릭을 계산하려면 타겟 이미지가 필요합니다.")
            self._emit_progress(emit_event, 3, "Loading target patches for metrics.")
            tgt_wsi_handle = open_wsi_handle(target_img_path)
            tgt_refs = self.grid_sampler.sample(tgt_wsi_handle)
            tgt_imgs = np.stack([load_patch(ref).img for ref in tgt_refs], axis=0)
            self._emit_progress(emit_event, 6, "Loaded target patches for metrics.")

        del target_img_path

        self._emit_progress(emit_event, 8, "Sampling source WSI patches.")
        src_wsi_handle = open_wsi_handle(src_img_path)
        level_count = len(src_wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise StainNetReadLevelError(
                f"read_level {self.config.read_level}은 0 이상 {level_count - 1} 이하이어야 합니다."
            )

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        level_downsample = float(
            src_wsi_handle.level_downsamples[self.config.read_level]
        )
        refs = self.grid_sampler.sample(src_wsi_handle)
        # output_path = self._build_output_path(src_img_path)
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
            use_custom = "custom" in metrics,
            target_patch = tgt_imgs
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
                eta_text = (
                    f"{eta_seconds:.1f}s"
                    if np.isfinite(eta_seconds)
                    else "unknown"
                )
                self._log(
                    f"Processed batch {batch_index}/{total_batches} "
                    f"({processed}/{total_refs} patches, "
                    f"{rate:.2f} patches/s, eta {eta_text})"
                )

        self._log("Finalizing MultiZarr writer and writing WSI TIFF...")
        self._emit_progress(emit_event, 88, "Writing output WSI TIFF.")
        final_output_path = writer.finalize()
        writer.close()
        total_elapsed = time.time() - run_start
        self._emit_progress(emit_event, 95, "Computing final metrics.")
        normalized_scores = metric.finalize()
        self._emit_progress(emit_event, 98, "Finalizing StainNet result.")

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
            raise ValueError("patch_size는 0보다 커야 합니다.")
        if self.config.stride <= 0:
            raise ValueError("stride는 0보다 커야 합니다.")
        if self.config.read_level < 0:
            raise ValueError("read_level은 0 이상이어야 합니다.")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size는 0보다 커야 합니다.")
        if self.config.tile_size <= 0:
            raise ValueError("tile_size는 0보다 커야 합니다.")

    def _load_model(self) -> StainNetModel:
        model = StainNetModel(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            n_layer=self.config.n_layer,
            n_channel=self.config.channels,
            kernel_size=self.config.kernel_size,
        )

        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            raise StainNetInvalidCheckpointError(
                f"StainNet state_dict를 불러오지 못했습니다: {error}"
            ) from error
        return model

    def _load_checkpoint(self, checkpoint_path: Path) -> Any:
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except pickle.UnpicklingError as exc:
            # Our own training checkpoints store config metadata containing
            # pathlib paths, which strict weights-only loading blocks unless
            # those classes are explicitly allowlisted.
            torch.serialization.add_safe_globals([Path, PosixPath, WindowsPath])
            try:
                return torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except Exception as error:
                raise StainNetCheckpointLoadError(
                    "StainNet checkpoint를 weights-only 모드로 불러오지 못했습니다. "
                    "이 프로젝트 외부에서 생성된 checkpoint라면 loader 제한을 완화하기 전에 "
                    "파일 내용을 먼저 확인하세요."
                ) from error
        except Exception as error:
            raise StainNetCheckpointLoadError(
                f"StainNet checkpoint를 불러오지 못했습니다: {checkpoint_path}. {error}"
            ) from error

    def _resolve_checkpoint_path(self) -> Path:
        if self.config.checkpoint_path is not None:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.is_file():
                raise StainNetCheckpointNotFoundError(
                    f"StainNet checkpoint를 찾을 수 없습니다: {checkpoint_path}"
                )
            return checkpoint_path

        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise StainNetCheckpointNotFoundError(
                f"StainNet checkpoint 디렉터리를 찾을 수 없습니다: {checkpoint_dir}"
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
            raise StainNetCheckpointSelectionError(
                "StainNet latest checkpoint 파일이 여러 개 발견되었습니다. "
                f"checkpoint_path를 명시적으로 지정하세요: {names}"
            )

        candidates = sorted(
            [
                *checkpoint_dir.glob("*.pth"),
                *checkpoint_dir.glob("*.pt"),
            ]
        )

        if not candidates:
            raise StainNetCheckpointNotFoundError(
                f"StainNet checkpoint를 찾을 수 없습니다: {checkpoint_dir}"
            )
        if len(candidates) > 1:
            names = ", ".join(str(path) for path in candidates)
            raise StainNetCheckpointSelectionError(
                "checkpoint 파일이 여러 개 발견되었습니다. checkpoint_path를 명시적으로 지정하세요: "
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

        raise StainNetInvalidCheckpointError("checkpoint에 유효한 state_dict가 없습니다.")

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

        # Original StainNet test code maps [0, 1] -> [-1, 1] before inference.
        tensor = (tensor - 0.5) * 2.0

        with torch.inference_mode():
            output = self.model(tensor)

        # Original StainNet test code maps model output back to [0, 1].
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0.0, 1.0)

        output_np = output.detach().cpu().numpy()
        output_np = np.rint(output_np * 255.0).astype(np.uint8)
        return [output_np[i] for i in range(output_np.shape[0])]

    def _select_device(self, device: str) -> torch.device:
        if device != "auto":
            resolved = torch.device(device)
            if resolved.type == "cuda" and (resolved.index is None or resolved.index == 0):
                raise StainNetDeviceError(
                    "이 프로젝트에서는 GPU 0을 사용할 수 없습니다. cuda:1, cuda:2 또는 cuda:3을 사용하세요."
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

    # def _build_output_path(self, src_img_path: Path) -> Path:
    #     self.config.output_dir.mkdir(parents=True, exist_ok=True)
    #     stem = Path(src_img_path).stem
    #     return self.config.output_dir / f"{stem}_stainnet.zarr"

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
            print(f"[StainNetPipeline] {message}", flush=True)
        if self.logger is not None:
            self.logger.info(message)


# Backward-compatible alias while the rest of the project catches up.
StainNet = StainNetPipeline
StainNetConfig = StainNetInferenceConfig
