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

from ai.metrics.base import Metric
from ai.metrics.ssim import SSIM
from ai.models.staingan.staingan_model import ResnetGenerator
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch, open_wsi_handle
from ai.wsi.writer import MultiZarrWSIWriter

"""
What it returns through writer.py:
1) result_img_path = <..._staingan.zarr> path to resulting image
2) metrics = Metrics(ssim=0.95, psnr=32.4, fid=60)
"""


@dataclass(slots=True)
class StainGANInferenceConfig:
    checkpoint_path: Path | None = None
    checkpoint_dir: Path = Path(__file__).resolve().parents[1] / "checkpoints" / "staingan"
    output_dir: Path = Path("result/staingan")

    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    generator_blocks: int = 9
    generator_direction: str = "a2b"

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


class StainGANPipeline(ModelPipeline):
    def __init__(
        self,
        logger: logging.Logger | None,
        config: StainGANInferenceConfig | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.config = config or StainGANInferenceConfig()
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
        target_img_path: Path | None = None,
        metrics: dict[str, Metric] | None = None,
    ) -> PipelineResult:
        del target_img_path

        src_wsi_handle = open_wsi_handle(src_img_path) #open source img using wsi handle
        level_count = len(src_wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise ValueError(
                f"read_level {self.config.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        level_downsample = float(src_wsi_handle.level_downsamples[self.config.read_level])
        refs = self.grid_sampler.sample(src_wsi_handle) #patch(?) reference
        output_path = self._build_output_path(src_img_path)
        total_refs = len(refs)
        total_batches = (total_refs + self.config.batch_size - 1) // self.config.batch_size

        checkpoint_path = self._resolve_checkpoint_path()
        self._log_run_summary(
            src_img_path=src_img_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            wsi_handle=src_wsi_handle,
            total_refs=total_refs,
            total_batches=total_batches,
        )

        writer = MultiZarrWSIWriter(
            output_path=output_path,
            width=read_w,
            height=read_h,
            level_downsample=level_downsample,
            channels=3,
            tile_size=self.config.tile_size,
            overwrite=True,
            pyramid_levels=self.config.pyramid_levels,
        )

        #Metric Calculation
        run_start = time.time()
        scores = dict.fromkeys(metrics.keys() if metrics is not None else [], 0.0)
        if self.config.compute_ssim and "ssim" not in scores:
            scores["ssim"] = 0.0
        metric_objects = dict(metrics or {})
        if self.config.compute_ssim and "ssim" not in metric_objects:
            metric_objects["ssim"] = SSIM()

        for start in range(0, len(refs), self.config.batch_size):
            batch_refs = refs[start:start + self.config.batch_size]
            batch_patches = [load_patch(ref).img for ref in batch_refs]
            normalized_batch = self._normalize_batch(batch_patches)
            batch_input = np.stack(batch_patches, axis=0)
            batch_output = np.stack(normalized_batch, axis=0)

            for key, metric in metric_objects.items():
                scores[key] += metric.evaluate(batch_input, batch_output)

            for ref, normalized_patch in zip(batch_refs, normalized_batch):
                writer.write_patch(ref, normalized_patch)

            batch_index = (start // self.config.batch_size) + 1
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % max(self.config.log_every_batches, 1) == 0
            ):
                elapsed = time.time() - run_start
                processed = min(start + len(batch_refs), total_refs)
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = total_refs - processed
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "unknown"
                self._log(
                    f"Processed batch {batch_index}/{total_batches} "
                    f"({processed}/{total_refs} patches, {rate:.2f} patches/s, eta {eta_text})"
                )

        self._log("Finalizing MultiZarr writer and writing thumbnail...")
        final_output_path = writer.finalize()
        total_elapsed = time.time() - run_start
        normalized_scores = {
            key: value / max(total_batches, 1) for key, value in scores.items()
        }

        self._log(
            f"Finished inference in {total_elapsed:.1f}s. "
            f"Output written to {final_output_path}"
        )
        self._log(f"Thumbnail written to {writer.thumbnail_path}")
        for key, value in normalized_scores.items():
            self._log(f"{key.upper()}: {value:.6f}")
        return PipelineResult(
            output_path=str(final_output_path),
            scores=normalized_scores,
            thumbnail_path=str(writer.thumbnail_path),
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
        if self.config.generator_direction not in {"a2b", "b2a"}:
            raise ValueError(
                f"generator_direction must be 'a2b' or 'b2a', got {self.config.generator_direction!r}"
            )
        
    # Build the generator and load trained weights into it
    def _load_model(self) -> ResnetGenerator:
        model = ResnetGenerator( #Create the generator architecture
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            ngf=self.config.ngf,
            n_blocks=self.config.generator_blocks,
        )

        checkpoint_path = self._resolve_checkpoint_path() #select which checkpoint path to load
        checkpoint = self._load_checkpoint(checkpoint_path) #load the checkpoint from path
        state_dict = self._extract_state_dict(checkpoint) #extract generator weights from ckpt
        state_dict = self._strip_module_prefix(state_dict) #strip unnecessary prefixes
        model.load_state_dict(state_dict) #load weights onto model
        return model

    # Actually opens and loads that checkpoint file with PyTorch
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
                    "Failed to load the StainGAN checkpoint in weights-only mode. "
                    "If this checkpoint was produced outside this project, inspect "
                    "its contents before relaxing the loader further."
                ) from exc

    # Figures out which checkpoint file to use
    # returns a Path to the checkpoint file
    def _resolve_checkpoint_path(self) -> Path:
        if self.config.checkpoint_path is not None: #if checkpoint path exists in config, use that file exactlys
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"StainGAN checkpoint not found: {checkpoint_path}")
            return checkpoint_path

        checkpoint_dir = Path(self.config.checkpoint_dir) # otherwise look inside checkpoint_dir
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"StainGAN checkpoint directory not found: {checkpoint_dir}"
            )

        latest_candidates = sorted( #look for latest checkpoints first (ones that end with .pth or .pt)
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
                "Multiple StainGAN latest checkpoint files found. "
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
                f"No StainGAN checkpoint found in: {checkpoint_dir}"
            )
        if len(candidates) > 1:
            names = ", ".join(str(path) for path in candidates)
            raise ValueError(
                "Multiple checkpoint files found. Pass checkpoint_path explicitly: "
                f"{names}"
            )
        return candidates[0]
    
    #From the loaded checkpoint, pull out the actual model weights dictionary, from whatever format was used
    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict): #if the checkpoint is a dict, defined preferred keys
            preferred_keys = ( #the list of key names it will search for
                f"g_{self.config.generator_direction}_state_dict", ##directional generator key (a to b or b to a)
                "g_a2b_state_dict", 
                "g_b2a_state_dict",
                "state_dict",
                "model_state_dict",
                "net",
                "model",
            )
            for key in preferred_keys: #for each key, get checkpoint[key] from dict
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint

        raise ValueError("Checkpoint does not contain a valid generator state_dict.")

    #Checking if parameter names in checkpoint have extra prefixes/modules that needs to be removed
    #E.g. "module.model.0.weight" instead of "module.0.weight"
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

    # takes a list of image patches, groups them into a batch, 
    # normalizes them into the numeric range the model expects, runs them through the StainGAN generator, 
    # and converts the outputs back into regular image arrays (uint8)
    def _normalize_batch(self, patches_chw: list[np.ndarray]) -> list[np.ndarray]: #input is a list of NumPy arrays, each shaped like (C, H, W)
        batch = np.stack(patches_chw, axis=0).astype(np.float32) / 255.0 # stack patches into one batch, scale to [0,1]
        tensor = torch.from_numpy(batch).to(self.device) #convert numpy -> tensor
        tensor = (tensor - 0.5) * 2.0

        with torch.inference_mode(): #inference mode only - no tracking gradients
            output = self.model(tensor) #applies trained staingan generator to the batch

        output = output * 0.5 + 0.5 #Convert output from [-1,1] back to [0,1]
        output = torch.clamp(output, 0.0, 1.0) #clamp to valid image range - second check
        output_np = output.detach().cpu().numpy()
        output_np = np.rint(output_np * 255.0).astype(np.uint8) #Convert from [0,1] floats to [0,255] uint8 image values
        return [output_np[i] for i in range(output_np.shape[0])] #split back into list of patches

    def _build_output_path(self, src_img_path: Path) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_img_path).stem
        return self.config.output_dir / f"{stem}_staingan.zarr"

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
        self._log(f"  generator_direction={self.config.generator_direction}")
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
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[StainGANPipeline] {message}", flush=True)
        if self.logger is not None:
            self.logger.info(message)


StainGAN = StainGANPipeline
StainGANConfig = StainGANInferenceConfig
