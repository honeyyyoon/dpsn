
"""
Inference Run Example:
    python -m ai.samplers.patch_sampler \
    --image /path/to/your_slide.tiff \
    --mode inference \
    --patch-size 256 \
    --read-level 0 \
    --save-debug

Training Run Example:
    python -m ai.samplers.patch_sampler \
    --image /path/to/your_slide.tiff \
    --mode training \
    --patch-size 256 \
    --read-level 0 \
    --max-patches 5000 \
    --seed 42 \
    --save-debug
    
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import openslide
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError as e:
    raise ImportError(
        "PatchSampler requires scipy. Please install it with: pip install scipy"
    ) from e

from ai.wsi.patch_ref import PatchRef
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import open_wsi_handle, load_patch


class PatchSamplerError(RuntimeError):
    """Base class for patch sampler errors."""


class MissingMPPError(PatchSamplerError):
    """Raised when a slide is missing MPP metadata and strict checking is enabled."""


class NoTissueFoundError(PatchSamplerError):
    """Raised when no tissue is found in the WSI thumbnail/mask."""


class NoValidPatchError(PatchSamplerError):
    """Raised when no valid patches satisfy the mask/filter conditions."""


class SlideOpenError(PatchSamplerError):
    """Raised when the WSI cannot be opened properly."""


class PatchSampler:
    """
    Patch sampler for WSI tissue masking + patch coordinate generation.

    Supports:
    - tissue masking on a low-resolution thumbnail
    - deterministic grid generation
    - training/inference mode thresholds
    - random subsampling for training when max_patches is set
    - debug image saving + text logging

    Notes
    - Patch coordinates are generated at `read_level`.
    - Returned PatchRefs store level-0 x/y because OpenSlide.read_region() expects that.
    - Patch size and stride are defined in read-level pixels.
    """

    def __init__(
        self,
        patch_size: int = 256,      # width, height of each patch (default: 256x256)
        stride: int | None = None,  # for sampling window. To be set when called
        read_level: int = 0,        # level to use  when defining patch coordinates
        training_tissue_threshold: float = 0.7,   # patch is accepted only if >= 70% of its area is tissue
        inference_tissue_threshold: float = 0.5,  # accepted if >= 50% of its area is tissue
        mask_longest_side: int = 2048,            # controls how large the mask image should be
        mask_saturation_threshold: float = 0.05,  # cutoff when determining if a pixel looks like tissue
        mask_white_threshold: float = 0.92,       
        morphology_kernel_size: int = 5,          # stronger smoothing/cleanup after thresholding
        min_mask_region_area: int = 256,          # eliminating connected tissue region after tissue masking
        strict_mpp_check: bool = True,            # whether sampler should fail if the WSI is missing MPP metadata
        result_dir: str | Path = "result",        # folder where outputs get saved:
        verbose: bool = False,
        log_to_file: bool = True,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if stride is None:
            stride = max(1, patch_size // 4)  # 25% of patch size
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if read_level < 0:
            raise ValueError(f"read_level must be >= 0, got {read_level}")
        if not (0.0 <= training_tissue_threshold <= 1.0):
            raise ValueError("training_tissue_threshold must be in [0, 1]")
        if not (0.0 <= inference_tissue_threshold <= 1.0):
            raise ValueError("inference_tissue_threshold must be in [0, 1]")
        if mask_longest_side <= 0:
            raise ValueError("mask_longest_side must be > 0")
        if not (0.0 <= mask_saturation_threshold <= 1.0):
            raise ValueError("mask_saturation_threshold must be in [0, 1]")
        if not (0.0 <= mask_white_threshold <= 1.0):
            raise ValueError("mask_white_threshold must be in [0, 1]")
        if morphology_kernel_size <= 0:
            raise ValueError("morphology_kernel_size must be > 0")
        if min_mask_region_area < 0:
            raise ValueError("min_mask_region_area must be >= 0")

        self.patch_size = patch_size
        self.stride = stride
        self.read_level = read_level
        self.training_tissue_threshold = training_tissue_threshold
        self.inference_tissue_threshold = inference_tissue_threshold
        self.mask_longest_side = mask_longest_side
        self.mask_saturation_threshold = mask_saturation_threshold
        self.mask_white_threshold = mask_white_threshold
        self.morphology_kernel_size = morphology_kernel_size
        self.min_mask_region_area = min_mask_region_area
        self.strict_mpp_check = strict_mpp_check
        self.verbose = bool(verbose)
        self.log_to_file = bool(log_to_file)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.result_dir / f"patch_sampler_{timestamp}.txt"
        self.logger = self._build_logger(self.log_path, log_to_file=self.log_to_file)

        self.logger.info("Initialized PatchSampler")
        self.logger.info("patch_size=%d", self.patch_size)
        self.logger.info("stride=%d", self.stride)
        self.logger.info("read_level=%d", self.read_level)
        self.logger.info("training_tissue_threshold=%.4f", self.training_tissue_threshold)
        self.logger.info("inference_tissue_threshold=%.4f", self.inference_tissue_threshold)
        self.logger.info("mask_longest_side=%d", self.mask_longest_side)

    def _build_logger(self, log_path: Path, log_to_file: bool = True) -> logging.Logger:
        logger_name = f"PatchSampler:{log_path.stem}:{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # avoid duplicated handlers if recreated
        if logger.handlers:
            logger.handlers.clear()

        if not log_to_file:
            logger.addHandler(logging.NullHandler())
            return logger

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def _console(self, message: str) -> None:
        if self.verbose:
            print(f"[PatchSampler] {message}", flush=True)
    
    # Main function
    def sample(
        self,
        wsi_handle: WSIHandle,
        mode: Literal["training", "inference"] = "inference",
        max_patches: int | None = None, #If slide produces too many valid patches
        seed: int | None = None,        #For random subsampling of patches
        save_debug: bool = True,        #If true: code logs thumbnail, tissue mask, tissue-mask overlay, and patch metadata json
    ) -> list[PatchRef]:
        """
        Generate PatchRefs from a single source WSI.

        Parameters
        ----------
        wsi_handle:
            Source WSIHandle only.
        mode:
            "training" or "inference"
        max_patches:
            - inference: keep all if None
            - training: if set and valid patches exceed it, random subsampling is applied
        seed:
            Random seed for reproducible training subsampling
        save_debug:
            Save thumbnail / tissue mask / overlay / sampled patch metadata
        """
        if mode not in {"training", "inference"}: #check that mode is valid
            raise ValueError(f"mode must be 'training' or 'inference', got {mode}")

        sample_start = time.perf_counter()
        self._console(
            f"sample start image={wsi_handle.image_path.name} mode={mode} "
            f"patch_size={self.patch_size} read_level={self.read_level} "
            f"dims={wsi_handle.level_dimensions} downsamples={wsi_handle.level_downsamples} "
            f"mpp={wsi_handle.mpp}"
        )

        if self.strict_mpp_check: #Check whether slide has valid mpp values
            self._validate_mpp(wsi_handle)

        self._validate_read_level(wsi_handle) #check whether slide has valid read level

        #Start logging
        self.logger.info("----- New sampling run -----")
        self.logger.info("image_path=%s", wsi_handle.image_path)
        self.logger.info("mode=%s", mode)
        self.logger.info("max_patches=%s", max_patches)
        self.logger.info("seed=%s", seed)

        #Building tisue mask
        self._console("building tissue mask...")
        step_start = time.perf_counter()
        mask_bundle = self._build_tissue_mask(wsi_handle) #returns mask, thumbnail, and mask level
        self._console(f"built tissue mask in {time.perf_counter() - step_start:.2f}s")
        tissue_mask = mask_bundle["mask"] #binary array of 1s (tissue) and 0s (background)
        thumbnail_rgb = mask_bundle["thumbnail_rgb"] #rgb image at low res
        mask_level = mask_bundle["mask_level"] #level used to make the mask

        tissue_threshold = ( #which threshold to use depending on mode
            self.training_tissue_threshold if mode == "training"
            else self.inference_tissue_threshold
        )

        self._console("generating patch refs...")
        step_start = time.perf_counter()
        patch_refs = self._generate_patch_refs( #iterate over slides and create PatchRef
            wsi_handle=wsi_handle,
            tissue_mask=tissue_mask,
            mask_level=mask_level,
            tissue_threshold=tissue_threshold,
        )
        self._console(
            f"generated {len(patch_refs)} valid patch refs in {time.perf_counter() - step_start:.2f}s"
        )

        if not patch_refs: #if no valid patches exist
            self.logger.error("No valid patches found after tissue filtering.")
            raise NoValidPatchError(
                f"No valid patches found for slide: {wsi_handle.image_path}"
            )
        
        #log number of valid patches
        self.logger.info("valid_patch_count_before_sampling=%d", len(patch_refs))

        #random subsampling for training mode
        if mode == "training" and max_patches is not None and len(patch_refs) > max_patches:
            rng = random.Random(seed)
            original_count = len(patch_refs)
            selected_indices = rng.sample(range(len(patch_refs)), k=max_patches)
            patch_refs = [patch_refs[i] for i in selected_indices]
            self.logger.info(
                "Applied training random subsampling: kept %d / %d patches",
                max_patches,
                original_count,
            )
            self._console(f"subsampled {max_patches} / {original_count} patch refs")
        elif mode == "inference":
            # keep deterministic top-to-bottom, left-to-right order
            pass

        if save_debug: #Save debug outputs if requested
            self._save_debug_outputs(
                wsi_handle=wsi_handle,
                thumbnail_rgb=thumbnail_rgb,
                tissue_mask=tissue_mask,
                mask_level=mask_level,
                patch_refs=patch_refs,
                mode=mode,
            )

        #log final patch count
        self.logger.info("final_patch_count=%d", len(patch_refs))
        self.logger.info("Sampling completed successfully.")
        self._console(
            f"sample finished final_patch_count={len(patch_refs)} "
            f"elapsed={time.perf_counter() - sample_start:.2f}s"
        )
        return patch_refs

    def _validate_mpp(self, wsi_handle: WSIHandle) -> None:
        mpp_x, mpp_y = wsi_handle.mpp
        if mpp_x <= 0 or mpp_y <= 0:
            self.logger.error("Missing or invalid MPP metadata: mpp=%s", wsi_handle.mpp)
            raise MissingMPPError(
                f"Slide is missing valid MPP metadata: {wsi_handle.image_path}, mpp={wsi_handle.mpp}"
            )

    def _validate_read_level(self, wsi_handle: WSIHandle) -> None:
        level_count = len(wsi_handle.level_dimensions)
        if not (0 <= self.read_level < level_count):
            self.logger.error(
                "Invalid read_level=%d for level_count=%d",
                self.read_level, level_count
            )
            raise ValueError(
                f"read_level {self.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = wsi_handle.level_dimensions[self.read_level] #width and height at specified level
        if self.patch_size > read_w or self.patch_size > read_h:
            self.logger.error(
                "patch_size=%d is larger than read-level dimensions=%s",
                self.patch_size, (read_w, read_h)
            )
            raise ValueError(
                f"patch_size={self.patch_size} is larger than read level dimensions {(read_w, read_h)}"
            )

    def _build_tissue_mask(self, wsi_handle: WSIHandle) -> dict:
        """
        Build tissue mask from a low-resolution thumbnail.

        Method:
        1) choose low-res level automatically based on mask_longest_side
        2) read entire level as RGB
        3) compute grayscale + saturation
        4) tissue candidate = not-too-white AND (dark-enough OR saturated-enough)
        5) morphology cleanup
        """
        mask_level = self._select_mask_level(wsi_handle)
        level_w, level_h = wsi_handle.level_dimensions[mask_level] #getting width and height at the chosen level

        self.logger.info("Selected mask_level=%d with dimensions=%s", mask_level, (level_w, level_h)) #log chosen level
        self._console(f"selected mask_level={mask_level} dims={(level_w, level_h)}")

        ref = wsi_handle.make_ref((0, 0), mask_level, (level_w, level_h))
        self._console("loading mask-level thumbnail patch...")
        load_start = time.perf_counter()
        rgb = load_patch(ref).img.transpose([1, 2, 0])
        self._console(
            f"loaded thumbnail patch shape={rgb.shape} in {time.perf_counter() - load_start:.2f}s"
        )
        thumbnail_rgb = rgb.copy()

        mask_start = time.perf_counter()
        gray = self._rgb_to_gray(rgb)               # [0, 1] Convert RBG image to grayscale
        sat = self._rgb_to_saturation(rgb)          # [0, 1] Computes saturation
        otsu_t = self._otsu_threshold(gray)         # Computes automatic threshold from grayscale image

        white_mask = gray < self.mask_white_threshold # filtering condition for being nonwhite enough - ndarray
        dark_or_saturated = (gray < otsu_t) | (sat > self.mask_saturation_threshold) # filtering condition for being dark and saturated enough
        tissue = white_mask & dark_or_saturated # combined condition for being a tissue; rough binary mask

        tissue = self._clean_binary_mask(tissue) #filling holes, removing tiny components, etc
        self._console(f"computed/cleaned tissue mask in {time.perf_counter() - mask_start:.2f}s")

        tissue_ratio = float(tissue.mean()) # computing how much of thumbnail is tissue
        self.logger.info("Otsu threshold (gray) = %.6f", otsu_t)
        self.logger.info("Mask tissue ratio = %.6f", tissue_ratio)

        if tissue_ratio <= 0.0: # error if no tissue is found
            self.logger.error("No tissue found in thumbnail mask.")
            raise NoTissueFoundError(
                f"No tissue found in slide thumbnail: {wsi_handle.image_path}"
            )

        return {
            "mask": tissue.astype(bool),
            "thumbnail_rgb": thumbnail_rgb,
            "mask_level": mask_level,
        }
    
    # Decides which WSI level is small enough for tissue masking
    def _select_mask_level(self, wsi_handle: WSIHandle) -> int:
        """
        Select the lowest resolution (highest level) whose longest side is <= mask_longest_side.
        If none satisfy it, choose the coarsest available level.
        """
        selected_level = len(wsi_handle.level_dimensions) - 1 # since there are 4 levels but starts from 0

        for level, dims in enumerate(wsi_handle.level_dimensions):
            longest_side = max(dims)
            if longest_side <= self.mask_longest_side:
                selected_level = level
                break

        return selected_level
    
    # Receives tissue mask and returns list of valid PatchRefs
    def _generate_patch_refs(
        self,
        wsi_handle: WSIHandle,
        tissue_mask: np.ndarray,
        mask_level: int,
        tissue_threshold: float, # min tissue fraction requires to accept a patch
    ) -> list[PatchRef]:
        """
        Deterministic grid generation:
        top-to-bottom, left-to-right

        Patch size and stride are defined in read-level coordinates.
        """
        read_w, read_h = wsi_handle.level_dimensions[self.read_level]
        read_ds = float(wsi_handle.level_downsamples[self.read_level]) # get downsampling factor of read level
        mask_ds = float(wsi_handle.level_downsamples[mask_level])      # get downsampling factor of mask level

        refs: list[PatchRef] = []

        # Edge handling: choose top-left corner so that the full patch still fits inside the image
        max_y = read_h - self.patch_size #If starting point is more than this value, it is invalid
        max_x = read_w - self.patch_size

        for y in range(0, max_y + 1, self.stride): # Iterating down the slide
            for x in range(0, max_x + 1, self.stride): # Iterating across each row
                if self._patch_tissue_fraction( # checking how much of this patch is tissue
                    x=x,
                    y=y,
                    tissue_mask=tissue_mask,
                    read_ds=read_ds,
                    mask_ds=mask_ds,
                ) >= tissue_threshold:
                    ref = wsi_handle.make_ref( #If valid, create PatchRef
                        pos=(x, y),
                        level=self.read_level,
                        dim=(self.patch_size, self.patch_size),
                    )
                    refs.append(ref)

        return refs

    def _patch_tissue_fraction(
        self,
        x: int,         #top left coordinate of the patch
        y: int,
        tissue_mask: np.ndarray,
        read_ds: float, #downsample factor of read level
        mask_ds: float, #downsample factor of mask level
    ) -> float:
        """
        Compute how much of this read-level patch is tissue in the thumbnail mask.

        read-level coords -> level-0 -> mask-level coords
        """
        scale = read_ds / mask_ds

        # Compute top left corner of the patch in mask-level coordinates
        mx0 = int(math.floor(x * scale))
        my0 = int(math.floor(y * scale))
        # Convert bottom-right boundary of patch in mask-level coordinates
        mx1 = int(math.ceil((x + self.patch_size) * scale))
        my1 = int(math.ceil((y + self.patch_size) * scale))

        h, w = tissue_mask.shape # Get height and width of tissue mask array
        # Keep all coordinates within valid bounds
        mx0 = max(0, min(mx0, w))
        mx1 = max(0, min(mx1, w))
        my0 = max(0, min(my0, h))
        my1 = max(0, min(my1, h))

        if mx1 <= mx0 or my1 <= my0:
            return 0.0

        patch_mask = tissue_mask[my0:my1, mx0:mx1] #Extract corresponding patch region from tissue mask
        if patch_mask.size == 0:
            return 0.0

        return float(patch_mask.mean())

    def _rgb_to_gray(self, rgb: np.ndarray) -> np.ndarray:
        rgb_f = rgb.astype(np.float32) / 255.0
        gray = (
            0.299 * rgb_f[..., 0]
            + 0.587 * rgb_f[..., 1]
            + 0.114 * rgb_f[..., 2]
        )
        return gray

    def _rgb_to_saturation(self, rgb: np.ndarray) -> np.ndarray:
        rgb_f = rgb.astype(np.float32) / 255.0
        cmax = np.max(rgb_f, axis=-1)
        cmin = np.min(rgb_f, axis=-1)
        sat = np.zeros_like(cmax, dtype=np.float32)
        denom = np.maximum(cmax, 1e-8)
        sat = (cmax - cmin) / denom
        sat[cmax <= 1e-8] = 0.0
        return sat

    # Receives grayscale image whose values are in range [0(very dark), 1(very bright)]; finds threshold that separates image into 2 groups
    def _otsu_threshold(self, gray: np.ndarray) -> float:
        """
        Otsu threshold on [0,1] grayscale.
        """
        values = np.clip((gray * 255.0).astype(np.uint8), 0, 255) #convert grayscale from [0,1] to [0,255]
        hist = np.bincount(values.ravel(), minlength=256).astype(np.float64) #make histogram of grayscale intensity bins
        total = hist.sum() #adds up all histogram counts
        if total <= 0: #if there are no pixels - error
            raise NoTissueFoundError("Empty grayscale histogram while building mask.")

        prob = hist / total #convert histogram to probabilities
        omega = np.cumsum(prob) #total % of all gray levels - eg. omega[t] : total % of all darker levels from 0 till t
        mu = np.cumsum(prob * np.arange(256)) #cumulative weighted sum of grayscale values up to threshold t
        mu_t = mu[-1] # mu[-1] is the last value of mu - the total mean intensity of the whole image

        sigma_b2 = np.zeros(256, dtype=np.float64) # creates an array of 256 zeros.
        denom = omega * (1.0 - omega)
        valid = denom > 0 # checking if denominator is positive
        sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid] #compute how well that threshold separates the image into two classes.

        threshold = np.argmax(sigma_b2) / 255.0 #the index where sigma_b2 is largest
        return float(threshold)

    def _clean_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphology cleanup:
        - opening
        - closing
        - fill holes
        - remove small objects
        """
        structure = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            dtype=bool
        ) # Create np with 1s

        cleaned = ndi.binary_opening(mask, structure=structure) #Remove if mask has tiny isolated tissue dots caused by noise
        cleaned = ndi.binary_closing(cleaned, structure=structure) #Repair if tissue regions have little breaks or narrow holes
        cleaned = ndi.binary_fill_holes(cleaned) #fill background holes surrounded by tissue
        cleaned = self._remove_small_objects(cleaned, self.min_mask_region_area) #remove connected components that are too small
        return cleaned.astype(bool)

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        if min_size <= 0:
            return mask.astype(bool)

        labeled, num_features = ndi.label(mask)
        if num_features == 0:
            return mask.astype(bool)

        counts = np.bincount(labeled.ravel())
        keep = counts >= min_size
        keep[0] = False  # background
        cleaned = keep[labeled]
        return cleaned.astype(bool)

    def _save_debug_outputs(
        self,
        wsi_handle: WSIHandle,
        thumbnail_rgb: Image.Image,
        tissue_mask: np.ndarray,
        mask_level: int,
        patch_refs: list[PatchRef],
        mode: str,
    ) -> None:
        # Specifying output file paths
        stem = Path(wsi_handle.image_path).stem
        thumb_path = self.result_dir / f"{stem}_{mode}_thumbnail.png"
        mask_path = self.result_dir / f"{stem}_{mode}_tissue_mask.png"
        overlay_path = self.result_dir / f"{stem}_{mode}_mask_overlay.png"
        json_path = self.result_dir / f"{stem}_{mode}_patch_refs.json"

        thumbnail_rgb.save(thumb_path) #save thumbnail (low res RGB image of slide)

        # Convert tisuse mask into savable image; binary array -> actual image file
        mask_img = Image.fromarray((tissue_mask.astype(np.uint8) * 255), mode="L")
        mask_img.save(mask_path)

        # Overlay: where the mask lies on top of the actual slide image - the image version of the mask
        overlay = self._make_overlay(thumbnail_rgb, tissue_mask)
        overlay.save(overlay_path)

        #Build a Python dictionary that will later be saved as JSON
        payload = {
            "image_path": str(wsi_handle.image_path),
            "mode": mode,
            "read_level": self.read_level,
            "mask_level": mask_level,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "patch_count": len(patch_refs),
            "patch_refs": [
                {
                    "image_path": str(ref.image_path),
                    "x": ref.x,
                    "y": ref.y,
                    "width": ref.width,
                    "height": ref.height,
                    "read_level": ref.read_level,
                    "mpp_x": ref.mpp_x,
                    "mpp_y": ref.mpp_y,
                }
                for ref in patch_refs
            ],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.logger.info("Saved thumbnail: %s", thumb_path)
        self.logger.info("Saved tissue mask: %s", mask_path)
        self.logger.info("Saved overlay: %s", overlay_path)
        self.logger.info("Saved patch refs json: %s", json_path)

    def _make_overlay(self, thumbnail_rgb: Image.Image, tissue_mask: np.ndarray) -> Image.Image:
        """
        Make a simple green overlay on top of thumbnail where tissue is detected.
        """
        base = np.asarray(thumbnail_rgb.convert("RGB"), dtype=np.uint8).copy()
        overlay = base.copy()

        green = np.zeros_like(base, dtype=np.uint8)
        green[..., 1] = 255

        alpha = 0.35
        tissue = tissue_mask.astype(bool)

        overlay[tissue] = (
            (1.0 - alpha) * base[tissue] + alpha * green[tissue]
        ).astype(np.uint8)

        return Image.fromarray(overlay, mode="RGB")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WSI Patch Sampler")

    parser.add_argument("--image", type=str, required=True, help="Path to .tif/.tiff WSI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default="inference",
        help="Sampling mode"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in read-level pixels"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in read-level pixels (default: patch_size // 4)"
    )
    parser.add_argument(
        "--read-level",
        type=int,
        default=0,
        help="WSI level used for patch coordinate generation"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Maximum number of patches (mainly useful for training)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training subsampling"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="result",
        help="Directory for logs/debug outputs"
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save thumbnail / tissue mask / overlay / patch refs json"
    )
    parser.add_argument(
        "--no-strict-mpp-check",
        action="store_true",
        help="Disable strict error on missing MPP metadata"
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    wsi_handle = open_wsi_handle(args.image)

    sampler = PatchSampler(
        patch_size=args.patch_size,
        stride=args.stride,
        read_level=args.read_level,
        training_tissue_threshold=0.7,
        inference_tissue_threshold=0.5,
        mask_longest_side=2048,
        mask_saturation_threshold=0.05,
        mask_white_threshold=0.92,
        morphology_kernel_size=5,
        min_mask_region_area=256,
        strict_mpp_check=not args.no_strict_mpp_check,
        result_dir=args.result_dir,
    )

    patch_refs = sampler.sample(
        wsi_handle=wsi_handle,
        mode=args.mode,
        max_patches=args.max_patches,
        seed=args.seed,
        save_debug=args.save_debug,
    )

    print(f"Sampled {len(patch_refs)} patches from: {wsi_handle.image_path}")
    print(f"Log file saved to: {sampler.log_path}")


if __name__ == "__main__":
    main()
