from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class PairedAlignedImageDataset(Dataset):
    """
    Paired aligned dataset for StainNet training.

    Each source image must have a matching target image with the same filename.
    The dataset returns RGB tensors in CHW format normalized to [-1, 1], which
    matches the original StainNet training/inference convention.
    """

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        image_size: int = 256,
        recursive: bool = True, #whether the dataset loader searches only the top-level folder or also all nested subfolders for images
        source_prefix: str = "A",
        target_prefix: str = "H",
        quality_filter: bool = True,
        min_tissue_fraction: float = 0.2,
        max_black_fraction: float = 0.1,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_size = int(image_size)
        self.recursive = bool(recursive)
        self.source_prefix = str(source_prefix)
        self.target_prefix = str(target_prefix)
        self.quality_filter = bool(quality_filter)
        self.min_tissue_fraction = float(min_tissue_fraction)
        self.max_black_fraction = float(max_black_fraction)

        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not self.target_dir.is_dir():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if len(self.source_prefix) != 1:
            raise ValueError(
                f"source_prefix must be a single character, got {self.source_prefix!r}"
            )
        if len(self.target_prefix) != 1:
            raise ValueError(
                f"target_prefix must be a single character, got {self.target_prefix!r}"
            )
        if not 0.0 <= self.min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be between 0 and 1.")
        if not 0.0 <= self.max_black_fraction <= 1.0:
            raise ValueError("max_black_fraction must be between 0 and 1.")

        source_files = self._list_images(self.source_dir)
        target_files = self._list_images(self.target_dir)

        source_map = self._build_file_map(
            source_files,
            expected_prefix=self.source_prefix,
        )
        target_map = self._build_file_map(
            target_files,
            expected_prefix=self.target_prefix,
        )

        missing_in_target = sorted(set(source_map) - set(target_map))
        missing_in_source = sorted(set(target_map) - set(source_map))
        if missing_in_target or missing_in_source:
            details: list[str] = []
            if missing_in_target:
                details.append(f"missing target files: {missing_in_target[:10]}")
            if missing_in_source:
                details.append(f"missing source files: {missing_in_source[:10]}")
            raise ValueError(
                "MITOS source/target folders must contain matching A... / H... pairs; "
                + "; ".join(details)
            )

        self.source_map = source_map
        self.target_map = target_map
        self.filenames = sorted(source_map.keys())

        if self.quality_filter:
            before_count = len(self.filenames)
            self.filenames = [
                filename
                for filename in self.filenames
                if self._is_quality_image(self.source_map[filename])
                and self._is_quality_image(self.target_map[filename])
            ]
            removed_count = before_count - len(self.filenames)
            if removed_count:
                print(
                    "[PairedAlignedImageDataset] filtered "
                    f"{removed_count}/{before_count} low-quality pair(s)",
                    flush=True,
                )

        if not self.filenames:
            raise ValueError(
                "No valid MITOS training pairs found. Expected source files like "
                f"{self.source_prefix}... and target files like {self.target_prefix}..."
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]: # get the source and target file of corresponding index
        filename = self.filenames[index]
        source = self._load_image(self.source_map[filename])
        target = self._load_image(self.target_map[filename])
        return source, target, filename
    
        # looks inside a folder and returns a sorted list of image file paths
    def _list_images(self, directory: Path) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        return sorted(
            [
                path
                for path in directory.glob(pattern)
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
        )
    
    # Building a pair-key -> path map where the pair key is the filename stem
    # after the first scanner prefix character. Files that do not match the
    # expected prefix are ignored so annotation/preview images are skipped.
    def _build_file_map(
        self,
        files: list[Path],
        expected_prefix: str,
    ) -> dict[str, Path]:
        file_map: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in files:
            key = self._pair_key(path, expected_prefix=expected_prefix)
            if key is None:
                continue
            if key in file_map:
                duplicates.append(key)
            file_map[key] = path

        if duplicates: #if duplicates exist, raise error
            dup_preview = ", ".join(sorted(set(duplicates))[:10])
            raise ValueError(
                "Duplicate MITOS pair keys are not supported for paired aligned training. "
                f"Found duplicates such as: {dup_preview}"
            )

        return file_map

    def _pair_key(self, path: Path, expected_prefix: str) -> str | None:
        stem = path.stem
        if not stem or stem[0] != expected_prefix:
            return None

        pair_key = stem[1:]
        if not pair_key:
            return None

        return pair_key
    
    # loads one image file from disk and converts it into the exact numeric format the model wants
    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB") # open and force into RGB format
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR) #resizes the image to a fixed square size(256 x 256)
        image_np = np.asarray(image, dtype=np.float32) / 255.0 # converts the PIL image into a NumPy array (dividing by 255 converts them to values 0-1)
        image_np = np.transpose(image_np, (2, 0, 1)) # Change from HWC to CHW
        image_np = (image_np - 0.5) * 2.0 # Normalize from [0,1] to [-1,1], because that's what stainnet expects
        return image_np.astype(np.float32) # return processed image

    def _is_quality_image(self, path: Path) -> bool:
        image = Image.open(path).convert("RGB")
        image.thumbnail((self.image_size, self.image_size), Image.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        max_rgb = rgb.max(axis=2)
        min_rgb = rgb.min(axis=2)
        saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-6)
        black_fraction = float((max_rgb < 0.06).mean())
        tissue_fraction = float(
            (
                (max_rgb > 0.08)
                & (max_rgb < 0.95)
                & (saturation > 0.04)
            ).mean()
        )
        return (
            tissue_fraction >= self.min_tissue_fraction
            and black_fraction <= self.max_black_fraction
        )
