from __future__ import annotations
from pathlib import Path

import openslide

from ai.wsi.handle import WSIHandle
from ai.wsi.loaders.openslide_loader import OpenSlideLoader
from ai.wsi.loaders.pillow_loader import PillowLoader
from ai.wsi.loaders.tifffile_loader import TiffFileLoader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef


class WSILoaderError(RuntimeError):
    """Base class for WSI loader dispatch errors."""


class WSIHandleOpenError(WSILoaderError):
    """Raised when no available backend can open a WSI/image file."""


class PatchLoadError(WSILoaderError):
    """Raised when no available backend can load a patch."""


# 사용자로부터 Image path를 전달받아서, 
# 다양한 wsi format(.tif ...)에 따라 WSIHandle을 구성하고 리턴
def open_wsi_handle(image_path: str | Path) -> WSIHandle:
    image_path = Path(image_path)
    
    if openslide.OpenSlide.detect_format(image_path):
        try:
            return OpenSlideLoader.open_wsi_handle(image_path)
        except Exception as openslide_error:
            try:
                return TiffFileLoader.open_wsi_handle(image_path)
            except Exception as tiff_error:
                raise WSIHandleOpenError(
                    f"Failed to open WSI file with OpenSlide or tifffile: {image_path}. "
                    f"OpenSlide error: {openslide_error}; tifffile error: {tiff_error}"
                ) from tiff_error

    try:
        return PillowLoader.open_wsi_handle(image_path)
    except Exception as pillow_error:
        raise WSIHandleOpenError(
            f"Failed to open image file with Pillow: {image_path}. "
            f"Pillow error: {pillow_error}"
        ) from pillow_error


def load_patch(ref: PatchRef) -> Patch:
    """
    Load a single patch from a WSI using the metadata stored in PatchRef.

    Returns
    -------
    Patch
        Patch image data in [C, H, W] RGB uint8 format.
    """
    if openslide.OpenSlide.detect_format(ref.image_path):
        try:
            return OpenSlideLoader.load_patch(ref)
        except Exception as openslide_error:
            try:
                return TiffFileLoader.load_patch(ref)
            except Exception as tiff_error:
                raise PatchLoadError(
                    f"Failed to load patch with OpenSlide or tifffile: {ref.image_path}. "
                    f"OpenSlide error: {openslide_error}; tifffile error: {tiff_error}"
                ) from tiff_error

    try:
        return PillowLoader.load_patch(ref)
    except Exception as pillow_error:
        raise PatchLoadError(
            f"Failed to load patch with Pillow: {ref.image_path}. "
            f"Pillow error: {pillow_error}"
        ) from pillow_error
