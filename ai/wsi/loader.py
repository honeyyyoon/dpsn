from __future__ import annotations
from pathlib import Path

import numpy as np
import openslide

from ai.wsi.handle import WSIHandle
from ai.wsi.loaders.openslide_loader import OpenSlideLoader
from ai.wsi.loaders.pillow_loader import PillowLoader
from ai.wsi.loaders.tifffile_loader import TiffFileLoader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef


# 사용자로부터 Image path를 전달받아서, 
# 다양한 wsi format(.tif ...)에 따라 WSIHandle을 구성하고 리턴
def open_wsi_handle(image_path: str | Path) -> WSIHandle:
    image_path = Path(image_path)
    
    if openslide.OpenSlide.detect_format(image_path):
        try:
            return OpenSlideLoader.open_wsi_handle(image_path)
        except:
            try:
                return TiffFileLoader.open_wsi_handle(image_path)
            except:
                raise ValueError()
    else:
        return PillowLoader.open_wsi_handle(image_path)


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
        except:
            try:
                return TiffFileLoader.load_patch(ref)
            except:
                raise ValueError()
    else:
        return PillowLoader.load_patch(ref)