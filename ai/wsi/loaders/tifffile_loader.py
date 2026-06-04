from __future__ import annotations
from pathlib import Path

import numpy as np
import tifffile

from ai.wsi.handle import WSIHandle
from ai.wsi.loaders.base import Loader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef


class TiffFileLoaderError(RuntimeError):
    """Base class for tifffile loader errors."""


class UnsupportedTiffShapeError(TiffFileLoaderError):
    """Raised when a TIFF page shape cannot be interpreted as an image."""


class TiffFileLoader(Loader):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def open_wsi_handle(img_path: Path) -> WSIHandle:
        with tifffile.TiffFile(img_path) as tif:
            page = tif.pages[0]

            if len(page.shape) == 2:
                h, w = page.shape
            elif len(page.shape) == 3:
                h, w = page.shape[:2]
            else:
                raise UnsupportedTiffShapeError(
                    f"지원하지 않는 TIFF shape입니다: {page.shape}"
                )

            dim = (w, h)

            return WSIHandle(
                image_path=img_path,
                dim=dim,
                mpp=(-1, -1),
                level_dimensions=(dim,),
                level_downsamples=(1,)
            )
    
    @staticmethod
    def load_patch(patch_ref: PatchRef) -> Patch:
        img = tifffile.imread(patch_ref.image_path)
        if img.ndim != 3:
            raise UnsupportedTiffShapeError(
                f"패치 로딩에 지원하지 않는 TIFF 이미지 shape입니다: {img.shape}"
            )

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        img_chw = np.transpose(img, (2, 0, 1))

        pos_y = patch_ref.y // patch_ref.downsample
        pos_x = patch_ref.x // patch_ref.downsample

        img_cropped = img_chw[:, pos_y:pos_y + patch_ref.height, pos_x:pos_x + patch_ref.width]

        return Patch(
            ref=patch_ref,
            img=img_cropped
        )
