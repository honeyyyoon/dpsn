from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ai.wsi.handle import WSIHandle
from ai.wsi.loaders.base import Loader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef

class TiffFileLoader(Loader):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def open_wsi_handle(img_path: Path) -> WSIHandle:
        with Image.open(img_path) as img:
            return WSIHandle(
                image_path=img_path,
                dim=(img.width, img.height),
                mpp=(-1, -1),
                level_dimensions=(img.width, img.height),
                level_downsamples=(1,)
            )
    
    @staticmethod
    def load_patch(patch_ref: PatchRef) -> Patch:
        with Image.open(patch_ref.image_path) as img:
            # Open the image and crop the specified region
            img_cropped = img.crop((
                patch_ref.x,
                patch_ref.y,
                patch_ref.x + patch_ref.width,
                patch_ref.y + patch_ref.height
            ))

            img_chw = np.array(img_cropped).transpose(2, 0, 1) # Convert to CHW format

            return Patch(
                ref=patch_ref,
                img=img_chw
            )
