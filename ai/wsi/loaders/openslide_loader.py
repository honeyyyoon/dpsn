from __future__ import annotations
from pathlib import Path

import numpy as np
import openslide

from ai.wsi.handle import UnsupportedWSIFormatError, WSIHandle
from ai.wsi.loaders.base import Loader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef

class OpenSlideLoader(Loader):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def open_wsi_handle(img_path: str | Path) -> "WSIHandle":
        image_path = Path(img_path)
        if openslide.OpenSlide.detect_format(image_path):
            with openslide.OpenSlide(image_path) as slide:
                width, height = slide.dimensions
                props = slide.properties
                mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
                mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)

                # mpp_x와 mpp_y가 항상 제공되지 않을 수 있다.
                # 이때 우선 -1을 저장하는데, 추후 처리가 필요할 듯 하다.
                mpp_x = float(mpp_x if mpp_x is not None else "-1")
                mpp_y = float(mpp_y if mpp_y is not None else "-1")
                
                level_dimensions = slide.level_dimensions # 각 레벨별 Image Size
                level_downsamples = slide.level_downsamples # 레벨 0에 비해 얼마나 downsample 되었는지
            
                return WSIHandle(
                    image_path = image_path,
                    dim = (width, height),
                    mpp = (mpp_x, mpp_y),
                    level_dimensions = level_dimensions,
                    level_downsamples = level_downsamples,
                )
        else:
            raise UnsupportedWSIFormatError(
                f"OpenSlide does not support file format: {img_path}"
            )
    
    @staticmethod
    def load_patch(patch_ref: PatchRef) -> Patch:
        # OpenSlide expects:
        # - location in level-0 coordinates
        # - size in pixels at ref.read_level
        ref = patch_ref
        with openslide.OpenSlide(str(ref.image_path)) as slide: # opens the WSI file and reads one patch from it
            region = slide.read_region(
                ref.level0_pos, # Top-left patch location in level-0 coordinates
                ref.read_level, # Which level to read from
                ref.read_size,  # Patch size in pixels at read_level
            )

        # OpenSlide returns a PIL image in RGBA. Convert to RGB - drop alpha channel
        rgb = region.convert("RGB")

        # PIL gives HWC (height, width, channel); Patch expects [C, H, W], so transpose the axes.
        img_hwc = np.asarray(rgb, dtype=np.uint8)
        img_chw = np.transpose(img_hwc, (2, 0, 1))

        return Patch(
            ref=ref,
            img=img_chw,
        )
