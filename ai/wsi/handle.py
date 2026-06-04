from dataclasses import dataclass
from pathlib import Path

from ai.wsi.patch_ref import PatchRef

class WSIHandleError(RuntimeError):
    """Base class for WSI Handle errors."""


class UnsupportedWSIFormatError(WSIHandleError):
    """Raised when the file format is not supported by the WSI backend."""

@dataclass
class WSIHandle:
    image_path: Path          #file path of WSI
    dim: tuple[int, int]      #full image size at lv 0, [w, h]
    mpp: tuple[float, float]  #microns per pixel for x, y
    level_dimensions: tuple   #image size for each pyramid level eg. (80000, 60000), (20000, 15000),...
    level_downsamples: tuple  #how much each level is downsampled relative to lv 0 eg. 4 : downsampled by 4

    @property
    def max_level(self): return len(self.level_dimensions) - 1
    
    # 해당 레벨에서, Image의 (pos[0], pos[1]) 부터 (pos[0] + dim[0], pos[1] + dim[1]) 까지의 범위를 crop. PatchRef object 를 만든다.
    def make_ref(self, pos: tuple[int, int] = (0, 0), level: int = 0, dim: tuple[int, int] | None = None) -> PatchRef:
        level_count = len(self.level_dimensions) #count how many levels exist

        if not (0 <= level < level_count): #check if requested level is valid
            raise ValueError(
                f"level {level}은 0 이상 {level_count - 1} 이하이어야 합니다."
            )
        
        # Image의 범위가 유효한지 확인
        img_width, img_height = self.level_dimensions[level]
        dim = dim or (img_width, img_height)

        if not (0 <= pos[0] <= img_width - dim[0]):
            raise ValueError(
                f"너비 범위가 유효하지 않습니다: {(pos[0], pos[0] + dim[0])}, 허용 범위: {(0, img_width)}"
            )
        
        if not (0 <= pos[1] <= img_height - dim[1]):
            raise ValueError(
                f"높이 범위가 유효하지 않습니다: {(pos[1], pos[1] + dim[1])}, 허용 범위: {(0, img_height)}"
            )
        
        downsample = float(self.level_downsamples[level]) #get downsample factor for that lv eg. 1 px at lv2 = 16 px at lv0

        # OpenSlide read_region() uses level-0 coordinates for location - convert patch position to lv0 coordinates
        level0_x = int(round(pos[0] * downsample))
        level0_y = int(round(pos[1] * downsample))
        
        return PatchRef(
            image_path=self.image_path,
            x=level0_x,
            y=level0_y,
            width=int(dim[0]),
            height=int(dim[1]),
            read_level=int(level),
            downsample=int(self.level_downsamples[level]),
            mpp_x=float(self.mpp[0]),
            mpp_y=float(self.mpp[1]),
        )
    
