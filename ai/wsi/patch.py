from dataclasses import dataclass

import numpy as np

from ai.wsi.patch_ref import PatchRef

@dataclass
class Patch:
    ref: PatchRef
    img: np.ndarray # [C, H, W]

    def __post_init__(self) -> None:
      if not isinstance(self.ref, PatchRef): #should be a PatchRef
          raise TypeError(f"ref는 PatchRef 타입이어야 합니다. 입력 타입: {type(self.ref).__name__}")

      if not isinstance(self.img, np.ndarray): # should be an array
          raise TypeError(f"img는 numpy.ndarray 타입이어야 합니다. 입력 타입: {type(self.img).__name__}")

      if self.img.ndim != 3: #should have 3 dimensions
          raise ValueError(f"img는 3차원 [C, H, W]이어야 합니다. 입력 shape: {self.img.shape}")

      if self.img.shape[0] != 3:
          raise ValueError(
              f"img는 CHW 형식의 3채널 이미지여야 합니다. 입력 shape: {self.img.shape}"
          )

      if self.img.shape[1] <= 0 or self.img.shape[2] <= 0:
          raise ValueError(f"img의 높이와 너비는 0보다 커야 합니다. 입력 shape: {self.img.shape}")
