import numpy as np
from skimage.metrics import structural_similarity as ssim

from ai.metrics.base import Metric


class SSIM(Metric):
    def evaluate(
        self,
        origin_patch: np.ndarray,
        normalized_patch: np.ndarray
    ) -> float:
        if origin_patch.shape != normalized_patch.shape:
            raise ValueError(
                f"origin_patch and normalized_patch must have the same shape, got "
                f"{origin_patch.shape} vs {normalized_patch.shape}"
            )

        if origin_patch.ndim == 4:
            scores = [
                ssim(
                    origin_patch[i],
                    normalized_patch[i],
                    channel_axis=0,
                    data_range=255,
                )
                for i in range(origin_patch.shape[0])
            ]
            return float(np.mean(scores))

        if origin_patch.ndim == 3:
            return float(
                ssim(
                    origin_patch,
                    normalized_patch,
                    channel_axis=0,
                    data_range=255,
                )
            )

        raise ValueError(
            f"SSIM expects CHW or BCHW arrays, got shape {origin_patch.shape}"
        )
