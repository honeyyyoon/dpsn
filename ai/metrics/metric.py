import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

class Metric:
    def __init__(
        self,
        use_ssim: bool = True,
        use_psnr: bool = True,
        use_fid: bool = True,
        target_patch: np.ndarray | None = None
    ):
        self.use_ssim = use_ssim
        self.use_psnr = use_psnr
        self.use_fid = use_fid

        if self.use_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=False)

            if target_patch is None:
                raise ValueError("FID needs target patch but got None")
            
            if target_patch.ndim == 3:
                target_patch = target_patch[np.newaxis, ...]
            
            tgt_imgs = torch.from_numpy(target_patch).to(dtype=torch.uint8) # [B, C, H, W]
            self.fid.update(tgt_imgs, real=True)

        self.scores = {
            "ssim": 0.,
            "psnr": 0.,
            "fid": 0.
        }
        self.counts = {
            "ssim": 0,
            "psnr": 0
        }

    def evaluate(
        self, 
        source_patch: np.ndarray,
        output_patch: np.ndarray
    ) -> None:
        if source_patch.shape != output_patch.shape:
            raise ValueError(
                f"source_patch and output_patch must have the same shape, got "
                f"{source_patch.shape} vs {output_patch.shape}"
            )

        if source_patch.ndim == 3:
            source_patch = source_patch[np.newaxis, ...]
            output_patch = output_patch[np.newaxis, ...]
        
        source_patch = source_patch.transpose([0, 2, 3, 1]) # [B, H, W, C]
        output_patch = output_patch.transpose([0, 2, 3, 1]) # [B, H, W, C]

        if self.use_ssim:
            s = [
                ssim(
                    source_patch[i],
                    output_patch[i],
                    channel_axis=-1,
                    data_range=255,
                )
                for i in range(source_patch.shape[0])
            ]
            self.scores['ssim'] += np.sum(s)
            self.counts['ssim'] += len(s)
        
        if self.use_psnr:
            s = [
                psnr(
                    source_patch[i],
                    output_patch[i],
                    data_range=255,
                )
                for i in range(source_patch.shape[0])
            ]
            self.scores['psnr'] += np.sum(s)
            self.counts['psnr'] += len(s)
        
        if self.use_fid:
            norm_imgs = torch.from_numpy(output_patch).to(dtype=torch.uint8)
            norm_imgs = norm_imgs.permute([0, 3, 1, 2]) # [B, C, H, W]
            self.fid.update(norm_imgs, real=False)


    def finalize(self) -> dict:
        if self.use_fid:
            self.scores['fid'] = float(self.fid.compute().item())

        return {
            "ssim": self.scores['ssim'] / self.counts['ssim'] if self.use_ssim else None,
            "psnr": self.scores['psnr'] / self.counts['psnr'] if self.use_psnr else None,
            "fid": self.scores['fid'] if self.use_fid else None,
        }
