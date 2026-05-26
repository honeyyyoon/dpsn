import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

class Metric:
    def __init__(
        self,
        use_ssim: bool = True,
        use_psnr: bool = True,
        use_fid: bool = True,
        target_patch: np.ndarray | torch.Tensor | None = None,
        precision: int = 6
    ):
        self.use_ssim = use_ssim
        self.use_psnr = use_psnr
        self.use_fid = use_fid
        self.fid_device = torch.device("cpu")
        self.precision = precision

        if self.use_fid:
            self.fid = FrechetInceptionDistance(
                feature=2048,
                normalize=False,
            ).to(self.fid_device)

            if target_patch is None:
                raise ValueError("FID needs target patch but got None")
            
            if target_patch.ndim == 3:
                target_patch = target_patch[np.newaxis, ...]
            
            tgt_imgs = self._to_uint8_images(target_patch).to(self.fid_device) # [B, C, H, W]
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
            norm_imgs = norm_imgs.to(self.fid_device)
            self.fid.update(norm_imgs, real=False)

    def evaluate_torch(
        self,
        source_patch: torch.Tensor,
        output_patch: torch.Tensor,
    ) -> None:
        if source_patch.shape != output_patch.shape:
            raise ValueError(
                f"source_patch and output_patch must have the same shape, got "
                f"{source_patch.shape} vs {output_patch.shape}"
            )

        with torch.no_grad():
            source_patch = self._to_bchw_tensor(source_patch)
            output_patch = self._to_bchw_tensor(output_patch)

            source_float = self._to_metric_float(source_patch)
            output_float = self._to_metric_float(output_patch)

            if self.use_ssim:
                score = structural_similarity_index_measure(
                    output_float,
                    source_float,
                    gaussian_kernel=False,
                    kernel_size=self._ssim_kernel_size(source_float),
                    reduction="none",
                    data_range=255.0,
                )
                self.scores["ssim"] += float(score.detach().sum().cpu().item())
                self.counts["ssim"] += int(score.numel())

            if self.use_psnr:
                mse = torch.mean(
                    (source_float - output_float) ** 2,
                    dim=tuple(range(1, source_float.ndim)),
                )
                score = 10.0 * torch.log10((255.0 ** 2) / mse)
                self.scores["psnr"] += float(score.detach().sum().cpu().item())
                self.counts["psnr"] += int(score.numel())

            if self.use_fid:
                norm_imgs = self._to_uint8_images(output_patch).to(self.fid_device)
                self.fid.update(norm_imgs, real=False)

    def finalize(self) -> dict:
        if self.use_fid:
            self.fid = self.fid.to(self.fid_device)
            self.scores['fid'] = float(self.fid.compute().item())

        return {
            "ssim": round(self.scores['ssim'] / self.counts['ssim'], 6) if self.use_ssim else None,
            "psnr": round(self.scores['psnr'] / self.counts['psnr'], 6) if self.use_psnr else None,
            "fid": round(self.scores['fid'], 6) if self.use_fid else None,
        }

    def _to_bchw_tensor(self, patch: torch.Tensor) -> torch.Tensor:
        if not isinstance(patch, torch.Tensor):
            raise TypeError(f"patch must be a torch.Tensor, got {type(patch).__name__}")
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        if patch.ndim != 4:
            raise ValueError(f"patch must have shape [B, C, H, W], got {tuple(patch.shape)}")
        if patch.shape[1] != 3:
            raise ValueError(f"patch must have 3 channels in CHW format, got {tuple(patch.shape)}")

        return patch

    def _to_metric_float(self, patch: torch.Tensor) -> torch.Tensor:
        patch = patch.to(dtype=torch.float32)
        if patch.max() <= 1.0:
            patch = patch * 255.0

        return patch.clamp(0, 255)

    def _to_uint8_images(
        self,
        patch: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch)
        patch = self._to_bchw_tensor(patch)
        if patch.dtype == torch.uint8:
            return patch

        return self._to_metric_float(patch).round().to(dtype=torch.uint8)

    def _ssim_kernel_size(self, patch: torch.Tensor) -> int:
        kernel_size = min(7, int(patch.shape[-2]), int(patch.shape[-1]))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            raise ValueError(
                f"SSIM needs patch height and width >= 3, got {tuple(patch.shape[-2:])}"
            )

        return kernel_size
