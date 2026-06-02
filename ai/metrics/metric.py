import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

from ai.metrics.custom import CustomStainMetric


class MetricError(RuntimeError):
    """Base class for metric calculation errors."""


class MetricInputError(MetricError):
    """Raised when metric input patches are invalid."""


class MissingTargetPatchError(MetricInputError):
    """Raised when a target-dependent metric is enabled without a target patch."""


class MetricShapeError(MetricInputError):
    """Raised when metric input shapes are invalid or incompatible."""


class Metric:
    def __init__(
        self,
        use_ssim: bool = True,
        use_psnr: bool = True,
        use_fid: bool = True,
        use_custom: bool = False,
        target_patch: np.ndarray | torch.Tensor | None = None,
        fid_feature: int = 64,
        precision: int = 6
    ):
        self.use_ssim = use_ssim
        self.use_psnr = use_psnr
        self.use_fid = use_fid
        self.use_custom = use_custom
        self.fid_device = torch.device("cpu")
        self.fid_feature = fid_feature
        self.precision = precision

        if (self.use_fid or self.use_custom) and target_patch is None:
            raise MissingTargetPatchError(
                "타겟 의존 메트릭을 계산하려면 target_patch가 필요합니다."
            )

        if self.use_fid:
            self.fid = FrechetInceptionDistance(
                feature=self.fid_feature,
                normalize=False,
            ).to(self.fid_device)

            if target_patch.ndim == 3:
                target_patch = target_patch[np.newaxis, ...]
            
            tgt_imgs = self._to_uint8_images(target_patch).to(self.fid_device) # [B, C, H, W]
            self.fid.update(tgt_imgs, real=True)

        if self.use_custom:
            self.custom_metric = CustomStainMetric(
                target_patch=target_patch,
            )

        self.scores = {
            "ssim": 0.,
            "psnr": 0.,
            "fid": 0.,
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
            raise MetricShapeError(
                f"source_patch와 output_patch의 shape이 같아야 합니다. "
                f"입력 shape: {source_patch.shape} vs {output_patch.shape}"
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

        if self.use_custom:
            self.custom_metric.evaluate(source_patch, output_patch)

    def evaluate_torch(
        self,
        source_patch: torch.Tensor,
        output_patch: torch.Tensor,
    ) -> None:
        if source_patch.shape != output_patch.shape:
            raise MetricShapeError(
                f"source_patch와 output_patch의 shape이 같아야 합니다. "
                f"입력 shape: {source_patch.shape} vs {output_patch.shape}"
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

            if self.use_custom:
                self.custom_metric.evaluate(source_patch, output_patch)

    def finalize(self) -> dict:
        if self.use_fid:
            self.fid = self.fid.to(self.fid_device)
            self.scores['fid'] = max(0.0, float(self.fid.compute().item()))

        scores = {
            "ssim": round(self.scores['ssim'] / self.counts['ssim'], 6) if self.use_ssim else None,
            "psnr": round(self.scores['psnr'] / self.counts['psnr'], 6) if self.use_psnr else None,
            "fid": round(self.scores['fid'], 6) if self.use_fid else None,
        }
        if self.use_custom:
            scores.update(self.custom_metric.finalize())
        else:
            scores.update(
                {
                    "stain_preservation_corr": None,
                    "normalized_target_stain_angle_deg": None,
                    "source_target_stain_angle_deg": None,
                    "stain_angle_improvement_deg": None,
                    "custom_structure_score": None,
                    "custom_color_score": None,
                    "source_stain_rank": None,
                    "normalized_stain_rank": None,
                    "target_stain_rank": None,
                }
            )

        return scores

    def _to_bchw_tensor(self, patch: torch.Tensor) -> torch.Tensor:
        if not isinstance(patch, torch.Tensor):
            raise MetricInputError(
                f"patch는 torch.Tensor 타입이어야 합니다. 입력 타입: {type(patch).__name__}"
            )
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        if patch.ndim != 4:
            raise MetricShapeError(
                f"patch는 [B, C, H, W] shape이어야 합니다. 입력 shape: {tuple(patch.shape)}"
            )
        if patch.shape[1] != 3:
            raise MetricShapeError(
                f"patch는 CHW 형식의 3채널이어야 합니다. 입력 shape: {tuple(patch.shape)}"
            )

        return patch

    def _to_metric_float(self, patch: torch.Tensor) -> torch.Tensor:
        patch = patch.to(dtype=torch.float32)
        if float(patch.max().detach().cpu().item()) <= 1.0:
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
            raise MetricShapeError(
                f"SSIM 계산에는 높이와 너비가 각각 3 이상인 patch가 필요합니다. 입력 크기: {tuple(patch.shape[-2:])}"
            )

        return kernel_size
