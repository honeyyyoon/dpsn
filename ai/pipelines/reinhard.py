from collections import defaultdict
import logging
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from skimage import color
from tqdm import tqdm

from ai.metrics.base import Metric
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import load_patches_from_image, load_patch, open_wsi_handle
from ai.wsi.writer import ZarrWSIWriter

class Reinhard(ModelPipeline):
    target_sampler: PatchSampler

    def __init__(
        self, 
        logger: logging.Logger,
        batch_size: int = 64,
        patch_size: int = 256,
        max_sample_patches: int = 16,
        max_iteration: int = 32
    ):
        super().__init__(logger=logger)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.max_sample_patches = int(max_sample_patches)
        self.max_iteration = int(max_iteration)

    def run(
        self,
        src_img_path: Path,
        result_path: Path, 
        target_img_path: Path | None,
        metrics: dict[str, Metric]
    ) -> PipelineResult:
        self.logger.info("Run Reinhard")
        
        if target_img_path is None:
            raise ValueError("Reinhard needs a target image.")
        
        self.logger.info("Read source image")
        src_wsi_handle = open_wsi_handle(src_img_path)
        self.logger.info(f"Size: {src_wsi_handle.dim[0]} x {src_wsi_handle.dim[1]}")
        self.logger.info(f"Level: 0 - {src_wsi_handle.max_level}")
        self.logger.info(f"Mpp: {src_wsi_handle.mpp}")

        self.logger.info("Read target image")
        target_wsi_handle = open_wsi_handle(target_img_path)
        self.logger.info(f"Size: {target_wsi_handle.dim[0]} x {target_wsi_handle.dim[1]}")
        self.logger.info(f"Level: 0 - {target_wsi_handle.max_level}")
        self.logger.info(f"Mpp: {target_wsi_handle.mpp}")

        src_thumb_ref = src_wsi_handle.make_ref((0, 0), src_wsi_handle.max_level)
        target_thumb_ref = target_wsi_handle.make_ref((0, 0), target_wsi_handle.max_level)
        src_thumb = Image.fromarray(load_patch(src_thumb_ref).img.transpose([1, 2, 0]))
        target_thumb = Image.fromarray(load_patch(target_thumb_ref).img.transpose([1, 2, 0]))
        
        src_thumb.save("result/source.png")
        target_thumb.save("result/target.png")

        patch_sampler = PatchSampler()

        self.logger.info("Sample source image")
        src_ref_list = patch_sampler.sample(
            src_wsi_handle, 
            max_patches=self.max_sample_patches,
            mode="training",
            save_debug=False
        )
        src_ref_patches = [load_patch(ref) for ref in src_ref_list]
        self.logger.info(f"Patches: {len(src_ref_patches)}")

        self.logger.info("Sample target image")
        tgt_ref_list = patch_sampler.sample(
            target_wsi_handle, 
            max_patches=self.max_sample_patches,
            mode="training",
            save_debug=False
        )
        tgt_ref_patches = [load_patch(ref) for ref in tgt_ref_list]
        self.logger.info(f"Patches: {len(tgt_ref_patches)}")

        tgt_images = np.stack([patch.img for patch in tgt_ref_patches], axis=0)
        src_images = np.stack([patch.img for patch in src_ref_patches], axis=0)

        self.logger.info("Get Reinhard Stats")
        target_means, target_stds = self.get_reinhard_stats(tgt_images)
        self.logger.info(f"Target stat: means={target_means.round(2)}, stds={target_stds.round(2)}")
        src_means, src_stds = self.get_reinhard_stats(src_images)
        self.logger.info(f"Source stat: means={src_means.round(2)}, stds={src_stds.round(2)}")

        level = 0
        size = src_wsi_handle.level_dimensions[level]
        expected_iteration = (size[0] // self.patch_size) * (size[1] // self.patch_size) // self.batch_size

        while expected_iteration > self.max_iteration and level < src_wsi_handle.max_level:
            level += 1
            size = src_wsi_handle.level_dimensions[level]
            expected_iteration = (size[0] // self.patch_size) * (size[1] // self.patch_size) // self.batch_size
        
        if expected_iteration > self.max_iteration:
            ValueError(f"Image is too big! Expected iteration: {expected_iteration}, Max iteration: {self.max_iteration}")
        
        self.logger.info(f"Grid Sample from Source Image: patch_size={self.patch_size}, read_level={level}, downsamples={src_wsi_handle.level_downsamples[level]}")
        grid_sampler = GridSampler(patch_size=self.patch_size, read_level=level)

        src_refs = grid_sampler.sample(src_wsi_handle)
        self.logger.info(f"Sampled: {len(src_refs)}")

        writer = ZarrWSIWriter(
            result_path,
            src_wsi_handle.level_dimensions[0][0], 
            src_wsi_handle.level_dimensions[0][1],
            level_downsample=src_wsi_handle.level_downsamples[level],
            tile_size = src_refs[0].width
        )

        timer = defaultdict(float)
        scores = defaultdict(float)

        iter = 0
        for idx in tqdm(range(0, len(src_refs), self.batch_size)):
            iter += 1
            t0 = time.time()
            batch_ref = src_refs[idx:idx + self.batch_size]
            patches = [load_patch(ref) for ref in batch_ref]
            timer['load'] += time.time() - t0

            patches = np.stack([patch.img for patch in patches], axis=0)

            t0 = time.time()
            new_patches = self.transform_image(patches, target_means, target_stds, src_means, src_stds)
            timer['transform'] += time.time() - t0

            t0 = time.time()
            for key, metric in metrics.items():
                scores[key] += metric.evaluate(patches, new_patches)

            t0 = time.time()
            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i].astype(np.uint8))
            timer['writer'] += time.time() - t0

        for key, score in scores.items():
            scores[key] /= iter
        scores = dict(scores)
        
        self.logger.info("Finish normalize")
        self.logger.info(f"Elapsed time: load({timer['load']:.4f}s), transform({timer['transform']:.4f}s), writer({timer['writer']:.4f}s)")
        self.logger.info(f"Metric: ssim({scores['ssim']:.4f}), psnr({scores['psnr']:.4f}), fid({scores['fid']:.4f})")
        
        output_path = writer.finalize()
        writer.close()
        # self.logger.info(f"Save Normalized Image: {image.width} x {image.height}")
        

        return PipelineResult(
            output_path=output_path,
            scores=scores,
            thumbnail_path=output_path,
        )
        
    
    def get_reinhard_stats(self, image: np.ndarray):
        """
        image: RGB image, shape (B, C, H, W), uint8 or float
        return:
            means: [L_mean, a_mean, b_mean]
            stds:  [L_std,  a_std,  b_std]
        """
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        image = image.transpose([0, 2, 3, 1]) # [H, W, C]

        lab = color.rgb2lab(image / 255.0)  # L: [0,100], a/b: roughly [-128,127]

        means = lab.reshape(-1, 3).mean(axis=0)
        stds = lab.reshape(-1, 3).std(axis=0)

        return means, stds
    
    def transform_image(
        self, 
        image: np.ndarray, 
        target_means: np.ndarray,
        target_stds: np.ndarray,
        src_means: np.ndarray,
        src_stds: np.ndarray, 
    ) -> np.ndarray:
        image = image.transpose([0, 2, 3, 1])

        lab = color.rgb2lab(image / 255.0)
        lab = (lab - src_means)/src_stds * target_stds + target_means
        
        lab[:, 0, :, :] = np.clip(lab[:, 0, :, :], 0, 100)
        lab[:, 1:, :, :] = np.clip(lab[:, 1:, :, :], -128, 127)

        image = np.clip(color.lab2rgb(lab) * 255.0, 0, 255)
        image = image.transpose([0, 3, 1, 2])

        return image
