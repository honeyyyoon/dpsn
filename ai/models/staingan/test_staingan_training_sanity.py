"""
Running example:
./.venv/bin/python -m ai.models.staingan.test_staingan_training_sanity \
--dataset-dir /mnt/Disk1/dpsn_datasets/multiscanner_dataset \
--image-size 128 \
--device cpu

"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from ai.models.staingan.multidomain_wsi_dataset import MultiDomainWSIPatchDataset
from ai.models.staingan.train_staingan import (
    StainGANTrainingConfig,
    content_loss,
    create_models,
    select_device,
)


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


def build_synthetic_multiscanner_dataset(
    root: Path,
    sample_count: int = 4,
    image_size: int = 384,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    tissue = ((xx - image_size // 2) ** 2 + (yy - image_size // 2) ** 2) < (image_size // 3) ** 2

    for sample_idx in range(1, sample_count + 1):
        base = np.full((image_size, image_size, 3), 245, dtype=np.uint8)
        base[tissue, 0] = 150 + sample_idx * 8
        base[tissue, 1] = 70
        base[tissue, 2] = 150
        for scanner_idx, scanner in enumerate(SCANNERS):
            image = base.copy().astype(np.int16)
            image[..., 0] += scanner_idx * 8
            image[..., 1] -= scanner_idx * 4
            image = np.clip(image, 0, 255).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(root / f"scc_{sample_idx:02d}_{scanner}.png")
    return root


def run_sanity_check(
    dataset_dir: Path | None,
    image_size: int,
    device_name: str,
) -> None:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if dataset_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        dataset_dir = build_synthetic_multiscanner_dataset(Path(temp_dir.name))

    dataset = MultiDomainWSIPatchDataset(
        dataset_dir=dataset_dir,
        canonical_domain="nz210",
        sample_ids=["01", "02", "03"],
        image_size=image_size,
        target_mpp=0.25,
        patches_per_source_slide=2,
        strict_mpp_check=False,
        seed=7,
        sampler_result_dir=Path(tempfile.gettempdir()) / "staingan_sanity_sampler",
    )
    sample = dataset[0]

    device = select_device(device_name, gpu_ids=())
    config = StainGANTrainingConfig(
        dataset_dir=dataset_dir,
        image_size=image_size,
        ngf=8,
        ndf=8,
        generator_blocks=1,
        strict_mpp_check=False,
        gpu_ids=(),
    )
    generator, discriminator = create_models(config, device)

    source = torch.from_numpy(np.stack([sample["source"], sample["source"]], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    canonical = torch.from_numpy(
        np.stack([sample["canonical"], sample["canonical"]], axis=0)
    ).to(device=device, dtype=torch.float32)

    fake = generator(source)
    same = generator(canonical)
    pred_real = discriminator(canonical)
    pred_fake = discriminator(fake)
    loss_content = content_loss(source, fake, "grayscale_l1")

    if fake.shape != source.shape:
        raise AssertionError(f"Generator shape mismatch: {fake.shape} vs {source.shape}")
    if same.shape != canonical.shape:
        raise AssertionError("Identity generator shape does not match canonical input.")
    if pred_real.ndim != 4 or pred_fake.ndim != 4:
        raise AssertionError("Patch discriminator must return 4D patch scores.")
    if not torch.isfinite(loss_content):
        raise AssertionError("Content loss must be finite.")

    print("StainGAN many-to-one training sanity check passed.")
    print(f"dataset size: {len(dataset)}")
    print(f"source domains: {dataset.source_domains}")
    print(f"canonical domain: {dataset.canonical_domain}")
    print(f"device: {device}")
    print(f"generator output shape: {tuple(fake.shape)}")
    print(f"discriminator output shape: {tuple(pred_real.shape)}")

    if temp_dir is not None:
        temp_dir.cleanup()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a StainGAN training sanity check.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_sanity_check(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
