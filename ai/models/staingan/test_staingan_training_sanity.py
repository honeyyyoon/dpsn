from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from ai.models.staingan.train_staingan import (
    StainGANTrainingConfig,
    create_models,
    select_device,
)
from ai.models.staingan.unpaired_domain_dataset import UnpairedDomainImageDataset


def build_synthetic_domains(root: Path, count: int = 4, image_size: int = 256) -> tuple[Path, Path]:
    domain_a = root / "aperio"
    domain_b = root / "hamamatsu"
    domain_a.mkdir(parents=True, exist_ok=True)
    domain_b.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        image_a = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image_a[..., 0] = np.linspace(0, 255, image_size, dtype=np.uint8)
        image_a[..., 1] = (idx + 1) * 30
        image_b = np.rot90(image_a, k=1).copy()
        Image.fromarray(image_a, mode="RGB").save(domain_a / f"sample_{idx:03d}.png")
        Image.fromarray(image_b, mode="RGB").save(domain_b / f"sample_{idx:03d}.png")

    return domain_a, domain_b


def run_sanity_check(
    domain_a_dir: Path | None,
    domain_b_dir: Path | None,
    image_size: int,
    device_name: str,
) -> None:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if domain_a_dir is None or domain_b_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        domain_a_dir, domain_b_dir = build_synthetic_domains(
            Path(temp_dir.name),
            count=4,
            image_size=image_size,
        )

    dataset = UnpairedDomainImageDataset(
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        image_size=image_size,
    )
    sample_a, sample_b, _, _ = dataset[0]

    device = select_device(device_name)
    config = StainGANTrainingConfig(
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        image_size=image_size,
    )
    g_a2b, g_b2a, d_a, d_b = create_models(config, device)

    batch_a = torch.from_numpy(np.stack([sample_a, sample_a], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    batch_b = torch.from_numpy(np.stack([sample_b, sample_b], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )

    fake_b = g_a2b(batch_a)
    fake_a = g_b2a(batch_b)
    rec_a = g_b2a(fake_b)
    rec_b = g_a2b(fake_a)
    pred_a = d_a(batch_a)
    pred_b = d_b(batch_b)

    if fake_b.shape != batch_a.shape:
        raise AssertionError(f"Generator A->B shape mismatch: {fake_b.shape} vs {batch_a.shape}")
    if fake_a.shape != batch_b.shape:
        raise AssertionError(f"Generator B->A shape mismatch: {fake_a.shape} vs {batch_b.shape}")
    if rec_a.shape != batch_a.shape or rec_b.shape != batch_b.shape:
        raise AssertionError("Cycle reconstruction shapes do not match input shapes.")
    if pred_a.ndim != 4 or pred_b.ndim != 4:
        raise AssertionError("Patch discriminators must return 4D patch scores.")

    print("StainGAN training sanity check passed.")
    print(f"dataset size: {len(dataset)}")
    print(f"device: {device}")
    print(f"generator output shape: {tuple(fake_b.shape)}")
    print(f"discriminator output shape: {tuple(pred_a.shape)}")

    if temp_dir is not None:
        temp_dir.cleanup()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a StainGAN training sanity check.")
    parser.add_argument("--domain-a-dir", type=Path, default=None)
    parser.add_argument("--domain-b-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_sanity_check(
        domain_a_dir=args.domain_a_dir,
        domain_b_dir=args.domain_b_dir,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
