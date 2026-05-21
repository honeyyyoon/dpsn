"""
Run example:
.venv/bin/python -m ai.models.stainswin.test_stainswin_training_sanity

or

.venv/bin/python -m ai.models.stainswin.test_stainswin_training_sanity \
  --source-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio \
  --target-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu \
  --image-size 128 \
  --device auto

"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.optim import AdamW

from ai.models.stainnet.paired_aligned_dataset import PairedAlignedImageDataset
from ai.models.stainswin.train_stainswin import (
    CharbonnierLoss,
    StainSWINTrainingConfig,
    create_model,
    select_device,
)


def build_synthetic_dataset(
    root: Path,
    count: int = 4,
    image_size: int = 128,
) -> tuple[Path, Path]:
    source_dir = root / "source"
    target_dir = root / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        base[..., 0] = (idx + 1) * 20
        base[..., 1] = np.linspace(0, 255, image_size, dtype=np.uint8)[:, None]
        base[..., 2] = np.linspace(255, 0, image_size, dtype=np.uint8)[None, :]

        source = base
        target = np.rot90(base, k=1).copy()

        suffix = f"sample_{idx:03d}"
        Image.fromarray(source, mode="RGB").save(source_dir / f"A{suffix}.png")
        Image.fromarray(target, mode="RGB").save(target_dir / f"H{suffix}.png")

    Image.fromarray(source, mode="RGB").save(source_dir / "annotation_overlay.png")
    Image.fromarray(target, mode="RGB").save(target_dir / "thumb_preview.png")

    return source_dir, target_dir


def run_sanity_check(
    source_dir: Path | None,
    target_dir: Path | None,
    image_size: int,
    device_name: str,
) -> None:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if source_dir is None or target_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        source_dir, target_dir = build_synthetic_dataset(
            Path(temp_dir.name),
            count=4,
            image_size=image_size,
        )

    dataset = PairedAlignedImageDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=image_size,
    )
    source, target, filename = dataset[0]

    model = create_model(
        StainSWINTrainingConfig(
            train_source_dir=source_dir,
            train_target_dir=target_dir,
            image_size=image_size,
            embed_dim=30,
            num_heads=6,
            num_res_blocks=2,
            stbs_per_block=2,
            window_size=8,
        )
    )
    device = select_device(device_name, gpu_ids=(1, 2, 3))
    model = model.to(device)

    batch_source = torch.from_numpy(np.stack([source, source], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    batch_target = torch.from_numpy(np.stack([target, target], axis=0)).to(
        device=device,
        dtype=torch.float32,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = CharbonnierLoss(epsilon=1e-3)

    optimizer.zero_grad()
    output = model(batch_source)
    loss = loss_fn(output, batch_target)
    loss.backward()
    optimizer.step()

    if output.shape != batch_target.shape:
        raise AssertionError(
            f"Output shape {output.shape} does not match target shape {batch_target.shape}"
        )
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss.item()}")

    print("StainSWIN training sanity check passed.")
    print(f"paired sample filename: {filename}")
    print(f"dataset size: {len(dataset)}")
    print(f"device: {device}")
    print(f"batch shape: {tuple(batch_source.shape)}")
    print(f"loss: {float(loss.item()):.6f}")

    if temp_dir is not None:
        temp_dir.cleanup()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a StainSWIN training sanity check."
    )
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_sanity_check(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
