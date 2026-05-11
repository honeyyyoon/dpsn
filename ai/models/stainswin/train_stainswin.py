"""
Run example:
./.venv/bin/python -m ai.models.stainswin.train_stainswin \
  --train-source-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio \
  --train-target-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu \
  --checkpoints-dir ai/checkpoints/stainswin \
  --experiment-name stainswin_aperio_to_hamamatsu \
  --image-size 256 \
  --batch-size 8 \
  --epochs 40 \
  --device auto
"""


from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ai.models.stainnet.paired_aligned_dataset import PairedAlignedImageDataset
from ai.models.stainswin.stainswin_model import StainSWIN


DEFAULT_APERIO_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio"
)
DEFAULT_HAMAMATSU_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu"
)


@dataclass(slots=True)
class StainSWINTrainingConfig:
    train_source_dir: Path = DEFAULT_APERIO_DIR
    train_target_dir: Path = DEFAULT_HAMAMATSU_DIR
    val_source_dir: Path | None = None
    val_target_dir: Path | None = None
    checkpoints_dir: Path = Path("checkpoints/stainswin")
    image_size: int = 256

    input_nc: int = 3
    output_nc: int = 3
    embed_dim: int = 96
    num_heads: int = 6
    num_res_blocks: int = 6
    stbs_per_block: int = 2
    window_size: int = 8
    mlp_ratio: float = 4.0
    conv_kernel_size: int = 3
    reconstruction_channels: int | None = None
    use_image_residual: bool = True

    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    device: str = "auto"
    experiment_name: str = "stainswin"
    recursive: bool = True
    source_prefix: str = "A"
    target_prefix: str = "H"
    gpu_ids: tuple[int, ...] = (1, 2, 3)


def create_model(config: StainSWINTrainingConfig) -> StainSWIN:
    return StainSWIN(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_res_blocks=config.num_res_blocks,
        stbs_per_block=config.stbs_per_block,
        window_size=config.window_size,
        mlp_ratio=config.mlp_ratio,
        conv_kernel_size=config.conv_kernel_size,
        reconstruction_channels=config.reconstruction_channels,
        use_image_residual=config.use_image_residual,
    )


def create_dataloader(
    source_dir: Path,
    target_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    recursive: bool,
    source_prefix: str,
    target_prefix: str,
) -> DataLoader:
    dataset = PairedAlignedImageDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=image_size,
        recursive=recursive,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(
        dataloader,
        desc=f"StainSWIN train {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
    )

    for source, target, _ in progress:
        source = source.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(source)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                l1=f"{loss.item():.4f}",
                avg=f"{total_loss / total_batches:.4f}",
            )

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc=f"StainSWIN val {epoch}/{total_epochs}",
            unit="batch",
            leave=False,
        )
        for source, target, _ in progress:
            source = source.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            output = model(source)
            loss = loss_fn(output, target)
            total_loss += float(loss.item())
            total_batches += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    l1=f"{loss.item():.4f}",
                    avg=f"{total_loss / total_batches:.4f}",
                )

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: nn.Module,
    config: StainSWINTrainingConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": config.experiment_name,
            "model_state_dict": unwrap_parallel(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def train(config: StainSWINTrainingConfig) -> Path:
    device = select_device(config.device, config.gpu_ids)
    train_loader = create_dataloader(
        source_dir=config.train_source_dir,
        target_dir=config.train_target_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        recursive=config.recursive,
        source_prefix=config.source_prefix,
        target_prefix=config.target_prefix,
    )

    val_loader = None
    if config.val_source_dir is not None and config.val_target_dir is not None:
        val_loader = create_dataloader(
            source_dir=config.val_source_dir,
            target_dir=config.val_target_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            recursive=config.recursive,
            source_prefix=config.source_prefix,
            target_prefix=config.target_prefix,
        )

    model = create_model(config).to(device)
    model = maybe_wrap_dataparallel(
        model=model,
        device=device,
        gpu_ids=config.gpu_ids,
    )

    if device.type == "cuda":
        print(
            f"Using CUDA with DataParallel on gpu_ids={config.gpu_ids} "
            f"(primary device: cuda:{config.gpu_ids[0]})"
        )
    else:
        print(f"Using device: {device}")

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.L1Loss()

    latest_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                total_epochs=config.epochs,
            )
            print(
                f"epoch {epoch}/{config.epochs} - "
                f"train_l1={train_loss:.6f} val_l1={val_loss:.6f}"
            )
        else:
            print(f"epoch {epoch}/{config.epochs} - train_l1={train_loss:.6f}")

        scheduler.step()

        epoch_checkpoint_path = (
            config.checkpoints_dir
            / f"{config.experiment_name}_epoch_{epoch:03d}.pth"
        )
        save_checkpoint(
            model=model,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            path=epoch_checkpoint_path,
        )
        save_checkpoint(
            model=model,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            path=latest_checkpoint_path,
        )

    return latest_checkpoint_path


def select_device(device: str, gpu_ids: tuple[int, ...]) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        if not gpu_ids:
            raise ValueError("gpu_ids must not be empty when using auto CUDA selection.")
        return torch.device(f"cuda:{gpu_ids[0]}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_wrap_dataparallel(
    model: nn.Module,
    device: torch.device,
    gpu_ids: tuple[int, ...],
) -> nn.Module:
    if device.type != "cuda":
        return model

    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty when using CUDA training.")

    available_gpu_count = torch.cuda.device_count()
    max_gpu_id = max(gpu_ids)
    if available_gpu_count <= max_gpu_id:
        raise ValueError(
            f"Requested gpu_ids {gpu_ids}, but only {available_gpu_count} CUDA device(s) are available."
        )

    return nn.DataParallel(model, device_ids=list(gpu_ids), output_device=gpu_ids[0])


def unwrap_parallel(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train StainSWIN on paired aligned images."
    )
    parser.add_argument(
        "--train-source-dir",
        dest="train_source_dir",
        type=Path,
        default=DEFAULT_APERIO_DIR,
    )
    parser.add_argument(
        "--train-target-dir",
        dest="train_target_dir",
        type=Path,
        default=DEFAULT_HAMAMATSU_DIR,
    )
    parser.add_argument("--val-source-dir", type=Path, default=None)
    parser.add_argument("--val-target-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("ai/checkpoints/stainswin"),
    )
    parser.add_argument("--experiment-name", type=str, default="stainswin")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-nc", type=int, default=3)
    parser.add_argument("--output-nc", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--stbs-per-block", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--reconstruction-channels", type=int, default=None)
    parser.add_argument(
        "--use-image-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--source-prefix", type=str, default="A")
    parser.add_argument("--target-prefix", type=str, default="H")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[1, 2, 3])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = StainSWINTrainingConfig(
        train_source_dir=args.train_source_dir,
        train_target_dir=args.train_target_dir,
        val_source_dir=args.val_source_dir,
        val_target_dir=args.val_target_dir,
        checkpoints_dir=args.checkpoints_dir,
        image_size=args.image_size,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_res_blocks=args.num_res_blocks,
        stbs_per_block=args.stbs_per_block,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        conv_kernel_size=args.conv_kernel_size,
        reconstruction_channels=args.reconstruction_channels,
        use_image_residual=args.use_image_residual,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        experiment_name=args.experiment_name,
        recursive=args.recursive,
        source_prefix=args.source_prefix,
        target_prefix=args.target_prefix,
        gpu_ids=tuple(args.gpu_ids),
    )
    checkpoint_path = train(config)
    print(f"Saved latest checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
