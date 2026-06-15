"""
Example:
.venv/bin/python -m ai.models.multistain.multistain_train \
--dataset-dir /mnt/Disk1/dpsn_datasets/multiscanner_dataset \
--checkpoints-dir /mnt/Disk1/dpsn_outputs/checkpoints/multistain \
--canonical-domain nz210 \
--source-domains cs2,nz20,p1000,gt450 \
--image-size 256 \
--target-mpp 0.22 \
--batch-size 4 \
--epochs 80 \
--checkpoint-interval 2 \
--gpu-ids 1,2,3 \
--device auto
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ai.models.multistain.config import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_PATCH_CACHE_DIR,
    MultiStainCycleGANConfig,
)
from ai.models.multistain.multistain_model import MultiStainCycleGANModel


def create_dataloaders(
    config: MultiStainCycleGANConfig,
) -> tuple[DataLoader, DataLoader]:
    from ai.models.multistain.dataset import create_datasets

    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    print(
        "MultiStain-CycleGAN dataset ready: "
        f"train_patches={len(train_dataset)} val_patches={len(val_dataset)} "
        f"source_domains={train_dataset.source_scanner_ids} "
        f"canonical_domain={config.canonical_domain}",
        flush=True,
    )
    return train_loader, val_loader


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> LambdaLR:
    """Keep LR stable for half of training, then linearly decay to zero."""

    constant_epochs = max(1, epochs // 2)
    decay_epochs = max(1, epochs - constant_epochs)

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if epoch_number <= constant_epochs:
            return 1.0
        return max(0.0, 1.0 - (epoch_number - constant_epochs) / decay_epochs)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: MultiStainCycleGANModel,
    dataloader: DataLoader,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    running: defaultdict[str, float] = defaultdict(float)
    batches = 0
    progress = tqdm(
        dataloader,
        desc=f"MultiStain train {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
    )

    for batch in progress:
        losses = model.train_step(batch)
        batches += 1
        for key, value in losses.items():
            running[key] += float(value)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                g=f"{losses['g']:.4f}",
                d=f"{losses['d']:.4f}",
                cyc=f"{losses['cycle_source'] + losses['cycle_target']:.4f}",
            )

    return average_losses(running, batches)


def validate(
    model: MultiStainCycleGANModel,
    dataloader: DataLoader,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    running: defaultdict[str, float] = defaultdict(float)
    batches = 0
    progress = tqdm(
        dataloader,
        desc=f"MultiStain val {epoch}/{total_epochs}",
        unit="batch",
        leave=False,
    )

    for batch in progress:
        losses = model.validation_step(batch)
        batches += 1
        for key, value in losses.items():
            running[key] += float(value)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                g=f"{losses['g']:.4f}",
                aligned_l1=f"{losses.get('aligned_l1', 0.0):.4f}",
            )

    return average_losses(running, batches)


def save_checkpoint(
    model: MultiStainCycleGANModel,
    path: Path,
    epoch: int,
    metrics: dict[str, float],
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    best_val_g: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint_state(epoch=epoch, metrics=metrics)
    checkpoint["scheduler_g"] = scheduler_g.state_dict()
    checkpoint["scheduler_d"] = scheduler_d.state_dict()
    checkpoint["best_val_g"] = best_val_g
    torch.save(checkpoint, path)


def train(config: MultiStainCycleGANConfig) -> Path:
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = create_dataloaders(config)
    model = MultiStainCycleGANModel(config)

    if model.device.type == "cuda":
        print(
            f"Using CUDA DataParallel on gpu_ids={config.gpu_ids} "
            f"(primary device: cuda:{config.gpu_ids[0]})",
            flush=True,
        )
    else:
        print(f"Using device: {model.device}", flush=True)

    scheduler_g = build_lr_scheduler(model.optimizer_g, config.epochs)
    scheduler_d = build_lr_scheduler(model.optimizer_d, config.epochs)

    best_val_g = float("inf")
    start_epoch = 0
    if config.resume_checkpoint is not None:
        if not config.resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {config.resume_checkpoint}"
            )
        checkpoint = model.load_checkpoint(config.resume_checkpoint, load_optimizers=True)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_g = float(checkpoint.get("best_val_g", best_val_g))

        scheduler_g_state = checkpoint.get("scheduler_g")
        scheduler_d_state = checkpoint.get("scheduler_d")
        if isinstance(scheduler_g_state, dict) and isinstance(scheduler_d_state, dict):
            scheduler_g.load_state_dict(scheduler_g_state)
            scheduler_d.load_state_dict(scheduler_d_state)
        else:
            for optimizer in (model.optimizer_g, model.optimizer_d):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr
            scheduler_g = build_lr_scheduler(model.optimizer_g, config.epochs)
            scheduler_d = build_lr_scheduler(model.optimizer_d, config.epochs)
            for _ in range(start_epoch):
                scheduler_g.step()
                scheduler_d.step()

        if best_val_g == float("inf") and config.best_checkpoint_path.is_file():
            best_checkpoint = torch.load(
                config.best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            best_metrics = best_checkpoint.get("metrics", {})
            best_val_g = float(best_metrics.get("val_g", best_val_g))

        print(
            f"Resuming from {config.resume_checkpoint} at epoch {start_epoch + 1}/"
            f"{config.epochs}",
            flush=True,
        )

    if start_epoch >= config.epochs:
        raise ValueError(
            f"Resume checkpoint is at epoch {start_epoch}, but configured total "
            f"epochs is {config.epochs}."
        )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        val_losses = validate(
            model=model,
            dataloader=val_loader,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        scheduler_g.step()
        scheduler_d.step()

        metrics = {f"train_{k}": v for k, v in train_losses.items()}
        metrics.update({f"val_{k}": v for k, v in val_losses.items()})
        print(format_epoch_summary(epoch, config.epochs, metrics), flush=True)

        val_g = val_losses.get("g", float("inf"))
        if val_g < best_val_g:
            best_val_g = val_g
            save_checkpoint(
                model=model,
                path=config.best_checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                best_val_g=best_val_g,
            )
            print(
                f"Updated best checkpoint: {config.best_checkpoint_path} "
                f"val_g={best_val_g:.6f}",
                flush=True,
            )

        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                path=config.latest_checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                best_val_g=best_val_g,
            )
            checkpoint_path = config.epoch_checkpoint_path(epoch)
            save_checkpoint(
                model=model,
                path=checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                best_val_g=best_val_g,
            )
            print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    print(
        f"Finished MultiStain-CycleGAN training. Best checkpoint: {config.best_checkpoint_path}",
        flush=True,
    )
    return config.best_checkpoint_path


def average_losses(
    running: defaultdict[str, float],
    batches: int,
) -> dict[str, float]:
    divisor = max(int(batches), 1)
    return {key: value / divisor for key, value in running.items()}


def format_epoch_summary(
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
) -> str:
    keys = [
        "train_g",
        "train_d",
        "train_cycle_source",
        "train_cycle_target",
        "train_identity_target",
        "val_g",
        "val_aligned_l1",
    ]
    parts = [
        f"{key}={metrics[key]:.6f}"
        for key in keys
        if key in metrics
    ]
    return f"epoch {epoch}/{total_epochs} - " + " ".join(parts)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train many-source-to-one MultiStain-CycleGAN."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--patch-cache-dir", type=Path, default=DEFAULT_PATCH_CACHE_DIR)
    parser.add_argument("--experiment-name", type=str, default="multistain_many_to_nz210")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--canonical-domain", type=str, default="nz210")
    parser.add_argument(
        "--source-domains",
        type=str,
        default="cs2,nz20,p1000,gt450",
        help="Comma-separated scanner ids used as source domains.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-mpp", type=float, default=None)
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--patches-per-source-slide", type=int, default=128)
    parser.add_argument("--patches-per-target-slide", type=int, default=128)
    parser.add_argument("--mask-longest-side", type=int, default=1024)
    parser.add_argument("--strict-mpp-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-patch-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sample-count", type=int, default=36)
    parser.add_argument("--val-sample-count", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--generator-blocks", type=int, default=9)
    parser.add_argument("--discriminator-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-content", type=float, default=0.0)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="1,2,3",
        help="Comma-separated CUDA device ids for DataParallel.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> MultiStainCycleGANConfig:
    source_domains = tuple(parse_csv(args.source_domains))
    gpu_ids = tuple(int(token) for token in parse_csv(args.gpu_ids))
    return MultiStainCycleGANConfig(
        dataset_dir=args.dataset_dir,
        checkpoints_dir=args.checkpoints_dir,
        patch_cache_dir=args.patch_cache_dir,
        experiment_name=args.experiment_name,
        resume_checkpoint=args.resume_checkpoint,
        canonical_domain=args.canonical_domain,
        source_domains=source_domains,
        image_size=args.image_size,
        target_mpp=args.target_mpp,
        read_level=args.read_level,
        patches_per_source_slide=args.patches_per_source_slide,
        patches_per_target_slide=args.patches_per_target_slide,
        mask_longest_side=args.mask_longest_side,
        strict_mpp_check=args.strict_mpp_check,
        recursive=args.recursive,
        use_patch_cache=args.use_patch_cache,
        train_sample_count=args.train_sample_count,
        val_sample_count=args.val_sample_count,
        split_seed=args.split_seed,
        ngf=args.ngf,
        ndf=args.ndf,
        generator_blocks=args.generator_blocks,
        discriminator_layers=args.discriminator_layers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        lr=args.lr,
        beta1=args.beta1,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_content=args.lambda_content,
        pool_size=args.pool_size,
        device=args.device,
        gpu_ids=gpu_ids,
        verbose=args.verbose,
    )


def parse_csv(value: str) -> list[str]:
    tokens = [token.strip() for token in value.split(",")]
    return [token for token in tokens if token]


def main() -> None:
    args = build_argparser().parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
