"""
Run example:
.venv/bin/python -m ai.models.stainswin.train_stainswin \
--train-source-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio \
--train-target-dir /mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu \
--checkpoints-dir ai/checkpoints/stainswin \
--experiment-name stainswin_aperio_to_hamamatsu \
--image-size 256 \
--epochs 40 \
--device auto
"""


from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
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
DEFAULT_CHECKPOINTS_DIR = Path("/mnt/Disk1/dpsn_outputs/checkpoints/stainswin")


@dataclass(slots=True)
class StainSWINTrainingConfig:
    train_source_dir: Path = DEFAULT_APERIO_DIR
    train_target_dir: Path = DEFAULT_HAMAMATSU_DIR
    val_source_dir: Path | None = None
    val_target_dir: Path | None = None
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    image_size: int = 256

    input_nc: int = 3
    output_nc: int = 3
    embed_dim: int = 30
    num_heads: int = 3
    num_res_blocks: int = 2
    stbs_per_block: int = 4
    window_size: int = 8
    mlp_ratio: float = 4.0
    conv_kernel_size: int = 3
    reconstruction_channels: int | None = None
    use_image_residual: bool = True

    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    device: str = "auto"
    experiment_name: str = "stainswin"
    recursive: bool = True
    source_prefix: str = "A"
    target_prefix: str = "H"
    gpu_ids: tuple[int, ...] = (1, 2, 3)
    quality_filter: bool = True
    min_tissue_fraction: float = 0.2
    max_black_fraction: float = 0.1
    save_every_epochs: int = 2
    auto_resume: bool = True
    resume_checkpoint: Path | None = None


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))


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
    quality_filter: bool,
    min_tissue_fraction: float,
    max_black_fraction: float,
) -> DataLoader:
    dataset = PairedAlignedImageDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=image_size,
        recursive=recursive,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
        quality_filter=quality_filter,
        min_tissue_fraction=min_tissue_fraction,
        max_black_fraction=max_black_fraction,
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
    scaler: GradScaler | None = None,
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
        amp_enabled = scaler is not None and device.type == "cuda"
        with autocast(device_type=device.type, enabled=amp_enabled):
            output = model(source)
            loss = loss_fn(output, target)

        if amp_enabled:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    val_loss: float | None = None,
    train_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": config.experiment_name,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "model_state_dict": unwrap_parallel(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def resolve_resume_checkpoint(config: StainSWINTrainingConfig) -> Path | None:
    if config.resume_checkpoint is not None:
        if not config.resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"resume checkpoint not found: {config.resume_checkpoint}"
            )
        return config.resume_checkpoint
    if not config.auto_resume:
        return None
    candidates = [
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth",
        config.checkpoints_dir / f"{config.experiment_name}_best.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    unwrap_parallel(model).load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_loss = checkpoint.get("val_loss")
    if best_loss is None:
        best_loss = checkpoint.get("train_loss", float("inf"))
    return start_epoch, float(best_loss)


def _train_with_batch_size(
    config: StainSWINTrainingConfig,
    batch_size: int,
) -> Path:
    device = select_device(config.device, config.gpu_ids)
    if config.save_every_epochs <= 0:
        raise ValueError("save_every_epochs must be > 0.")
    train_loader = create_dataloader(
        source_dir=config.train_source_dir,
        target_dir=config.train_target_dir,
        image_size=config.image_size,
        batch_size=batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        recursive=config.recursive,
        source_prefix=config.source_prefix,
        target_prefix=config.target_prefix,
        quality_filter=config.quality_filter,
        min_tissue_fraction=config.min_tissue_fraction,
        max_black_fraction=config.max_black_fraction,
    )

    val_loader = None
    if config.val_source_dir is not None and config.val_target_dir is not None:
        val_loader = create_dataloader(
            source_dir=config.val_source_dir,
            target_dir=config.val_target_dir,
            image_size=config.image_size,
            batch_size=batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            recursive=config.recursive,
            source_prefix=config.source_prefix,
            target_prefix=config.target_prefix,
            quality_filter=config.quality_filter,
            min_tissue_fraction=config.min_tissue_fraction,
            max_black_fraction=config.max_black_fraction,
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
    loss_fn = CharbonnierLoss(epsilon=1e-3)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    latest_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    )
    best_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_best.pth"
    )
    best_loss = float("inf")

    start_epoch = 0
    resume_checkpoint = resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        start_epoch, best_loss = load_training_checkpoint(
            path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        for _ in range(start_epoch):
            scheduler.step()
        print(
            f"Resumed StainSWIN from {resume_checkpoint} at epoch {start_epoch}; "
            f"best_loss={best_loss:.6f}"
        )
    else:
        print("Starting StainSWIN from scratch.")

    if start_epoch >= config.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch}; "
            f"requested epochs={config.epochs}."
        )
        return best_checkpoint_path if best_checkpoint_path.is_file() else latest_checkpoint_path

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
            scaler=scaler,
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
            val_loss = None
            print(f"epoch {epoch}/{config.epochs} - train_l1={train_loss:.6f}")

        scheduler.step()
        score_loss = val_loss if val_loss is not None else train_loss

        if epoch % config.save_every_epochs == 0 or epoch == config.epochs:
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
                val_loss=val_loss,
                train_loss=train_loss,
            )
        save_checkpoint(
            model=model,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            path=latest_checkpoint_path,
            val_loss=val_loss,
            train_loss=train_loss,
        )
        if score_loss < best_loss:
            best_loss = score_loss
            save_checkpoint(
                model=model,
                config=config,
                epoch=epoch,
                optimizer=optimizer,
                path=best_checkpoint_path,
                val_loss=val_loss,
                train_loss=train_loss,
            )
            print(
                f"Updated best checkpoint: {best_checkpoint_path} "
                f"loss={best_loss:.6f}"
            )

    return best_checkpoint_path if best_checkpoint_path.is_file() else latest_checkpoint_path


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _candidate_batch_sizes(requested_batch_size: int) -> list[int]:
    candidates = [32, 16, 8, 4, 3, 2, 1]
    return [batch_size for batch_size in candidates if batch_size <= requested_batch_size]


def train(config: StainSWINTrainingConfig) -> Path:
    candidate_batch_sizes = _candidate_batch_sizes(config.batch_size)
    if not candidate_batch_sizes:
        raise ValueError(
            f"batch_size must be at least 4 for fallback logic, got {config.batch_size}"
        )

    last_error: RuntimeError | None = None
    for batch_size in candidate_batch_sizes:
        try:
            if batch_size != config.batch_size:
                print(
                    f"Retrying StainSWIN training with reduced batch_size={batch_size} "
                    f"(requested {config.batch_size})."
                )
            return _train_with_batch_size(config=config, batch_size=batch_size)
        except RuntimeError as error:
            if not torch.cuda.is_available() or not _is_cuda_oom(error):
                raise
            last_error = error
            print(
                f"CUDA OOM encountered with batch_size={batch_size}. "
                "Trying a smaller batch size..."
            )
            torch.cuda.empty_cache()

    assert last_error is not None
    raise RuntimeError(
        "StainSWIN training ran out of GPU memory even after trying batch sizes "
        "32, 16, 8, 4, 3, 2, and 1."
    ) from last_error


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
        default=DEFAULT_CHECKPOINTS_DIR,
    )
    parser.add_argument("--experiment-name", type=str, default="stainswin")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-nc", type=int, default=3)
    parser.add_argument("--output-nc", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--stbs-per-block", type=int, default=6)
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
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.2)
    parser.add_argument("--max-black-fraction", type=float, default=0.1)
    parser.add_argument("--save-every-epochs", type=int, default=2)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
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
        quality_filter=args.quality_filter,
        min_tissue_fraction=args.min_tissue_fraction,
        max_black_fraction=args.max_black_fraction,
        save_every_epochs=args.save_every_epochs,
        auto_resume=args.auto_resume,
        resume_checkpoint=args.resume_checkpoint,
    )
    checkpoint_path = train(config)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
