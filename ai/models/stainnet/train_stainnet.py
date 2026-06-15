from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ai.models.stainnet.paired_aligned_dataset import PairedAlignedImageDataset
from ai.models.stainnet.stainnet_model import StainNet

DEFAULT_APERIO_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio"
)
DEFAULT_HAMAMATSU_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu"
)
DEFAULT_CHECKPOINTS_DIR = Path("/mnt/Disk1/dpsn_outputs/checkpoints/stainnet")


@dataclass(slots=True)
class StainNetTrainingConfig:
    train_source_dir: Path = DEFAULT_APERIO_DIR
    train_target_dir: Path = DEFAULT_HAMAMATSU_DIR
    val_source_dir: Path | None = None
    val_target_dir: Path | None = None
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    image_size: int = 256
    input_nc: int = 3
    output_nc: int = 3
    channels: int = 32
    n_layer: int = 3
    kernel_size: int = 1
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 0.01
    epochs: int = 50
    device: str = "auto"
    experiment_name: str = "stainnet"
    recursive: bool = True # whether the dataset loader searches only the top-level folder or also all nested subfolders for images
    source_prefix: str = "A"
    target_prefix: str = "H"
    gpu_ids: tuple[int, ...] = (1, 2, 3)
    quality_filter: bool = True
    min_tissue_fraction: float = 0.2
    max_black_fraction: float = 0.1
    save_every_epochs: int = 2
    auto_resume: bool = True
    resume_checkpoint: Path | None = None


def create_model(config: StainNetTrainingConfig) -> StainNet:
    return StainNet(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        n_layer=config.n_layer,
        n_channel=config.channels,
        kernel_size=config.kernel_size,
    )

# this function does two things: 1) create the dataset object 2) wrap that dataset in a PyTorch DataLoader
# it turns source, target folder into smth you can iterate in training
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
        shuffle=shuffle, # Whether to shuffle the dataset order
        num_workers=num_workers, # How many worker processes PyTorch should use for loading data
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
        desc=f"StainNet train {epoch}/{total_epochs}",
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
            desc=f"StainNet val {epoch}/{total_epochs}",
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
    config: StainNetTrainingConfig,
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


def resolve_resume_checkpoint(config: StainNetTrainingConfig) -> Path | None:
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


def train(config: StainNetTrainingConfig) -> Path:
    device = select_device(config.device)
    if config.save_every_epochs <= 0:
        raise ValueError("save_every_epochs must be > 0.")
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
            batch_size=config.batch_size,
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
    optimizer = SGD(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs) # A scheduler gradually adjusts the learning rate 
    loss_fn = nn.L1Loss()

    latest_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    )
    best_checkpoint_path = config.checkpoints_dir / f"{config.experiment_name}_best.pth"
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
            f"Resumed StainNet from {resume_checkpoint} at epoch {start_epoch}; "
            f"best_loss={best_loss:.6f}"
        )
    else:
        print("Starting StainNet from scratch.")

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
                f"epoch {epoch}/{config.epochs} - train_l1={train_loss:.6f} val_l1={val_loss:.6f}"
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


def select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:1")
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
    parser = argparse.ArgumentParser(description="Train StainNet on paired aligned images.")
    parser.add_argument("--train-source-dir", "--source-root-train", dest="train_source_dir", type=Path, default=DEFAULT_APERIO_DIR)
    parser.add_argument("--train-target-dir", "--gt-root-train", dest="train_target_dir", type=Path, default=DEFAULT_HAMAMATSU_DIR)
    parser.add_argument("--val-source-dir", "--source-root-test", dest="val_source_dir", type=Path, default=None)
    parser.add_argument("--val-target-dir", "--gt-root-test", dest="val_target_dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--image-size", "--fineSize", dest="image_size", type=int, default=256)
    parser.add_argument("--batch-size", "--batchSize", dest="batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--nThreads", dest="num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--experiment-name", type=str, default="stainnet")
    parser.add_argument("--input-nc", type=int, default=3)
    parser.add_argument("--output-nc", type=int, default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=1)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
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
    config = StainNetTrainingConfig(
        train_source_dir=args.train_source_dir,
        train_target_dir=args.train_target_dir,
        val_source_dir=args.val_source_dir,
        val_target_dir=args.val_target_dir,
        checkpoints_dir=args.checkpoints_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        experiment_name=args.experiment_name,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        channels=args.channels,
        n_layer=args.n_layer,
        kernel_size=args.kernel_size,
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
