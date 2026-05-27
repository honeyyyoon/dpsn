"""
Running example:
./.venv/bin/python -m ai.models.staingan.train_staingan \
--dataset-dir /mnt/Disk1/dpsn_datasets/multiscanner_dataset \
--canonical-domain nz210 \
--checkpoints-dir ai/checkpoints/staingan \
--experiment-name staingan_many_to_nz210 \
--image-size 256 \
--target-mpp 0.25 \
--patches-per-source-slide 128 \
--train-sample-count 36 \
--val-sample-count 8
--mask-longest-side 512 \
--verbose
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from ai.models.staingan.multidomain_wsi_dataset import (
    MultiDomainWSIPatchDataset,
    split_sample_ids,
)
from ai.models.staingan.staingan_model import (
    GANLoss,
    ImagePool,
    NLayerDiscriminator,
    ResnetGenerator,
)


DEFAULT_MULTISCANNER_DIR = Path("/mnt/Disk1/dpsn_datasets/multiscanner_dataset")
DEFAULT_CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "staingan"


@dataclass(slots=True)
class StainGANTrainingConfig:
    dataset_dir: Path = DEFAULT_MULTISCANNER_DIR
    canonical_domain: str = "nz210"
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    experiment_name: str = "staingan_many_to_nz210"
    image_size: int = 256
    target_mpp: float = 0.25
    read_level: int = 0
    patches_per_source_slide: int = 128
    mask_longest_side: int = 1024
    patch_cache_dir: Path = Path("/mnt/Disk1/dpsn_patch_cache/staingan_patch_cache")
    use_patch_cache: bool = True
    train_sample_count: int = 36
    val_sample_count: int = 8
    split_seed: int = 0
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    generator_blocks: int = 9
    discriminator_layers: int = 3
    batch_size: int = 4
    num_workers: int = 0
    epochs_constant_lr: int = 20
    epochs_decay_lr: int = 20
    lr: float = 0.0002
    beta1: float = 0.5
    lambda_identity: float = 5.0
    lambda_content: float = 10.0
    content_loss: Literal["grayscale_l1", "grayscale_ssim", "none"] = "grayscale_l1"
    pool_size: int = 50
    device: str = "auto"
    recursive: bool = False
    strict_mpp_check: bool = True
    verbose: bool = False
    gpu_ids: tuple[int, ...] = (1, 2, 3)

    @property
    def total_epochs(self) -> int:
        return self.epochs_constant_lr + self.epochs_decay_lr


def select_device(device: str, gpu_ids: tuple[int, ...] = (1, 2, 3)) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        if not gpu_ids:
            raise ValueError("gpu_ids must not be empty when using auto CUDA selection.")
        return torch.device(f"cuda:{gpu_ids[0]}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_datasets(
    config: StainGANTrainingConfig,
) -> tuple[MultiDomainWSIPatchDataset, MultiDomainWSIPatchDataset]:
    print(
        "Preparing StainGAN datasets: "
        f"dataset_dir={config.dataset_dir} canonical_domain={config.canonical_domain} "
        f"train_sample_count={config.train_sample_count} val_sample_count={config.val_sample_count} "
        f"patches_per_source_slide={config.patches_per_source_slide}",
        flush=True,
    )
    train_ids, val_ids = split_sample_ids(
        config.dataset_dir,
        train_count=config.train_sample_count,
        val_count=config.val_sample_count,
        seed=config.split_seed,
        recursive=config.recursive,
    )
    if not train_ids:
        raise ValueError("Training split is empty.")
    if not val_ids:
        raise ValueError("Validation split is empty.")
    print(
        f"Split sample ids: train={train_ids} val={val_ids}",
        flush=True,
    )

    print("Initializing training patch dataset...", flush=True)
    train_dataset = MultiDomainWSIPatchDataset(
        dataset_dir=config.dataset_dir,
        canonical_domain=config.canonical_domain,
        sample_ids=train_ids,
        image_size=config.image_size,
        target_mpp=config.target_mpp,
        read_level=config.read_level,
        patches_per_source_slide=config.patches_per_source_slide,
        mask_longest_side=config.mask_longest_side,
        strict_mpp_check=config.strict_mpp_check,
        recursive=config.recursive,
        seed=config.split_seed,
        sampler_result_dir=Path("result") / "staingan_patch_sampler" / "train",
        cache_dir=config.patch_cache_dir / "train",
        use_patch_cache=config.use_patch_cache,
        verbose=config.verbose,
    )
    print(
        f"Training dataset ready: {len(train_dataset)} patch item(s).",
        flush=True,
    )
    print("Initializing validation patch dataset...", flush=True)
    val_dataset = MultiDomainWSIPatchDataset(
        dataset_dir=config.dataset_dir,
        canonical_domain=config.canonical_domain,
        sample_ids=val_ids,
        image_size=config.image_size,
        target_mpp=config.target_mpp,
        read_level=config.read_level,
        patches_per_source_slide=max(1, config.patches_per_source_slide // 4),
        mask_longest_side=config.mask_longest_side,
        strict_mpp_check=config.strict_mpp_check,
        recursive=config.recursive,
        seed=config.split_seed + 10_000,
        sampler_result_dir=Path("result") / "staingan_patch_sampler" / "val",
        cache_dir=config.patch_cache_dir / "val",
        use_patch_cache=config.use_patch_cache,
        verbose=config.verbose,
    )
    print(
        f"Validation dataset ready: {len(val_dataset)} patch item(s).",
        flush=True,
    )
    return train_dataset, val_dataset


def create_dataloaders(
    config: StainGANTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    print(
        "StainGAN splits: "
        f"train_samples={len(train_dataset.sample_ids)} val_samples={len(val_dataset.sample_ids)} "
        f"source_domains={train_dataset.source_domains} canonical_domain={config.canonical_domain}"
    )
    return train_loader, val_loader


def create_models(
    config: StainGANTrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    generator = ResnetGenerator(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf,
        n_blocks=config.generator_blocks,
    ).to(device)
    discriminator = NLayerDiscriminator(
        input_nc=config.output_nc,
        ndf=config.ndf,
        n_layers=config.discriminator_layers,
    ).to(device)
    generator = maybe_wrap_dataparallel(generator, device, config.gpu_ids)
    discriminator = maybe_wrap_dataparallel(discriminator, device, config.gpu_ids)
    return generator, discriminator


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: StainGANTrainingConfig,
) -> LambdaLR:
    constant_epochs = config.epochs_constant_lr
    total_epochs = config.total_epochs

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if epoch_number <= constant_epochs:
            return 1.0
        decay_progress = epoch_number - constant_epochs
        decay_epochs = max(total_epochs - constant_epochs, 1)
        return max(0.0, 1.0 - decay_progress / decay_epochs)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    config: StainGANTrainingConfig,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    path: Path,
    val_generator_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": config.experiment_name,
            "canonical_domain": config.canonical_domain,
            "val_generator_loss": val_generator_loss,
            "config": asdict(config),
            "g_source_to_canonical_state_dict": unwrap_parallel(generator).state_dict(),
            "d_canonical_state_dict": unwrap_parallel(discriminator).state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
        },
        path,
    )


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


def grayscale(tensor: torch.Tensor) -> torch.Tensor:
    weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (tensor * weights).sum(dim=1, keepdim=True)


def grayscale_ssim_loss(source: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    x = grayscale((source + 1.0) * 0.5)
    y = grayscale((generated + 1.0) * 0.5)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = x.mean(dim=(-2, -1), keepdim=True)
    mu_y = y.mean(dim=(-2, -1), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return 1.0 - ssim.mean()


def content_loss(source: torch.Tensor, generated: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return source.new_zeros(())
    if mode == "grayscale_l1":
        return nn.functional.l1_loss(grayscale(source), grayscale(generated))
    if mode == "grayscale_ssim":
        return grayscale_ssim_loss(source, generated)
    raise ValueError(f"Unsupported content_loss: {mode}")


def generator_losses(
    generator: nn.Module,
    discriminator: nn.Module,
    source: torch.Tensor,
    canonical: torch.Tensor,
    identity: torch.Tensor,
    gan_loss: GANLoss,
    l1_loss: nn.Module,
    config: StainGANTrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    del canonical
    fake_canonical = generator(source)
    pred_fake = discriminator(fake_canonical)
    loss_adv = gan_loss(pred_fake, True)
    same_canonical = generator(identity)
    loss_identity = l1_loss(same_canonical, identity) * config.lambda_identity
    loss_content = content_loss(source, fake_canonical, config.content_loss) * config.lambda_content
    loss_g = loss_adv + loss_identity + loss_content
    return loss_g, {
        "g": float(loss_g.item()),
        "adv": float(loss_adv.item()),
        "identity": float(loss_identity.item()),
        "content": float(loss_content.item()),
    }


def train_one_epoch(
    config: StainGANTrainingConfig,
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    gan_loss: GANLoss,
    l1_loss: nn.Module,
    fake_pool: ImagePool,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    running = {"g": 0.0, "d": 0.0, "adv": 0.0, "identity": 0.0, "content": 0.0, "batches": 0}
    progress = tqdm(
        dataloader,
        desc=f"StainGAN train {epoch}/{config.total_epochs}",
        unit="batch",
        leave=False,
    )

    for batch in progress:
        source = batch["source"].to(device=device, dtype=torch.float32)
        canonical = batch["canonical"].to(device=device, dtype=torch.float32)
        identity = batch["identity"].to(device=device, dtype=torch.float32)

        optimizer_g.zero_grad()
        loss_g, parts = generator_losses(
            generator=generator,
            discriminator=discriminator,
            source=source,
            canonical=canonical,
            identity=identity,
            gan_loss=gan_loss,
            l1_loss=l1_loss,
            config=config,
        )
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        with torch.no_grad():
            fake_canonical = generator(source)
        pred_real = discriminator(canonical)
        loss_d_real = gan_loss(pred_real, True)
        pred_fake = discriminator(fake_pool.query(fake_canonical))
        loss_d_fake = gan_loss(pred_fake, False)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        loss_d.backward()
        optimizer_d.step()

        running["g"] += parts["g"]
        running["adv"] += parts["adv"]
        running["identity"] += parts["identity"]
        running["content"] += parts["content"]
        running["d"] += float(loss_d.item())
        running["batches"] += 1

        source_domains = sorted(set(batch["source_domain"]))
        target_domains = sorted(set(batch["target_domain"]))
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                g=f"{parts['g']:.4f}",
                d=f"{loss_d.item():.4f}",
                src=",".join(source_domains),
                tgt=",".join(target_domains),
            )

    return average_losses(running)


@torch.inference_mode()
def validate(
    config: StainGANTrainingConfig,
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    gan_loss: GANLoss,
    l1_loss: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    running = {"g": 0.0, "adv": 0.0, "identity": 0.0, "content": 0.0, "batches": 0}
    progress = tqdm(
        dataloader,
        desc=f"StainGAN val {epoch}/{config.total_epochs}",
        unit="batch",
        leave=False,
    )
    for batch in progress:
        source = batch["source"].to(device=device, dtype=torch.float32)
        canonical = batch["canonical"].to(device=device, dtype=torch.float32)
        identity = batch["identity"].to(device=device, dtype=torch.float32)
        _, parts = generator_losses(
            generator=generator,
            discriminator=discriminator,
            source=source,
            canonical=canonical,
            identity=identity,
            gan_loss=gan_loss,
            l1_loss=l1_loss,
            config=config,
        )
        for key in ("g", "adv", "identity", "content"):
            running[key] += parts[key]
        running["batches"] += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(g=f"{parts['g']:.4f}")
    return average_losses(running)


def average_losses(running: dict[str, float]) -> dict[str, float]:
    batches = max(int(running.get("batches", 0)), 1)
    return {
        key: value / batches
        for key, value in running.items()
        if key != "batches"
    }


def train(config: StainGANTrainingConfig) -> Path:
    device = select_device(config.device, config.gpu_ids)
    train_loader, val_loader = create_dataloaders(config)
    generator, discriminator = create_models(config, device)

    if device.type == "cuda":
        print(
            f"Using CUDA with DataParallel on gpu_ids={config.gpu_ids} "
            f"(primary device: cuda:{config.gpu_ids[0]})"
        )
    else:
        print(f"Using device: {device}")

    optimizer_g = Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler_g = build_lr_scheduler(optimizer_g, config)
    scheduler_d = build_lr_scheduler(optimizer_d, config)

    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()
    fake_pool = ImagePool(config.pool_size)

    latest_checkpoint_path = config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    best_checkpoint_path = config.checkpoints_dir / f"{config.experiment_name}_best.pth"
    best_val_g = float("inf")

    for epoch in range(1, config.total_epochs + 1):
        train_losses = train_one_epoch(
            config=config,
            dataloader=train_loader,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            gan_loss=gan_loss,
            l1_loss=l1_loss,
            fake_pool=fake_pool,
            device=device,
            epoch=epoch,
        )
        val_losses = validate(
            config=config,
            dataloader=val_loader,
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            l1_loss=l1_loss,
            device=device,
            epoch=epoch,
        )
        scheduler_g.step()
        scheduler_d.step()

        print(
            f"epoch {epoch}/{config.total_epochs} "
            f"train_g={train_losses['g']:.6f} train_d={train_losses['d']:.6f} "
            f"train_adv={train_losses['adv']:.6f} "
            f"train_identity={train_losses['identity']:.6f} "
            f"train_content={train_losses['content']:.6f} "
            f"val_g={val_losses['g']:.6f} val_adv={val_losses['adv']:.6f} "
            f"val_identity={val_losses['identity']:.6f} "
            f"val_content={val_losses['content']:.6f}"
        )

        epoch_checkpoint_path = (
            config.checkpoints_dir / f"{config.experiment_name}_epoch_{epoch:04d}.pth"
        )
        save_checkpoint(
            config=config,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            path=epoch_checkpoint_path,
            val_generator_loss=val_losses["g"],
        )
        save_checkpoint(
            config=config,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            path=latest_checkpoint_path,
            val_generator_loss=val_losses["g"],
        )
        if val_losses["g"] < best_val_g:
            best_val_g = val_losses["g"]
            save_checkpoint(
                config=config,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                path=best_checkpoint_path,
                val_generator_loss=val_losses["g"],
            )
            print(f"Updated best checkpoint: {best_checkpoint_path} val_g={best_val_g:.6f}")

    return best_checkpoint_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train many-source-to-one StainGAN for canonical stain normalization."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MULTISCANNER_DIR)
    parser.add_argument("--canonical-domain", type=str, default="nz210")
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--experiment-name", type=str, default="staingan_many_to_nz210")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--patches-per-source-slide", type=int, default=128)
    parser.add_argument("--mask-longest-side", type=int, default=1024)
    parser.add_argument("--patch-cache-dir", type=Path, default=Path("/mnt/Disk1/dpsn_patch_cache/staingan_patch_cache"),)
    parser.add_argument("--use-patch-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sample-count", type=int, default=36)
    parser.add_argument("--val-sample-count", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", "--batchSize", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-workers", "--nThreads", dest="num_workers", type=int, default=0)
    parser.add_argument("--epochs-constant-lr", "--niter", dest="epochs_constant_lr", type=int, default=25)
    parser.add_argument("--epochs-decay-lr", "--niter_decay", dest="epochs_decay_lr", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-content", type=float, default=10.0)
    parser.add_argument(
        "--content-loss",
        choices=("grayscale_l1", "grayscale_ssim", "none"),
        default="grayscale_l1",
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--generator-blocks", type=int, default=9)
    parser.add_argument("--discriminator-layers", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-mpp-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[1, 2, 3])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = StainGANTrainingConfig(
        dataset_dir=args.dataset_dir,
        canonical_domain=args.canonical_domain,
        checkpoints_dir=args.checkpoints_dir,
        experiment_name=args.experiment_name,
        image_size=args.image_size,
        target_mpp=args.target_mpp,
        read_level=args.read_level,
        patches_per_source_slide=args.patches_per_source_slide,
        mask_longest_side=args.mask_longest_side,
        patch_cache_dir=args.patch_cache_dir,
        use_patch_cache=args.use_patch_cache,
        train_sample_count=args.train_sample_count,
        val_sample_count=args.val_sample_count,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_constant_lr=args.epochs_constant_lr,
        epochs_decay_lr=args.epochs_decay_lr,
        lr=args.lr,
        beta1=args.beta1,
        lambda_identity=args.lambda_identity,
        lambda_content=args.lambda_content,
        content_loss=args.content_loss,
        ngf=args.ngf,
        ndf=args.ndf,
        generator_blocks=args.generator_blocks,
        discriminator_layers=args.discriminator_layers,
        pool_size=args.pool_size,
        device=args.device,
        recursive=args.recursive,
        strict_mpp_check=args.strict_mpp_check,
        verbose=args.verbose,
        gpu_ids=tuple(args.gpu_ids),
    )
    checkpoint_path = train(config)
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
