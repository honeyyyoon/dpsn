from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

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

from ai.models.staingan.staingan_model import (
    GANLoss,
    ImagePool,
    NLayerDiscriminator,
    ResnetGenerator,
)
from ai.models.staingan.unpaired_domain_dataset import UnpairedDomainImageDataset


DEFAULT_APERIO_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_aperio"
)
DEFAULT_HAMAMATSU_DIR = Path(
    "/mnt/Disk1/dpsn_datasets/mitos_atypia_2014_training_hamamatsu"
)


@dataclass(slots=True)
class StainGANTrainingConfig:
    domain_a_dir: Path = DEFAULT_APERIO_DIR
    domain_b_dir: Path = DEFAULT_HAMAMATSU_DIR
    checkpoints_dir: Path = Path("checkpoints")
    experiment_name: str = "staingan_aperio_to_hamamatsu"
    image_size: int = 256
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64 # Base number of generator filters, default 64
    ndf: int = 64 # number of discriminator filters
    generator_blocks: int = 9 # how many ResNet blocks the generator has in the middle
    discriminator_layers: int = 3 # how many conv stages the discriminator has
    batch_size: int = 4
    num_workers: int = 0
    epochs_constant_lr: int = 25
    epochs_decay_lr: int = 25 # number of epochs where the learning rate gradually decreases to 0
    lr: float = 0.0002
    beta1: float = 0.5
    lambda_identity: float = 0.5
    lambda_cycle_a: float = 10.0
    lambda_cycle_b: float = 10.0
    pool_size: int = 50 #  size of the image replay buffer for discriminator
    device: str = "auto"
    recursive: bool = True
    gpu_ids: tuple[int, ...] = (1, 2, 3)

    @property
    def total_epochs(self) -> int:
        return self.epochs_constant_lr + self.epochs_decay_lr


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


def create_dataloader(config: StainGANTrainingConfig) -> DataLoader:
    dataset = UnpairedDomainImageDataset(
        domain_a_dir=config.domain_a_dir,
        domain_b_dir=config.domain_b_dir,
        image_size=config.image_size,
        recursive=config.recursive,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )


def create_models(
    config: StainGANTrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    g_a2b = ResnetGenerator(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf,
        n_blocks=config.generator_blocks,
    ).to(device)
    g_b2a = ResnetGenerator(
        input_nc=config.output_nc,
        output_nc=config.input_nc,
        ngf=config.ngf,
        n_blocks=config.generator_blocks,
    ).to(device)
    d_a = NLayerDiscriminator(
        input_nc=config.input_nc,
        ndf=config.ndf,
        n_layers=config.discriminator_layers,
    ).to(device)
    d_b = NLayerDiscriminator(
        input_nc=config.output_nc,
        ndf=config.ndf,
        n_layers=config.discriminator_layers,
    ).to(device)
    g_a2b = maybe_wrap_dataparallel(g_a2b, device, config.gpu_ids)
    g_b2a = maybe_wrap_dataparallel(g_b2a, device, config.gpu_ids)
    d_a = maybe_wrap_dataparallel(d_a, device, config.gpu_ids)
    d_b = maybe_wrap_dataparallel(d_b, device, config.gpu_ids)
    return g_a2b, g_b2a, d_a, d_b


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
    g_a2b: nn.Module,
    g_b2a: nn.Module,
    d_a: nn.Module,
    d_b: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": config.experiment_name,
            "config": asdict(config),
            "g_a2b_state_dict": unwrap_parallel(g_a2b).state_dict(),
            "g_b2a_state_dict": unwrap_parallel(g_b2a).state_dict(),
            "d_a_state_dict": unwrap_parallel(d_a).state_dict(),
            "d_b_state_dict": unwrap_parallel(d_b).state_dict(),
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


def _train_discriminator(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    gan_loss: GANLoss,
) -> torch.Tensor:
    pred_real = discriminator(real)
    loss_real = gan_loss(pred_real, True)
    pred_fake = discriminator(fake.detach())
    loss_fake = gan_loss(pred_fake, False)
    loss = 0.5 * (loss_real + loss_fake)
    loss.backward()
    return loss


def train(config: StainGANTrainingConfig) -> Path:
    device = select_device(config.device, config.gpu_ids)
    dataloader = create_dataloader(config)
    g_a2b, g_b2a, d_a, d_b = create_models(config, device) # make two generators (A->B, B->A), and two discriminators

    if device.type == "cuda":
        print(
            f"Using CUDA with DataParallel on gpu_ids={config.gpu_ids} "
            f"(primary device: cuda:{config.gpu_ids[0]})"
        )
    else:
        print(f"Using device: {device}")

    optimizer_g = Adam( # generator optimizer
        list(g_a2b.parameters()) + list(g_b2a.parameters()),
        lr=config.lr,
        betas=(config.beta1, 0.999),
    )
    optimizer_d = Adam( # discriminator optimizer
        list(d_a.parameters()) + list(d_b.parameters()),
        lr=config.lr,
        betas=(config.beta1, 0.999),
    )
    scheduler_g = build_lr_scheduler(optimizer_g, config)
    scheduler_d = build_lr_scheduler(optimizer_d, config)

    gan_loss = GANLoss().to(device)
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    fake_a_pool = ImagePool(config.pool_size) # store old generated fake imagess
    fake_b_pool = ImagePool(config.pool_size)

    latest_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    )

    for epoch in range(1, config.total_epochs + 1):
        running = { # collects cumulative loss values during the epoch
            "g": 0.0,
            "d_a": 0.0,
            "d_b": 0.0,
            "cycle": 0.0,
            "identity": 0.0,
            "batches": 0,
        }

        progress = tqdm(
            dataloader,
            desc=f"StainGAN train {epoch}/{config.total_epochs}",
            unit="batch",
            leave=False,
        )

        for real_a, real_b, _, _ in progress: # For each batch, what the dataloader returns (_: two filename outputs)
            real_a = real_a.to(device=device, dtype=torch.float32) # move images to device
            real_b = real_b.to(device=device, dtype=torch.float32)

            optimizer_g.zero_grad()

            same_b = g_a2b(real_b) # identity loss for B
            loss_idt_b = (
                identity_loss(same_b, real_b)
                * config.lambda_cycle_b
                * config.lambda_identity
            )
            same_a = g_b2a(real_a) # identity loss for A
            loss_idt_a = (
                identity_loss(same_a, real_a)
                * config.lambda_cycle_a
                * config.lambda_identity
            )

            # Forward translation A->B
            fake_b = g_a2b(real_a)  # generator makes fake b-style from A image
            pred_fake_b = d_b(fake_b) # Discriminator B looks at them
            loss_g_a2b = gan_loss(pred_fake_b, True) # Generator wants discriminator B to think these are real B images

            # Forward translation B->A
            fake_a = g_b2a(real_b)
            pred_fake_a = d_a(fake_a)
            loss_g_b2a = gan_loss(pred_fake_a, True)

            # Cycle reconstruction for A: A->B->A
            rec_a = g_b2a(fake_b)
            loss_cycle_a = cycle_loss(rec_a, real_a) * config.lambda_cycle_a # is rec_a close to real_a?
            
            # Cycle reconstruction for B: B->A->B
            rec_b = g_a2b(fake_a)
            loss_cycle_b = cycle_loss(rec_b, real_b) * config.lambda_cycle_b

            loss_g = ( # combines all generator-related objectives
                loss_g_a2b
                + loss_g_b2a
                + loss_cycle_a
                + loss_cycle_b
                + loss_idt_a
                + loss_idt_b
            )
            loss_g.backward() #computes gradients for generator parameters and updates both generators
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d_a = _train_discriminator( # Train discriminator A to distinguish fake and real
                discriminator=d_a,
                real=real_a,
                fake=fake_a_pool.query(fake_a),
                gan_loss=gan_loss,
            )
            loss_d_b = _train_discriminator(
                discriminator=d_b,
                real=real_b,
                fake=fake_b_pool.query(fake_b),
                gan_loss=gan_loss,
            )
            optimizer_d.step()

            # Record losses
            running["g"] += float(loss_g.item())
            running["d_a"] += float(loss_d_a.item())
            running["d_b"] += float(loss_d_b.item())
            running["cycle"] += float((loss_cycle_a + loss_cycle_b).item())
            running["identity"] += float((loss_idt_a + loss_idt_b).item())
            running["batches"] += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    g=f"{loss_g.item():.4f}",
                    d_a=f"{loss_d_a.item():.4f}",
                    d_b=f"{loss_d_b.item():.4f}",
                )

        scheduler_g.step()
        scheduler_d.step()

        batches = max(running["batches"], 1)
        print(
            f"epoch {epoch}/{config.total_epochs} "
            f"g={running['g']/batches:.6f} "
            f"d_a={running['d_a']/batches:.6f} "
            f"d_b={running['d_b']/batches:.6f} "
            f"cycle={running['cycle']/batches:.6f} "
            f"identity={running['identity']/batches:.6f}"
        )

        save_checkpoint(
            config=config,
            epoch=epoch,
            g_a2b=g_a2b,
            g_b2a=g_b2a,
            d_a=d_a,
            d_b=d_b,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            path=latest_checkpoint_path,
        )

    return latest_checkpoint_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train StainGAN (CycleGAN-style) between Aperio and Hamamatsu domains."
    )
    parser.add_argument("--domain-a-dir", "--source-root-train", dest="domain_a_dir", type=Path, default=DEFAULT_APERIO_DIR)
    parser.add_argument("--domain-b-dir", "--gt-root-train", dest="domain_b_dir", type=Path, default=DEFAULT_HAMAMATSU_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--experiment-name", type=str, default="staingan_aperio_to_hamamatsu")
    parser.add_argument("--image-size", "--fineSize", dest="image_size", type=int, default=256)
    parser.add_argument("--batch-size", "--batchSize", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-workers", "--nThreads", dest="num_workers", type=int, default=0)
    parser.add_argument("--epochs-constant-lr", "--niter", dest="epochs_constant_lr", type=int, default=25)
    parser.add_argument("--epochs-decay-lr", "--niter_decay", dest="epochs_decay_lr", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--lambda-cycle-a", type=float, default=10.0)
    parser.add_argument("--lambda-cycle-b", type=float, default=10.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--generator-blocks", type=int, default=9)
    parser.add_argument("--discriminator-layers", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[1, 2, 3])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = StainGANTrainingConfig(
        domain_a_dir=args.domain_a_dir,
        domain_b_dir=args.domain_b_dir,
        checkpoints_dir=args.checkpoints_dir,
        experiment_name=args.experiment_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_constant_lr=args.epochs_constant_lr,
        epochs_decay_lr=args.epochs_decay_lr,
        lr=args.lr,
        beta1=args.beta1,
        lambda_identity=args.lambda_identity,
        lambda_cycle_a=args.lambda_cycle_a,
        lambda_cycle_b=args.lambda_cycle_b,
        ngf=args.ngf,
        ndf=args.ndf,
        generator_blocks=args.generator_blocks,
        discriminator_layers=args.discriminator_layers,
        pool_size=args.pool_size,
        device=args.device,
        recursive=args.recursive,
        gpu_ids=tuple(args.gpu_ids),
    )
    checkpoint_path = train(config)
    print(f"Saved latest checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
