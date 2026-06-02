from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai.models.multistain.config import MultiStainCycleGANConfig
from ai.models.multistain.networks import (
    GANLoss,
    ImagePool,
    NLayerDiscriminator,
    ResnetGenerator,
    grayscale,
    init_network,
    make_optimizer,
    maybe_wrap_dataparallel,
    set_requires_grad,
    unwrap_parallel,
)


class MultiStainCycleGANModel:
    """
    Single shared many-source-to-one stain normalization model.

    Domain A is the mixture of non-canonical scanner patches. Domain B is the
    canonical NanoZoomer S210 distribution. Training follows CycleGAN: adversarial
    losses for both domains plus cycle-consistency and identity preservation.
    """

    def __init__(
        self,
        config: MultiStainCycleGANConfig,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or select_device(config.device, config.gpu_ids)

        self.net_g_source_to_target = self._build_generator()
        self.net_g_target_to_source = self._build_generator()
        self.net_d_source = self._build_discriminator()
        self.net_d_target = self._build_discriminator()

        self.criterion_gan = GANLoss().to(self.device)
        self.criterion_l1 = nn.L1Loss()

        self.fake_source_pool = ImagePool(config.pool_size)
        self.fake_target_pool = ImagePool(config.pool_size)

        self.optimizer_g = make_optimizer(
            [self.net_g_source_to_target, self.net_g_target_to_source],
            lr=config.lr,
            beta1=config.beta1,
        )
        self.optimizer_d = make_optimizer(
            [self.net_d_source, self.net_d_target],
            lr=config.lr,
            beta1=config.beta1,
        )

    def _build_generator(self) -> nn.Module:
        network = ResnetGenerator(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            ngf=self.config.ngf,
            n_blocks=self.config.generator_blocks,
        ).to(self.device)
        network = init_network(network)
        return maybe_wrap_dataparallel(network, self.device, self.config.gpu_ids)

    def _build_discriminator(self) -> nn.Module:
        network = NLayerDiscriminator(
            input_nc=self.config.output_nc,
            ndf=self.config.ndf,
            n_layers=self.config.discriminator_layers,
        ).to(self.device)
        network = init_network(network)
        return maybe_wrap_dataparallel(network, self.device, self.config.gpu_ids)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one full generator/discriminator optimization step."""

        self.train()
        source = self._batch_tensor(batch, "source")
        target = self._batch_tensor(batch, "target")

        set_requires_grad([self.net_d_source, self.net_d_target], False)
        self.optimizer_g.zero_grad(set_to_none=True)
        generated = self._forward_generators(source, target)
        loss_g, generator_losses = self._generator_loss(source, target, generated)
        loss_g.backward()
        self.optimizer_g.step()

        set_requires_grad([self.net_d_source, self.net_d_target], True)
        self.optimizer_d.zero_grad(set_to_none=True)
        loss_d_source, loss_d_target = self._discriminator_losses(
            source=source,
            target=target,
            fake_source=generated["fake_source"],
            fake_target=generated["fake_target"],
        )
        loss_d = loss_d_source + loss_d_target
        loss_d.backward()
        self.optimizer_d.step()

        return {
            **generator_losses,
            "d_source": float(loss_d_source.item()),
            "d_target": float(loss_d_target.item()),
            "d": float(loss_d.item()),
        }

    @torch.inference_mode()
    def validation_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Compute validation losses without updating model weights."""

        self.eval()
        source = self._batch_tensor(batch, "source")
        target = self._batch_tensor(batch, "target")
        generated = self._forward_generators(source, target)
        _, losses = self._generator_loss(source, target, generated)

        if "aligned_target" in batch:
            aligned_target = self._batch_tensor(batch, "aligned_target")
            losses["aligned_l1"] = float(
                self.criterion_l1(generated["fake_target"], aligned_target).item()
            )

        return losses

    @torch.inference_mode()
    def normalize(self, source: torch.Tensor) -> torch.Tensor:
        """Normalize source scanner patches into the canonical target style."""

        self.eval()
        source = source.to(device=self.device, dtype=torch.float32)
        return self.net_g_source_to_target(source)

    def train(self) -> None:
        self.net_g_source_to_target.train()
        self.net_g_target_to_source.train()
        self.net_d_source.train()
        self.net_d_target.train()

    def eval(self) -> None:
        self.net_g_source_to_target.eval()
        self.net_g_target_to_source.eval()
        self.net_d_source.eval()
        self.net_d_target.eval()

    def checkpoint_state(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return a serializable checkpoint payload for train.py."""

        return {
            "epoch": epoch,
            "experiment_name": self.config.experiment_name,
            "config": self.config.to_dict(),
            "metrics": metrics or {},
            "net_g_source_to_target": unwrap_parallel(
                self.net_g_source_to_target
            ).state_dict(),
            "net_g_target_to_source": unwrap_parallel(
                self.net_g_target_to_source
            ).state_dict(),
            "net_d_source": unwrap_parallel(self.net_d_source).state_dict(),
            "net_d_target": unwrap_parallel(self.net_d_target).state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        load_optimizers: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        unwrap_parallel(self.net_g_source_to_target).load_state_dict(
            checkpoint["net_g_source_to_target"]
        )
        unwrap_parallel(self.net_g_target_to_source).load_state_dict(
            checkpoint["net_g_target_to_source"]
        )
        unwrap_parallel(self.net_d_source).load_state_dict(checkpoint["net_d_source"])
        unwrap_parallel(self.net_d_target).load_state_dict(checkpoint["net_d_target"])

        if load_optimizers:
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        return checkpoint

    def _forward_generators(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fake_target = self.net_g_source_to_target(source)
        reconstructed_source = self.net_g_target_to_source(fake_target)

        fake_source = self.net_g_target_to_source(target)
        reconstructed_target = self.net_g_source_to_target(fake_source)

        identity_target = self.net_g_source_to_target(target)
        identity_source = self.net_g_target_to_source(source)

        return {
            "fake_target": fake_target,
            "fake_source": fake_source,
            "reconstructed_source": reconstructed_source,
            "reconstructed_target": reconstructed_target,
            "identity_target": identity_target,
            "identity_source": identity_source,
        }

    def _generator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        generated: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        fake_target = generated["fake_target"]
        fake_source = generated["fake_source"]

        loss_g_source_to_target = self.criterion_gan(
            self.net_d_target(fake_target),
            True,
        )
        loss_g_target_to_source = self.criterion_gan(
            self.net_d_source(fake_source),
            True,
        )

        loss_cycle_source = (
            self.criterion_l1(generated["reconstructed_source"], source)
            * self.config.lambda_cycle
        )
        loss_cycle_target = (
            self.criterion_l1(generated["reconstructed_target"], target)
            * self.config.lambda_cycle
        )

        loss_identity_source = (
            self.criterion_l1(generated["identity_source"], source)
            * self.config.lambda_identity
        )
        loss_identity_target = (
            self.criterion_l1(generated["identity_target"], target)
            * self.config.lambda_identity
        )

        loss_content = source.new_tensor(0.0)
        if self.config.lambda_content > 0:
            loss_content = (
                self.criterion_l1(grayscale(fake_target), grayscale(source))
                * self.config.lambda_content
            )

        loss_g = (
            loss_g_source_to_target
            + loss_g_target_to_source
            + loss_cycle_source
            + loss_cycle_target
            + loss_identity_source
            + loss_identity_target
            + loss_content
        )

        return loss_g, {
            "g": float(loss_g.item()),
            "g_source_to_target": float(loss_g_source_to_target.item()),
            "g_target_to_source": float(loss_g_target_to_source.item()),
            "cycle_source": float(loss_cycle_source.item()),
            "cycle_target": float(loss_cycle_target.item()),
            "identity_source": float(loss_identity_source.item()),
            "identity_target": float(loss_identity_target.item()),
            "content": float(loss_content.item()),
        }

    def _discriminator_losses(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake_source: torch.Tensor,
        fake_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_fake_source = self.fake_source_pool.query(fake_source)
        pooled_fake_target = self.fake_target_pool.query(fake_target)

        loss_d_source = self._discriminator_loss(
            discriminator=self.net_d_source,
            real=source,
            fake=pooled_fake_source,
        )
        loss_d_target = self._discriminator_loss(
            discriminator=self.net_d_target,
            real=target,
            fake=pooled_fake_target,
        )
        return loss_d_source, loss_d_target

    def _discriminator_loss(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        pred_real = discriminator(real)
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = discriminator(fake.detach())
        loss_fake = self.criterion_gan(pred_fake, False)

        return 0.5 * (loss_real + loss_fake)

    def _batch_tensor(self, batch: dict[str, Any], key: str) -> torch.Tensor:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.to(device=self.device, dtype=torch.float32)


def select_device(device: str, gpu_ids: tuple[int, ...] = (1, 2, 3)) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
