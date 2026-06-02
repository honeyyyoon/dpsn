from __future__ import annotations

import random
from collections import deque

import torch
from torch import nn


class ResnetBlock(nn.Module):
    """Residual block used by the CycleGAN ResNet generator."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet generator used for stain translation.

    This follows the standard CycleGAN generator family: 7x7 stem, two
    downsampling blocks, residual blocks, two upsampling blocks, and a tanh RGB
    output in the same numeric range as the training patches: [-1, 1].
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be > 0, got {n_blocks}")

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        in_channels = ngf
        out_channels = ngf * 2
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
            out_channels *= 2

        for _ in range(n_blocks):
            layers.append(ResnetBlock(in_channels))

        out_channels = in_channels // 2
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
            out_channels //= 2

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, output_nc, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator used by CycleGAN."""

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")

        kernel_size = 4
        padding = 1
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for layer_idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_idx, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    ndf * nf_mult,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANLoss(nn.Module):
    """Least-squares GAN loss used by the original CycleGAN implementation."""

    def __init__(
        self,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        label = self.real_label if target_is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target)


class ImagePool:
    """
    Replay buffer for generated images.

    CycleGAN feeds discriminators a mixture of current and previously generated
    fake images, which reduces oscillation during adversarial training.
    """

    def __init__(self, pool_size: int = 50) -> None:
        if pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {pool_size}")
        self.pool_size = int(pool_size)
        self.images: deque[torch.Tensor] = deque(maxlen=self.pool_size)

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images.detach()

        selected: list[torch.Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                selected.append(image)
                continue

            if random.random() > 0.5:
                index = random.randrange(len(self.images))
                old = self.images[index].clone()
                self.images[index] = image.clone()
                selected.append(old)
            else:
                selected.append(image)

        return torch.cat(selected, dim=0)


def init_weights(module: nn.Module, init_gain: float = 0.02) -> None:
    """Initialize convolutional layers with CycleGAN's normal initialization."""

    classname = module.__class__.__name__
    if hasattr(module, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        nn.init.normal_(module.weight.data, 0.0, init_gain)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, init_gain)
        nn.init.constant_(module.bias.data, 0.0)


def init_network(network: nn.Module, init_gain: float = 0.02) -> nn.Module:
    network.apply(lambda module: init_weights(module, init_gain=init_gain))
    return network


def set_requires_grad(networks: nn.Module | list[nn.Module], requires_grad: bool) -> None:
    if not isinstance(networks, list):
        networks = [networks]
    for network in networks:
        for parameter in network.parameters():
            parameter.requires_grad = requires_grad


def unwrap_parallel(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def maybe_wrap_dataparallel(
    module: nn.Module,
    device: torch.device,
    gpu_ids: tuple[int, ...],
) -> nn.Module:
    if device.type == "cuda" and len(gpu_ids) > 1:
        return nn.DataParallel(module, device_ids=list(gpu_ids))
    return module


def grayscale(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized RGB tensors to grayscale while preserving [-1, 1] range."""

    r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def make_optimizer(
    modules: list[nn.Module],
    lr: float,
    beta1: float,
) -> torch.optim.Adam:
    params = []
    for module in modules:
        params.extend(module.parameters())
    return torch.optim.Adam(params, lr=lr, betas=(beta1, 0.999))
