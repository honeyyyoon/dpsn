from __future__ import annotations

import random
from collections import deque

import torch
from torch import nn

# Residual block used inside the generator - learns how to adjust the current representation
class ResnetBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # reflection padding: pads the border by reflecting nearby pixels
            nn.Conv2d(channels, channels, kernel_size=3, bias=False), # 3x3 conv
            nn.InstanceNorm2d(channels), # instance normalization: normalizes each image independently
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    #it computes transformed features: self.block(x), then adds original input x (so output = input + learned adjustment)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

# Generator network of StainGAN
class ResnetGenerator(nn.Module): 
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64, # Base number of generator filters, default 64
        n_blocks: int = 9, # Number of residual blocks in the middle, default 9.
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be > 0, got {n_blocks}")
        
        # initial convolution: starts converting the RGB input into feature maps
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        in_channels = ngf
        out_channels = ngf * 2

        # Downsampling layers: two stride-2 convolutions.
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

        # Residual blocks: where the generator does most of its feature transformation
        for _ in range(n_blocks):
            layers.append(ResnetBlock(in_channels))

        out_channels = in_channels // 2

        # Upsampling layers: increase the spatial size back toward the original resolution
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

        # Final output layer: Converts the final features back to RGB
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

# Discriminator for StainGAN
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        kw = 4
        padw = 1

        # extracts first-level features and downsamples the image
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for layer_idx in range(1, n_layers): # adds several conv-norm-LeakyReLU blocks
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_idx, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
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
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=False,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), #final output is a 1-channel realism score map.
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Adversarial loss function used for GAN training
class GANLoss(nn.Module):
    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0) -> None:
        super().__init__()
        # Buffers store (real label = 1.0) (fake label = 0.0) as tensors attached to the module
        self.register_buffer(
            "real_label",
            torch.tensor(target_real_label, dtype=torch.float32),
        )
        self.register_buffer(
            "fake_label",
            torch.tensor(target_fake_label, dtype=torch.float32),
        )
        self.loss = nn.MSELoss()
    
    # Creates a tensor of the same shape as the discriminator output (real:1, fake:0)
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# Replay buffer used during GAN discriminator training
# Job: takes previously generated fake images and mixes them with current ones to make discriminator training more stable.
class ImagePool:
    """
    Replay buffer used by CycleGAN-style discriminator training.
    """

    def __init__(self, pool_size: int = 50) -> None: # stores up to 50 fake images by default
        if pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {pool_size}")
        self.pool_size = pool_size
        self.images: deque[torch.Tensor] = deque(maxlen=pool_size) # uses deque to keep history of generated images

    # takes a batch of newly generated fake images and returns the batch that should actually be used to train the discriminator
    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images

        selected: list[torch.Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                selected.append(image)
                continue

            if random.random() > 0.5:
                random_index = random.randrange(len(self.images))
                old = self.images[random_index].clone()
                self.images[random_index] = image.clone()
                selected.append(old)
            else:
                selected.append(image)

        return torch.cat(selected, dim=0)
