from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# 큰 feature map을 작은 window들로 나누는 함수
def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor: #입력 텐서 형식: (batch size, H: height, W: width, C: channel)
    """
    Partition a BHWC tensor into non-overlapping windows.
    """
    b, h, w, c = x.shape #extract the shape of each dimension
    x = x.view( #split in terms of height and width, and the rest (batch, channel) remain => windows
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() #차원 순서를 바꿔서 window를 한 덩어리씩 꺼내기 좋게 재배치 (B, H//ws, W//ws, ws, ws, C)
    return windows.view(-1, window_size * window_size, c) #모든 window를 하나의 큰 batch처럼 펼친다.
    #final shape: (8, 64, 30) (window, # of tokens, channel)

# 나눠진 window들을 다시 원래 feature map으로 합치는 함수. 위에서 잘라놓은 window를 다시 (B, H, W, C)로 바꾼다.
def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Restore window-partitioned tokens to a padded BHWC tensor.
    """
    num_windows_per_image = (h // window_size) * (w // window_size)
    b = windows.shape[0] // num_windows_per_image # batch size 다시 복원
    x = windows.view( #flat하게 펴 놨던 window들을 다시 grid 구조로 복원하는 단계
        b,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() #window 내부 픽셀 차원의 순서를 다시 원래 이미지 복원용으로 바꾼다
    return x.view(b, h, w, -1)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int, #dim of input feature
        mlp_ratio: float = 4.0, # hidden layer를 얼마나 크게 만들지
        dropout: float = 0.0, # 학습 중 일부 값을 랜덤하게 꺼서 과적합을 줄이는 regularization part
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() #GELU activation function
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Swin Transformer의 핵심 attention 연산
# 한 개의 local window 안에서 multi-head self-attention을 수행하고 relative position bias를 더해서 공간적 위치 정보를 반영하는 모듈
class WindowAttention(nn.Module):
    """
    Standard Swin-style window attention with relative position bias.
    input x dimension: (batch including # of windows, # of tokens inside window , # of channels)
    n = 64, then its doing attention across the 64 tokens
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int, # num of attention heads (for parallel processing of attention)
        qkv_bias: bool = True, # whether to include bias in q/k/v linear
        attn_dropout: float = 0.0, # attention map dropout
        proj_dropout: float = 0.0, # 마지막 projection dropout
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads #각 attention head가 담당하는 feature 차원
        self.scale = self.head_dim ** -0.5 #attention에서 query를 scaling할 때 쓰는 값

        #상대 위치별 bias 값 표
        relative_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_bias = relative_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads,
        )
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Shifted-window self-attention block operating directly on BCHW feature maps.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError(
                f"shift_size must be within [0, window_size), got {shift_size}"
            )

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def _build_attention_mask(
        self,
        padded_h: int,
        padded_w: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, padded_h, padded_w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        index = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = index
                index += 1

        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SwinTransformerBlock expects BCHW input, got {x.shape}")
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Channel dimension must match dim={self.dim}, got {x.shape[1]}"
            )

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1], x.shape[2]

        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
            attn_mask = self._build_attention_mask(padded_h, padded_w, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = _window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = _window_reverse(attn_windows, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :]

        x = shortcut[:, :h, :w, :] + x
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()


class ResStainSWINBlock(nn.Module):
    """
    Residual high-level feature block built from stacked Swin transformer blocks.

    The paper highlights ResStainSWIN as a residual super-resolution-style module
    that combines high-level features extracted by STBs. This implementation
    follows that description by alternating regular and shifted-window STBs,
    then fusing them through a convolutional residual projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        num_stb: int = 2,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_stb <= 0:
            raise ValueError(f"num_stb must be > 0, got {num_stb}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(
                f"conv_kernel_size must be a positive odd number, got {conv_kernel_size}"
            )

        shift_size = window_size // 2
        blocks: list[nn.Module] = []
        for index in range(num_stb):
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else shift_size,
                    mlp_ratio=mlp_ratio,
                )
            )

        self.stb_layers = nn.Sequential(*blocks)
        padding = conv_kernel_size // 2
        self.fuse = nn.Conv2d(
            dim,
            dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.stb_layers(x)
        x = self.fuse(x)
        return x + residual


class StainSWIN(nn.Module):
    """
    Best-faith PyTorch implementation of the StainSWIN model.

    The paper describes three stages:
    1. low-level feature extraction
    2. high-level feature extraction using STB and ResStainSWIN
    3. image reconstruction

    This implementation keeps that structure explicit and uses a global image
    residual connection, which is well aligned with the paper's residual
    super-resolution framing while preserving the original tissue structure.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        embed_dim: int = 96,
        num_heads: int = 6,
        num_res_blocks: int = 6,
        stbs_per_block: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        conv_kernel_size: int = 3,
        reconstruction_channels: int | None = None,
        use_image_residual: bool = True,
    ) -> None:
        super().__init__()
        if input_nc <= 0:
            raise ValueError(f"input_nc must be > 0, got {input_nc}")
        if output_nc <= 0:
            raise ValueError(f"output_nc must be > 0, got {output_nc}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
            )
        if num_res_blocks <= 0:
            raise ValueError(f"num_res_blocks must be > 0, got {num_res_blocks}")
        if stbs_per_block <= 0:
            raise ValueError(f"stbs_per_block must be > 0, got {stbs_per_block}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(
                f"conv_kernel_size must be a positive odd number, got {conv_kernel_size}"
            )

        padding = conv_kernel_size // 2
        recon_channels = reconstruction_channels or embed_dim

        # Low-level feature extraction.
        self.shallow_feature_extractor = nn.Conv2d(
            input_nc,
            embed_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            bias=True,
        )

        # High-level feature extraction with stacked residual StainSWIN blocks.
        self.high_level_feature_extractor = nn.Sequential(
            *[
                ResStainSWINBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    num_stb=stbs_per_block,
                    mlp_ratio=mlp_ratio,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.high_level_fuse = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
            bias=True,
        )

        # Image reconstruction.
        self.reconstruction = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                recon_channels,
                kernel_size=conv_kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                recon_channels,
                output_nc,
                kernel_size=conv_kernel_size,
                padding=padding,
                bias=True,
            ),
        )

        self.use_image_residual = use_image_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"StainSWIN expects [N, C, H, W], got {x.shape}")
        if x.shape[1] != self.shallow_feature_extractor.in_channels:
            raise ValueError(
                "Input channel count does not match model configuration: "
                f"expected {self.shallow_feature_extractor.in_channels}, got {x.shape[1]}"
            )

        shallow = self.shallow_feature_extractor(x)
        deep = self.high_level_feature_extractor(shallow)
        deep = self.high_level_fuse(deep) + shallow
        output = self.reconstruction(deep)

        if self.use_image_residual and x.shape[1] == output.shape[1]:
            output = output + x

        return output

