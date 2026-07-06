# Largely based on PDEArena pdearena/modules/twod_unet.py (MIT license)
# and https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .activations import ACTIVATION_REGISTRY


DEFAULT_CH_MULTS = (1, 2, 2, 3, 4)


def _default_ch_mults(depth: int = 4) -> Tuple[int, ...]:
    return DEFAULT_CH_MULTS


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, *spatial_shape)
        return res


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ModernUNet3D(nn.Module):
    """PDEArena Modern U-Net ported to 3D spatial dimensions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = DEFAULT_CH_MULTS,
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False, False),
        mid_attn: bool = True,
        n_blocks: int = 2,
        use1x1: bool = True,
    ) -> None:
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        n_resolutions = len(ch_mults)
        n_channels = hidden_channels

        if use1x1:
            self.image_proj = nn.Conv3d(in_channels, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv3d(in_channels, n_channels, kernel_size=3, padding=1)

        down = []
        out_ch = in_ch = n_channels
        for i in range(n_resolutions):
            out_ch = in_ch * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_ch,
                        out_ch,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
                in_ch = out_ch
            if i < n_resolutions - 1:
                down.append(Downsample(in_ch))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_ch, has_attn=mid_attn, activation=activation, norm=norm)

        up = []
        in_ch = out_ch
        for i in reversed(range(n_resolutions)):
            out_ch = in_ch
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_ch,
                        out_ch,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
            out_ch = in_ch // ch_mults[i]
            up.append(UpBlock(in_ch, out_ch, has_attn=is_attn[i], activation=activation, norm=norm))
            in_ch = out_ch
            if i > 0:
                up.append(Upsample(in_ch))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels) if norm else nn.Identity()
        if use1x1:
            self.final = nn.Conv3d(in_ch, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        return self.final(self.activation(self.norm(x)))


class UNet3D(nn.Module):
    """Drop-in wrapper matching SingleTensorAdapter with identity residual on state channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        depth: int = 3,
        channel_multipliers: Optional[Union[Tuple[int, ...], List[int]]] = None,
        n_blocks: int = 2,
        norm: bool = True,
        activation: str = "gelu",
        is_attn: Optional[Union[Tuple[bool, ...], List[bool]]] = None,
        mid_attn: bool = False,
        use1x1: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        ch_mults = tuple(channel_multipliers) if channel_multipliers is not None else _default_ch_mults(depth)
        if is_attn is None:
            is_attn = tuple(False for _ in ch_mults)

        self.core = ModernUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            norm=norm,
            ch_mults=ch_mults,
            is_attn=is_attn,
            mid_attn=mid_attn,
            n_blocks=n_blocks,
            use1x1=use1x1,
        )
        nn.init.zeros_(self.core.final.weight)
        nn.init.zeros_(self.core.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x) + x[:, : self.out_channels]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.unet3d import UNet3D  # noqa: E402

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x = torch.randn(1, 6, 16, 64, 32).to(device)
    model = UNet3D(in_channels=6, out_channels=4, hidden_channels=64).to(device)
    out = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"UNet3D — Input: {x.shape}, Output: {out.shape}, Params: {total:,}")
