"""
3D U-Net velocity field v_theta(x_t, t) for the CardioFlow flow-matching prior.

Convolutions are 3x3x3 over (frame, x, y), but down/up-sampling is spatial only
(stride (1,2,2)) -- the temporal axis keeps full resolution through the whole
network. That means the frame count is unconstrained; only H and W must be
divisible by 2 ** (len(channel_mult) - 1).

Temporal padding is circular, matching CineVN's `pad_mode_temp: circular`: a cine
clip covers (part of) one cardiac cycle, so zero-padding the frame axis would
fabricate a discontinuity at both clip edges in every single layer -- exactly the
wrong prior for a project about temporal consistency.

There are deliberately no attention layers. That keeps the network fully
convolutional, so a prior trained on crops can be applied at the full FOV the
Phase 3 k-t sampler needs (GroupNorm statistics still shift with input size, so
that transfer is worth checking rather than assuming).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding. `t` may be fractional (flow matching uses continuous time)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _norm(channels: int) -> nn.GroupNorm:
    # Skip-concatenation produces channel counts like 48 that 32 does not divide.
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _zero_init(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


class Conv3x3x3(nn.Module):
    """
    3x3x3 conv that pads circularly in time and with zeros in space.

    Conv3d applies one `padding_mode` to every padded axis, so the mixed scheme is
    done with two explicit F.pad calls instead (pad order is W, H, then D).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode='circular')
        x = F.pad(x, (1, 1, 1, 1, 0, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            _norm(in_ch), nn.SiLU(), Conv3x3x3(in_ch, out_ch)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, 2 * out_ch))
        self.out_norm = _norm(out_ch)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), _zero_init(Conv3x3x3(out_ch, out_ch))
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = Conv3x3x3(channels, channels, stride=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv3x3x3(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        return self.conv(x)


class _Sequential(nn.Sequential):
    """nn.Sequential that forwards the timestep embedding to ResBlocks."""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResBlock) else layer(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        model_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_downsamples = len(channel_mult) - 1

        emb_ch = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )

        ch = model_channels
        self.input_blocks = nn.ModuleList([_Sequential(Conv3x3x3(in_channels, ch))])
        skip_channels = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.input_blocks.append(_Sequential(ResBlock(ch, model_channels * mult, emb_ch, dropout)))
                ch = model_channels * mult
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(_Sequential(Downsample(ch)))
                skip_channels.append(ch)

        self.middle_block = _Sequential(
            ResBlock(ch, ch, emb_ch, dropout), ResBlock(ch, ch, emb_ch, dropout)
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers: list[nn.Module] = [
                    ResBlock(ch + skip_channels.pop(), model_channels * mult, emb_ch, dropout)
                ]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                self.output_blocks.append(_Sequential(*layers))

        self.out = nn.Sequential(
            _norm(ch), nn.SiLU(), _zero_init(Conv3x3x3(ch, out_channels))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F, H, W], t: [B] (already scaled for the embedding)."""
        divisor = 2 ** self.num_downsamples
        if x.shape[-1] % divisor or x.shape[-2] % divisor:
            raise ValueError(
                f'Spatial dims {tuple(x.shape[-2:])} must be divisible by {divisor}; adjust data.crop.'
            )

        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        hs = []
        h = x
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)
