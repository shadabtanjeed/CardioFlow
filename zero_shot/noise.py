"""
Where CardioFlow differs from Restora-Flow.

Restora-Flow restores single images, so every time it draws fresh noise it draws
`randn_like(x)` and there is nothing more to say. Applied frame-by-frame to a cine
clip that same call draws an *independent* noise field per frame. The clip's
anatomy is nearly identical from frame to frame, but each frame then gets pushed
in its own random direction, and the reconstruction flickers along the temporal
axis -- the artefact this project exists to remove.

`shared` draws one noise field and broadcasts it across all frames, so a
correction nudges every frame the same way and the temporal structure survives.
Everything else about the algorithm is unchanged, which is what makes the two
modes a clean A/B ablation.
"""

import torch


MODES = ('shared', 'independent')


def draw_noise(
    shape: tuple[int, ...],
    mode: str,
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Gaussian noise shaped [B, C, F, H, W].

      independent -- one draw per frame (the naive per-frame extension).
      shared      -- a single [B, C, 1, H, W] draw broadcast over the F frames.

    The generator lives on the CPU so a run is reproducible regardless of device.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown noise mode {mode!r}; expected one of {MODES}.')

    batch, channels, frames, height, width = shape
    draw_shape = (batch, channels, 1, height, width) if mode == 'shared' else shape
    noise = torch.randn(draw_shape, generator=generator, dtype=dtype)
    if mode == 'shared':
        noise = noise.expand(-1, -1, frames, -1, -1)
    return noise.to(device)
