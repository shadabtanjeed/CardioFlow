"""
Trajectory correction: look ahead to the finished clip, then step back into noise.

At a backward jump in the schedule the sampler does not integrate. Instead it asks
the prior where the current state would land if it ran straight to t = 1 in one
shot, then re-noises that estimate back to an earlier time and re-walks the
stretch. Errors baked in earlier get a second chance to be corrected, which is why
Restora-Flow needs far fewer ODE steps than methods without it.

The re-noising is the step this project changes. See noise.py: drawing per-frame
noise here is what makes a naive per-frame port of Restora-Flow flicker, and
drawing once and sharing it across frames is CardioFlow's fix.
"""

import torch

from noise import draw_noise


def look_ahead(x: torch.Tensor, t: float, velocity) -> torch.Tensor:
    """
    One-shot estimate of the clean clip from the current state.

    Along the straight interpolation path the prior is trained on, the velocity is
    constant, so a single step of length (1 - t) lands on t = 1 exactly if the
    prediction is right. That estimate is only as good as the prior is at this t.
    """
    return x + (1.0 - t) * velocity(x, t)


def correct(
    x: torch.Tensor,
    t_last: float,
    t_cur: float,
    velocity,
    noise_mode: str = 'shared',
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    One trajectory-correction step, jumping backwards from `t_last` to `t_cur`.

    x          -- current state, real [B, 2, F, H, W].
    velocity   -- callable (x, t) -> predicted velocity, i.e. the trained prior.
    noise_mode -- 'shared' (CardioFlow) or 'independent' (naive per-frame port).
    """
    estimate = look_ahead(x, t_last, velocity)
    noise = draw_noise(tuple(x.shape), noise_mode, x.device, generator)
    return t_cur * estimate + (1.0 - t_cur) * noise
