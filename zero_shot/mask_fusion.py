"""
Mask fusion: keep what the scanner actually measured, let the prior invent the rest.

Restora-Flow fuses in the pixel domain, because its masks are pixel masks. Here the
mask lives in k-space (whole phase-encode lines, varying per frame), so the fusion
moves there: transform the current state to k-space, overwrite the measured lines
with the observation re-noised to the current flow time, keep the model's own
values everywhere else, and transform back.

Re-noising the observation to time t -- `t * y + (1 - t) * noise` -- is what keeps
the injected data on the same interpolation path the prior was trained against
(`x_t = t * x1 + (1 - t) * x0` in flow_prior/flow.py). Injecting clean data into a
state that is still mostly noise would put the trajectory off-distribution.
"""

import torch

from kspace import to_channels, to_complex
from noise import draw_noise


def fuse(
    x: torch.Tensor,
    t: float,
    operator,
    observation: torch.Tensor,
    noise_mode: str = 'independent',
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    One mask-fusion step.

    x            -- current state, real [B, 2, F, H, W].
    t            -- current flow time; 0 is noise, 1 is data.
    operator     -- KTOperator carrying the mask and (optionally) the coil maps.
    observation  -- measured k-space [B, C, F, H, W], already on the prior's scale.
    noise_mode   -- 'independent' matches Restora-Flow; 'shared' applies the same
                    across-frame draw used for trajectory correction. Kept separate
                    from the correction's mode so the ablation can vary one at a time.
    """
    state_kspace = operator.unmasked_forward(to_complex(x))

    epsilon = to_complex(draw_noise(tuple(x.shape), noise_mode, x.device, generator))
    noise_kspace = operator.unmasked_forward(epsilon)

    observed_at_t = t * observation + (1.0 - t) * noise_kspace
    fused = operator.mask * observed_at_t + (1.0 - operator.mask) * state_kspace

    return to_channels(operator.adjoint(fused))
