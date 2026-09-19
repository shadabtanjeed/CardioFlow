"""
Flow matching objective and ODE sampler.

Convention (matches the proposal): t = 0 is noise, t = 1 is data.
    x_t      = (1 - t) * x0 + t * x1
    target   = x1 - x0
    L(theta) = E_{t, x0, x1} || v_theta(x_t, t) - (x1 - x0) ||^2
"""

import torch
import torch.nn as nn


class FlowMatching(nn.Module):
    def __init__(self, velocity_fn: nn.Module, time_scale: float = 1000.0):
        """`time_scale` maps t in [0,1] onto the range the sinusoidal embedding expects."""
        super().__init__()
        self.velocity_fn = velocity_fn
        self.time_scale = time_scale

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity_fn(x, t * self.time_scale)

    @staticmethod
    def interpolate(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ = t.reshape(-1, *([1] * (x1.ndim - 1)))
        return (1 - t_) * x0 + t_ * x1

    def loss(
        self,
        x1: torch.Tensor,
        t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        if x0 is None:
            x0 = torch.randn_like(x1)
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device)
        x_t = self.interpolate(x1, x0, t)
        v_pred = self.velocity(x_t, t)
        per_sample = ((v_pred - (x1 - x0)) ** 2).flatten(1).mean(1)
        return per_sample.mean() if reduce else per_sample

    @torch.no_grad()
    def integrate(
        self,
        x: torch.Tensor,
        steps: int,
        t_start: float = 0.0,
        t_end: float = 1.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Euler integration of dx/dt = v_theta(x, t) from t_start to t_end."""
        dt = (t_end - t_start) / steps
        trajectory = [x]
        for i in range(steps):
            t = torch.full((x.shape[0],), t_start + i * dt, device=x.device)
            x = x + self.velocity(x, t) * dt
            if return_trajectory:
                trajectory.append(x)
        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], steps: int, device, generator=None) -> torch.Tensor:
        """Generate clips from pure noise."""
        x0 = torch.randn(shape, generator=generator).to(device)
        return self.integrate(x0, steps=steps)

    @torch.no_grad()
    def restore(self, x1: torch.Tensor, t_start: float, steps: int, generator=None) -> torch.Tensor:
        """
        Noise a real clip up to `t_start`, then integrate back to t=1.

        Not a reconstruction task -- just a sanity check that the learned field
        transports corrupted cine data back onto the data manifold.
        """
        x0 = torch.randn(x1.shape, generator=generator).to(x1.device)
        x_t = self.interpolate(x1, x0, torch.full((x1.shape[0],), t_start, device=x1.device))
        return self.integrate(x_t, steps=steps, t_start=t_start, t_end=1.0)


def stratified_t(n: int, device, offset: float = 0.5) -> torch.Tensor:
    """Evenly spaced times in (0, 1) -- used for low-variance validation."""
    return ((torch.arange(n, device=device, dtype=torch.float32) + offset) / n).clamp(1e-4, 1 - 1e-4)
