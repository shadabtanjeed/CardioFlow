"""
k-space operators for the zero-shot k-t sampler.

Conventions (verified against the stored OCMR preprocessing output, 2026-09-19):
  * Array layout is `[slice, coil, frame, kx, ky]`; the undersampling mask is
    `[frame, ky]`, i.e. whole phase-encode lines are kept or dropped per frame.
    That per-frame variation is the "k-t" in k-t sampling.
  * `fftshift(ifft2(ifftshift(k), norm='ortho'))` over the last two axes
    round-trips the stored k-space to the stored `reconstruction_rss` exactly.
  * ESPIRiT maps from `coil_sens/<id>/coil_sens_avg.h5` satisfy sum_c |S_c|^2 = 1
    at every pixel (checked: min = max = 1.0000). So the weighted coil combination
    `sum_c conj(S_c) * x_c` is already the properly normalised SENSE adjoint and
    needs no extra division -- and A = M . F . S is an isometry on its range.

The prior is trained on `reconstruction_weighted`, i.e. the weighted coil-combined
complex image, so that is the domain every operator here maps into and out of.
"""

import torch

# Measured on the local test split (12 files x 4 accelerations, 2026-09-19):
# std of the zero-filled reconstruction divided by std of the fully-sampled
# reference, using the same `std over stacked (real, imag)` that
# flow_prior/dataset.py uses to normalise training clips.
#
# Two corrections to the single 0.44 constant recorded earlier, which does not
# reproduce:
#   1. The ratio is NOT acceleration-independent -- it falls steadily as R rises,
#      so one pooled constant would be off by up to 20%.
#   2. It differs by coil mode, because the multicoil adjoint loses energy to
#      aliasing that the single-coil projection does not.
# Within one (mode, R) cell it is stable to 2-7%, which is what makes it usable.
ZERO_FILLED_SCALE = {
    'multicoil': {8: 0.3202, 12: 0.2587, 16: 0.2235, 20: 0.1969},  # +/- 6-7%
    'combined': {8: 0.4933, 12: 0.4019, 16: 0.3475, 20: 0.3064},   # +/- 2-4%
}


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centred orthonormal 2D FFT over the last two axes."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1)
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centred orthonormal 2D inverse FFT over the last two axes."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1)
    )


def complex_std(x: torch.Tensor) -> torch.Tensor:
    """
    std over the stacked (real, imag) components -- byte-for-byte the same
    statistic `flow_prior/dataset.py::_scale` uses for `normalize: std`, so the
    scale we hand the prior at inference matches the one it trained under.
    """
    return torch.stack([x.real, x.imag]).std()


def to_channels(x: torch.Tensor) -> torch.Tensor:
    """complex [B, F, H, W] -> real [B, 2, F, H, W], the layout the prior expects."""
    return torch.stack([x.real, x.imag], dim=1)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """real [B, 2, F, H, W] -> complex [B, F, H, W]."""
    return torch.complex(x[:, 0], x[:, 1])


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Zero-pad the trailing two axes up to a multiple of `multiple`, symmetrically.

    The U-Net needs H and W divisible by 2 ** (len(channel_mult) - 1), but the
    acquisition grid does not oblige: of the 12 distinct k-space sizes in the test
    split, several have a phase-encode extent like 126, 150 or 174. Padding happens
    around each network call only -- every k-space operation stays on the true
    acquisition grid, so the forward model is never distorted.
    """
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if not pad_h and not pad_w:
        return x, (0, 0, 0, 0)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return torch.nn.functional.pad(x, pad), pad


def unpad(x: torch.Tensor, pad: tuple[int, ...]) -> torch.Tensor:
    """Undo `pad_to_multiple`."""
    left, right, top, bottom = pad
    if not any(pad):
        return x
    return x[..., top : x.shape[-2] - bottom, left : x.shape[-1] - right]


def center_crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Centre-crop the trailing two axes to `shape`.

    The stored `reconstruction_weighted` is a centre crop of the full acquisition
    FOV (phase oversampling removed) for 22 of the 38 test clips -- verified by
    searching all offsets for the one that best aligns the zero-filled recon with
    the reference, which lands on the centre every time. Reconstructions are made
    on the full grid and cropped to this shape only for scoring.
    """
    height, width = shape
    full_h, full_w = x.shape[-2:]
    if (full_h, full_w) == (height, width):
        return x
    top, left = (full_h - height) // 2, (full_w - width) // 2
    return x[..., top : top + height, left : left + width]


class KTOperator:
    """
    The forward model A(x) = M . F . S and its adjoint, for one clip.

    `sens` is the ESPIRiT map stack [B, C, 1, H, W], or None for `coil_mode:
    combined`, where the problem is treated as a single virtual coil (S = 1) and
    the undersampling is simulated directly on the coil-combined image. Combined
    mode makes mask fusion an exact orthogonal projection, which is the cleanest
    setting to isolate the shared-noise ablation in; multicoil mode is the real
    inverse problem CineVN is benchmarked on.

    `mask` is [B, 1, F, 1, W] (broadcast over coils and over the readout axis kx).
    """

    def __init__(self, mask: torch.Tensor, sens: torch.Tensor | None = None):
        self.mask = mask
        self.sens = sens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """coil-combined image [B, F, H, W] -> masked coil k-space [B, C, F, H, W]."""
        return self.mask * self.unmasked_forward(x)

    def unmasked_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same as `forward` but without applying the mask -- the model's own k-space."""
        coil_images = x.unsqueeze(1) if self.sens is None else self.sens * x.unsqueeze(1)
        return fft2c(coil_images)

    def adjoint(self, k: torch.Tensor) -> torch.Tensor:
        """coil k-space [B, C, F, H, W] -> coil-combined image [B, F, H, W]."""
        coil_images = ifft2c(k)
        if self.sens is None:
            return coil_images.squeeze(1)
        return (self.sens.conj() * coil_images).sum(dim=1)

    def zero_filled(self, y: torch.Tensor) -> torch.Tensor:
        """The adjoint reconstruction of the measured data -- the usual starting point."""
        return self.adjoint(y)


def estimate_scale(
    zero_filled: torch.Tensor,
    acceleration: int,
    mode: str = 'auto',
    constant: float | None = None,
    reference: torch.Tensor | None = None,
    coil_mode: str = 'multicoil',
) -> torch.Tensor:
    """
    The divisor that puts this clip on the scale the prior was trained at.

    Training divided each clip by `complex_std(fully_sampled_reference)`, which
    does not exist in the zero-shot setting. `auto` recovers it from the measured
    data alone via the per-R constants in ZERO_FILLED_SCALE.

      auto      -- complex_std(zero_filled) / ZERO_FILLED_SCALE[R]   (the real setting)
      constant  -- same, but with an explicit ratio supplied by the caller
      reference -- complex_std(fully-sampled reference). ORACLE: only valid for
                   debugging, since it reads data the sampler is not allowed to see.
                   Useful for separating "the scale estimate is off" from "the
                   sampler is wrong".
    """
    if mode == 'reference':
        if reference is None:
            raise ValueError("scale mode 'reference' needs the fully-sampled reference.")
        return complex_std(reference)

    if mode == 'constant':
        if constant is None:
            raise ValueError("scale mode 'constant' needs data.scale_constant.")
        ratio = constant
    elif mode == 'auto':
        table = ZERO_FILLED_SCALE.get(coil_mode)
        if table is None:
            raise ValueError(f'No scale table for coil_mode {coil_mode!r}.')
        if acceleration not in table:
            raise ValueError(
                f'No measured zero-filled scale constant for R={acceleration} in '
                f'{coil_mode} mode; known: {sorted(table)}. Pass '
                f'--set data.scale_mode=constant --set data.scale_constant=<ratio> instead.'
            )
        ratio = table[acceleration]
    else:
        raise ValueError(f'Unknown data.scale_mode {mode!r}; expected auto, constant, reference.')

    return complex_std(zero_filled) / ratio
