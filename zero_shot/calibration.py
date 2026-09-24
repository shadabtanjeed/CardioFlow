"""
On-the-fly scale calibration from `ocmr_val` -- the fix for a real leak.

`kspace.estimate_scale`'s `constant` mode needs a divisor that recovers the scale
`flow_prior` trained at (`complex_std(fully_sampled_reference)`) from the measured
data alone. That divisor used to be a hardcoded per-`(coil_mode, R)` lookup table
in `kspace.py`, `ZERO_FILLED_SCALE`, whose comment said outright it was "measured
on the local test split" -- i.e. the same 38 clips the ablation and headline
results are scored on. Not per-clip oracle leakage (`scale_mode: reference`, which
was correctly never used for the reported numbers), but a population-level
statistic derived from the test split's own ground truth, baked into a constant
and then applied back to that split. See notes.md.

This module replaces the table with a fresh measurement every run, from
`ocmr_val` -- confirmed patient-disjoint from every `ocmr_test_gro_*` split -- so
`data.scale_mode=auto` no longer touches test data at all. It works because
`gro.get_mask` is a deterministic function of shape and acceleration only (no
randomness, no dependency on having run the real preprocessing pipeline; see
gro.py), so it can undersample val's fully-sampled k-space the same way the test
split was undersampled once, and measure the same ratio the table used to.
"""

from pathlib import Path

import numpy as np
import torch

from data import find_files, load_clip
from kspace import KTOperator, center_crop, complex_std


def calibrate_scale(
    val_data_dir: Path,
    acceleration: int,
    coil_mode: str,
    device: torch.device,
    limit: int | None = None,
) -> tuple[float, int]:
    """
    Mean `complex_std(zero_filled) / complex_std(reference)` over `ocmr_val`, at
    this `(acceleration, coil_mode)` -- the value `estimate_scale(mode='constant')`
    needs. Returns `(ratio, n_clips)`; `n_clips` is worth logging; if it is small
    (a tight `--calib_limit`, or a thin val split) the ratio is noisier.
    """
    files = find_files(Path(val_data_dir), limit=limit, split='val')
    ratios = []
    for path in files:
        clip = load_clip(
            path, Path(val_data_dir), coil_mode=coil_mode, acceleration=acceleration,
        )
        mask = clip['mask'].to(device)
        sens = clip['sens'].to(device) if clip['sens'] is not None else None
        operator = KTOperator(mask, sens)
        zero_filled = operator.zero_filled(clip['kspace'].to(device))
        target = tuple(clip['reference'].shape[-2:])
        ratio = (
            complex_std(center_crop(zero_filled, target))
            / complex_std(clip['reference'].to(device))
        ).item()
        ratios.append(ratio)
    return float(np.mean(ratios)), len(ratios)
