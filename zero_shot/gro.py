"""
GRO (Golden Ratio offset) retrospective undersampling mask -- on the fly.

Vendored a second time from `preprocessing/preprocess_ocmr.py` (which is itself
vendored from CineVN's `sampling_patterns/dynamic/gro.py`; see CREDITS.md)
rather than imported, because that module pulls in sigpy/ismrmrd at import time
for pieces this file does not need -- it only needs the pure-NumPy mask math.

`get_mask` here is deliberately narrower than preprocessing's: no `mask_type`
(GRO only), no `padding_left`/`padding_right` (the already-preprocessed
`ocmr_val`/`ocmr_train` k-space has no encoding-grid padding left in it -- verified
below), no `rng` (the reference implementation accepts one but never reads it;
the pattern is fully deterministic given `(num_frames, num_cols, acceleration)`).

Verified byte-identical against the `mask` stored in `ocmr_test_gro_{08,12,16,20}`
for multiple subjects/matrix sizes (2026-09-21): calling
`get_mask(kspace.shape[1], kspace.shape[-1], acceleration)` on a test file's own
*masked* k-space shape reproduces its stored mask exactly. That is what makes it
safe to call the same way on `ocmr_val`'s fully-sampled k-space, whose shape has
the same convention -- the mask this project needs was never file-specific to
begin with, only shape- and acceleration-specific.
"""

from math import ceil, sqrt

import numpy as np

# --------------------------------------------------------------------------
# Copyright (c) 2014/2019 - The Ohio State University. All rights reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for educational, research, and not-for-profit purposes,
# without fee and without written agreement, is hereby granted, provided
# that the above copyright notice, the following two paragraphs, and the
# author attribution appear in all copies of this software. For commercial
# licensing possibilities, contact The Office of Technology Commercialization
# Office (http://tco.osu.edu/) at The Ohio State University.
#
# IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE TO ANY PARTY FOR
# DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
# OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE OHIO STATE
# UNIVERSITY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# THE OHIO STATE UNIVERSITY SPECIFICALLY DISCLAIMS ANY WARRANTIES INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
# BASIS, AND THE OHIO STATE UNIVERSITY HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#
# Author: Rizwan Ahmad (ahmad.46@osu.edu). Ported from the original MatLab
# implementation at https://github.com/OSU-CMR/GRO-CAVA, via
# D:\Projects\CineVN-main\src\sampling_patterns\dynamic\gro.py and
# preprocessing/preprocess_ocmr.py.
# --------------------------------------------------------------------------


def _round_away_from_zero(x):
    """Python uses round-half-to-even; MatLab (the GRO reference impl) uses round-away-from-zero."""
    if isinstance(x, np.ndarray):
        mask = x >= 0
        y = np.zeros_like(x)
        y[mask] = np.floor(x[mask] + 0.5)
        y[~mask] = np.ceil(x[~mask] - 0.5)
        return y
    if x >= 0.0:
        return np.floor(x + 0.5)
    return np.ceil(x - 0.5)


class GROParam:
    """Parameters for the GRO (Golden Ratio offset) sampling pattern."""

    def __init__(
        self,
        PE: int = 160,    # size of phase encoding (PE) grid
        FR: int = 64,     # number of frames
        n: int = 12,      # number of samples (readouts) per frame
        E: int = 1,       # number of encodings, E=1 for cine
        tau: float = 1,   # extent of shift between frames (1 or 2: golden ratio shift)
        s: float = 2.2,   # s>=1, larger means higher sampling density in the middle
        alph: float = 3,  # alph>1, larger means sharper transition from high- to low-density regions
        PF: int = 0,      # partial Fourier: discards PF samples from one side
    ):
        self.PE, self.FR, self.n, self.E = PE, FR, n, E
        self.tau, self.s, self.alph, self.PF = tau, s, alph, PF


def gro_sampling_pattern(param: GROParam, offset: float = 0) -> np.ndarray:
    """
    GRO (Golden Ratio offset) sampling pattern.

    Reference:
        Rizwan Ahmad, Ning Jin, Orlando Simonetti, Yingmin Liu, and Adam Rich.
        "Cartesian sampling for dynamic magnetic resonance imaging (MRI)",
        U.S. Patent Application No. 16/984,351 (pub. 2021-02-04).
        https://patents.justia.com/patent/20210033689
    Original MatLab implementation: https://github.com/OSU-CMR/GRO-CAVA
    """
    n, FR, N, E = param.n, param.FR, param.PE, param.E
    tau, PF, s, a = param.tau, param.PF, param.s, param.alph

    gr = (1 + sqrt(5)) / 2  # golden ratio
    gr = 1 / (gr + tau - 1)  # golden angle

    Ns = ceil(N * 1 / s)  # size of shrunk PE grid
    k = (N / 2 - Ns / 2) / ((Ns / 2) ** a)  # location specific displacement

    samp = np.zeros((N, FR, E))  # sampling on k-t grid
    PEInd = np.zeros(((n - PF) * FR, E))
    FRInd = np.zeros(((n - PF) * FR, 1))

    v0 = np.arange(1 / 2 + 1e-10, Ns + 1 / 2 - 1e-10, Ns / (n + PF))
    v0 += offset * Ns / (n + PF)
    for e in range(E):
        v0 = v0 + 1 / E * Ns / (n + PF)
        kk = E + 1 - (e + 1)
        for j in range(FR):
            v = ((v0 + j * Ns / (n + PF) * gr) - 1) % Ns + 1
            v = v - Ns * (v >= Ns + 0.5)

            if N % 2 == 0:
                vC = v - k * np.sign((Ns / 2 + 1 / 2) - v) * np.abs((Ns / 2 + 1 / 2) - v) ** a + (N - Ns) / 2 + 1 / 2
                vC = vC - N * (vC >= N + 0.5)
            else:
                vC = v - k * np.sign((Ns / 2 + 1 / 2) - v) * np.abs((Ns / 2 + 1 / 2) - v) ** a + (N - Ns) / 2
            vC = _round_away_from_zero(np.sort(vC))
            vC = vC[PF:]

            if (j + 1) * n > PEInd.shape[0]:
                PEInd_temp = np.zeros(((j + 1) * n, PEInd.shape[1]))
                PEInd_temp[:PEInd.shape[0], :PEInd.shape[1]] = PEInd
                PEInd = PEInd_temp
            if (j + 1) % 2 == 1:
                PEInd[j * n:(j + 1) * n, e] = vC - 1
            else:
                PEInd[j * n:(j + 1) * n, e] = vC[::-1] - 1
            FRInd[j * n:(j + 1) * n] = j

            samp[vC.astype(int) - 1, j, e] += kk

    return samp


def get_mask(num_frames: int, num_cols: int, acceleration: float) -> np.ndarray:
    """
    The GRO mask for a clip of this shape, as `[frame, ky]` uint8.

    Deterministic: same `(num_frames, num_cols, acceleration)` always gives the
    same mask. See the module docstring for the verification against the stored
    test-split masks.
    """
    num_samples_per_frame = round(num_cols / acceleration)
    param = GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
    samp = gro_sampling_pattern(param, offset=0)
    return samp.T[0].astype(np.uint8)
