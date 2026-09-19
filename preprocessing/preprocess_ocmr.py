"""
Self-contained OCMR preprocessing pipeline for CardioFlow.

Turns raw OCMR ISMRMRD files into per-slice training/eval samples (masked
k-space + RSS/sensitivity-weighted reference reconstructions), the same way
CineVN's own preprocessing does (Vornehm et al., MRM 2025;
D:\\Projects\\CineVN-main). The pieces CardioFlow actually needs are vendored
directly into this one file, so it has no import dependency on that sibling
repo or its `helpers` / `bart` / `sampling_patterns` packages:
  - ISMRMRD reading                              (helpers/datasets/*.py)
  - FFT, coil-combination, POCS reconstruction    (helpers/mri/{fftc,reconstruct,utils}.py)
  - ESPIRiT coil sensitivity estimation (sigpy)   (helpers/mri/coil_sensitivities.py)
  - GRO retrospective undersampling mask          (sampling_patterns/dynamic/{gro,rounding}.py)
  - GIF preview of reconstructions                (helpers/save.py)

Dropped relative to CineVN's own run_preprocessing.py, since this project
doesn't use them:
  - S3 auto-download of missing raw files -- replaced with a hard
    FileNotFoundError. This project works off the already-extracted local
    OCMR copy (see memory: dataset_ocmr_sizing), so a missing file means
    --raw_dir is misconfigured, not a reason to hit the network.
  - The BART/PICS compressed-sensing reconstruction path (`cs_dir`) and the
    `bart` coil-sensitivity backend -- CardioFlow doesn't use CS recon as a
    baseline, only sigpy/ESPIRiT.
  - Retrospective mask types other than GRO (random, vista, cava,
    equispaced, ...) -- the proposal only calls for GRO acceleration masks.

Requires (all already present in vision_env except sigpy):
    pip install sigpy
Expects ocmr.csv (CardioFlow's own copy of CineVN's OCMR metadata) next to
this file. --raw_dir and --target_dir are required (no machine-specific
default baked in). Coil sensitivity maps are cached under
<target_dir>/coil_sens, alongside the ocmr_train/ocmr_val/ocmr_test_* output
folders -- there's no separate --coil_sens_dir any more.

Example:
    python preprocess_ocmr.py --raw_dir G:/OCMR_extracted/OCMR_data --target_dir D:/Projects/CardioFlow/data/processed \
        --csv_query "smp=='fs'" --accelerations 8 12 16 20 --mask_types gro
"""

import argparse
import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Any, Sequence, TypeAlias

try:
    import cupy as cp
except ImportError:  # importing cupy fails if no cuda is available
    cp = None
import h5py
import numpy as np
import pandas as pd
import sigpy
import sigpy.mri
import skimage.exposure
import torch
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed
    import ismrmrd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s|%(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
)


# --------------------------------------------------------------------------
# ndarray helpers (helpers/ndarray.py)
# --------------------------------------------------------------------------

_xp = cp if cp is not None else np
ndarray: TypeAlias = np.ndarray | _xp.ndarray  # type: ignore


def get_ndarray_module(array: ndarray):
    if isinstance(array, np.ndarray):
        return np
    elif cp is not None and isinstance(array, cp.ndarray):
        return cp
    else:
        raise TypeError(f'Array has invalid type {type(array)}')


# --------------------------------------------------------------------------
# FFT utilities (helpers/mri/fftc.py)
# --------------------------------------------------------------------------

def fftnc(image: ndarray, axes: Sequence[int] = (-2, -1)) -> ndarray:
    xp = get_ndarray_module(image)
    kspace = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(image, axes=axes), norm='ortho', axes=axes), axes=axes)
    if image.dtype in [xp.complex64, xp.float32]:
        kspace = kspace.astype(xp.complex64)
    return kspace


def ifftnc(kspace: ndarray, axes: Sequence[int] = (-2, -1)) -> ndarray:
    xp = get_ndarray_module(kspace)
    image = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(kspace, axes=axes), norm='ortho', axes=axes), axes=axes)
    if kspace.dtype in [xp.complex64, xp.float32]:
        image = image.astype(xp.complex64)
    return image


# --------------------------------------------------------------------------
# k-space utilities (helpers/mri/utils.py)
# --------------------------------------------------------------------------

def apply_phase_padding(k_data: ndarray, phase_padding_left: int, phase_padding_right: int) -> ndarray:
    assert k_data.ndim == 9, f'k_data has the wrong number of dimensions, should be 9 but was {k_data.ndim}.'
    if phase_padding_left == 0 and phase_padding_right == 0:
        return k_data

    xp = get_ndarray_module(k_data)
    # padding is defined from first to last dimension; np.pad cannot handle negative values
    padding_plus = ((0, 0), (max(0, phase_padding_left), max(0, phase_padding_right))) + ((0, 0),) * 7
    k_data_padded = xp.pad(k_data, padding_plus)
    padding_minus = slice(max(0, -phase_padding_left), k_data_padded.shape[1] - max(0, -phase_padding_right))
    k_data_padded = k_data_padded[:, padding_minus]
    return k_data_padded


def center_crop(img: ndarray, axis: int, target_size: int) -> ndarray:
    low = img.shape[axis] // 2 - target_size // 2
    high = img.shape[axis] // 2 + (target_size + 1) // 2
    slices = [slice(None)] * img.ndim
    slices[axis] = slice(low, high)
    return img[tuple(slices)]


def crop_readout_oversampling(k_data: ndarray) -> ndarray:
    xp = get_ndarray_module(k_data)

    mask = xp.abs(k_data).sum(axis=0) > 0
    readouts = k_data[:, mask]

    pre_z = xp.where(xp.abs(readouts[:, 0]) > 0)[0][0]

    readouts = ifftnc(readouts, axes=(0,))  # [x, ky, kz=1, coil, frame, ...]
    readouts = center_crop(readouts, 0, readouts.shape[0] // 2)
    readouts = fftnc(readouts, axes=(0,))
    readouts[:pre_z // 2] = 0

    k_data = xp.zeros(shape=(readouts.shape[0],) + mask.shape, dtype=k_data.dtype)
    k_data[:, mask] = readouts
    return k_data


# --------------------------------------------------------------------------
# Reconstruction (helpers/mri/reconstruct.py, minus grappa/pics -- unused here)
# --------------------------------------------------------------------------

def rss(k_data: ndarray, fft_axes: tuple[int, int] = (-2, -1), coil_axis: int = 1, keep_coil_dim: bool = False) -> ndarray:
    xp = get_ndarray_module(k_data)
    coil_imgs = ifftnc(k_data, axes=fft_axes)
    return xp.sqrt(xp.sum(xp.abs(coil_imgs) ** 2, axis=coil_axis, keepdims=keep_coil_dim))


def sensitivity_weighted(
        k_data: ndarray,
        coil_sens: ndarray,
        fft_axes: tuple[int, int] = (-2, -1),
        coil_axis: int = 1,
        frame_axis: int = 2,
        keep_coil_dim: bool = False,
        noise_cov: ndarray | None = None,
) -> ndarray:
    """f = (C^H * Psi^-1 * C)^-1 * C^H * Psi^-1 * ifft(k)"""
    xp = get_ndarray_module(k_data)

    if noise_cov is None:
        noise_cov = xp.identity(k_data.shape[coil_axis])

    if k_data.ndim != coil_sens.ndim:
        coil_sens = xp.expand_dims(coil_sens, frame_axis)

    coil_imgs = ifftnc(k_data, axes=fft_axes)

    coil_imgs = xp.moveaxis(coil_imgs, coil_axis, -1)[..., None]  # m [..., coil, 1]
    coil_sens = xp.moveaxis(coil_sens, coil_axis, -1)[..., None]  # C [..., coil, 1]

    coil_sens_herm = xp.moveaxis(coil_sens.conj(), -1, -2)  # C^H [..., 1, coil]
    noise_cov_inv = xp.linalg.inv(noise_cov)  # Psi^-1 [coil, coil]

    norm = coil_sens_herm @ noise_cov_inv @ coil_sens  # [..., 1, 1]
    img = coil_sens_herm @ noise_cov_inv @ coil_imgs  # [..., 1, 1]
    if cp is not None and isinstance(img, cp.ndarray):
        img = cp.where(norm != 0, img / norm, img)
    else:
        img = np.divide(img, norm, where=(norm != 0), out=np.zeros_like(img))
    img = img[..., 0]
    if keep_coil_dim:
        img = xp.moveaxis(img, -1, coil_axis)
    else:
        img = img[..., 0]
    return img


def pocs(k_data: ndarray, n_iter: int = 10) -> ndarray:
    xp = get_ndarray_module(k_data)

    new_shape = (*k_data.shape[:2], *k_data.shape[3:5], k_data.shape[6])
    k_data = k_data.reshape(new_shape)

    mask = abs(k_data.mean(axis=2, keepdims=True)) > 0
    mask = mask.astype(np.uint8)
    mask = mask.any(axis=3).squeeze()

    sym_mask = mask.any(axis=0, keepdims=True) * mask.any(axis=1, keepdims=True)
    sym_mask *= xp.rot90(sym_mask, 2)
    nonzeros = sym_mask.nonzero()
    sym_kx = nonzeros[0].max() - nonzeros[0].min() + 1
    sym_ky = nonzeros[1].max() - nonzeros[1].min() + 1

    ham = xp.hamming(sym_kx.item())[:, None] * xp.hamming(sym_ky.item())[None, :]
    Wc = xp.zeros_like(sym_mask, dtype=xp.float32)
    Wc[sym_mask] = ham.reshape(-1)

    Ic_phase_exp = xp.exp(1j * xp.angle(ifftnc(Wc[..., None, None, None] * k_data, axes=(0, 1))))

    Sn = k_data  # zero-filled
    for _ in tqdm(range(n_iter), leave=False):
        In = ifftnc(Sn, axes=(0, 1))
        In = xp.abs(In) * Ic_phase_exp
        Sn = fftnc(In, axes=(0, 1))
        Sn[mask] = k_data[mask]

    Sn = Sn[:, :, None, :, :, None, :, None, None]
    return Sn


# --------------------------------------------------------------------------
# GRO retrospective undersampling mask (sampling_patterns/dynamic/{gro,rounding}.py)
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
        return floor(x + 0.5)
    return ceil(x - 0.5)


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


class Sampling:
    def __init__(self, name: str, acceleration: float | None = None, mask_type: str | None = None):
        self.name = name
        self.acceleration = acceleration
        self.mask_type = mask_type

    def __repr__(self):
        return f'Sampling("{self.name}", acceleration={self.acceleration}, mask_type={self.mask_type})'


def get_mask(
        num_frames: int,
        num_cols: int,
        mask_type: str,
        acceleration: float,
        padding_left: int = 0,
        padding_right: int = 0,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:
    mask_type = mask_type.lower()
    if mask_type != 'gro':
        raise NotImplementedError(
            f'Mask type {mask_type!r} is not vendored in this self-contained script (only GRO is, per project '
            f'scope). Add it from D:\\Projects\\CineVN-main\\src\\helpers\\mri\\subsampling.py if needed.'
        )

    center_idx_padded = (num_cols + padding_left + padding_right) // 2
    center_idx_unpadded = num_cols // 2
    if center_idx_padded - padding_left != center_idx_unpadded:
        raise NotImplementedError('GRO does not support off-center masking caused by uneven padding')

    num_samples_per_frame = round(num_cols / acceleration)
    param = GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
    samp = gro_sampling_pattern(param, offset=0)
    mask = samp.T[0]

    return mask.astype(np.uint8)


# --------------------------------------------------------------------------
# ESPIRiT coil sensitivity estimation via sigpy (helpers/mri/coil_sensitivities.py)
# --------------------------------------------------------------------------

def coil_sens_read(fname: Path, nmaps: int = -1) -> np.ndarray:
    with h5py.File(fname, 'r') as hf:
        coil_sens = np.array(hf['coil_sens'])  # [slice, map, coil, frame=1, x, y]
    if nmaps > 0:
        coil_sens = coil_sens[:, :nmaps]
    coil_sens = np.transpose(coil_sens, [4, 5, 2, 1, 3, 0])  # [x, y, coil, map, frame, slice]
    return coil_sens


def coil_sens_estimate(
        k_data: ndarray,  # [kx, ky, coil, frame, slice]
        fname: Path | None = None,
        nmaps: int = 1,
        calib_lines: int = 16,
        clip: bool = False,
        write_h5: bool = True,
) -> ndarray:
    if nmaps > 1:
        raise NotImplementedError(
            f'Sigpy only supports estimating one set of coil sensitivity maps, but {nmaps} were requested.'
        )
    xp = get_ndarray_module(k_data)

    # average over time
    mask = xp.abs(k_data) > 0
    k_data = xp.sum(k_data, axis=3) / (xp.sum(mask, axis=3) + xp.finfo(float).eps)

    crop = 0.8 if clip else 0  # 0.8 is ESPIRiT's/BART's default
    ksp = xp.transpose(k_data, [2, 0, 1, 3])  # [coil, kx, ky, slice]
    device = sigpy.Device(0) if (cp is not None and isinstance(k_data, cp.ndarray)) else sigpy.Device(-1)

    app = sigpy.mri.app.EspiritCalib(ksp, calib_width=calib_lines, crop=crop, device=device, show_pbar=False)
    coil_sens = app.run()  # [coil, x, y, slice]
    coil_sens = xp.transpose(coil_sens, [1, 2, 0, 3])  # type: ignore  # [x, y, coil, slice]
    coil_sens = coil_sens[:, :, :, None, None, :]  # [x, y, coil, map=1, frame=1, slice]

    if write_h5:
        assert fname is not None, '`fname` must be given if `write_h5` is True'
        fname.parent.mkdir(parents=True, exist_ok=True)
        coil_sens_ = xp.transpose(coil_sens, [5, 3, 2, 4, 0, 1])  # [slice, map, coil, frame=1, x, y]
        with h5py.File(fname, 'w') as hf:
            if cp is not None:
                coil_sens_ = cp.asnumpy(coil_sens_)
            hf.create_dataset('coil_sens', data=coil_sens_.astype(np.complex64))

    return coil_sens


# --------------------------------------------------------------------------
# GIF preview of a reconstruction (helpers/save.py)
# --------------------------------------------------------------------------

def save_movie(
        image: np.ndarray | torch.Tensor, fname: Path | str, clip: bool = False, equalize_histogram: bool = False,
        tres: float = 50, vmin: float | None = None, vmax: float | None = None,
):
    if clip and equalize_histogram:
        warnings.warn('Both clip and equalize_histogram are set to True. This is not recommended.')

    image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image

    if image_np.shape[-1] == 2:
        image_np = image_np[..., 0] + 1j * image_np[..., 1]
    if np.iscomplexobj(image_np):
        image_np = np.abs(image_np)

    if clip:
        image_np = np.clip(image_np, np.percentile(image_np, 3), np.percentile(image_np, 97))
    vmin_ = vmin or np.min(image_np)
    vmax_ = vmax or np.max(image_np)
    image_np = (image_np - vmin_) / (vmax_ - vmin_)
    if equalize_histogram:
        image_np = skimage.exposure.equalize_adapthist(image_np, clip_limit=0.02)
    image_np = (image_np * 255).astype(np.uint8)

    fname = Path(fname)
    images_pil = [Image.fromarray(img) for img in image_np]
    images_pil[0].save(fname.with_suffix('.gif'), save_all=True, append_images=images_pil[1:], duration=tres, loop=0)


# --------------------------------------------------------------------------
# ISMRMRD dataset reading (helpers/datasets/{dataset,ismrmrd}.py)
# --------------------------------------------------------------------------

class Dataset(ABC):
    split: str
    hdr: ismrmrd.xsd.ismrmrdHeader
    ecg: np.ndarray | None
    _slice_idx: int | None
    k_data_full: np.ndarray
    k_data: ndarray
    has_read_os: bool

    def __init__(self, name: str, filename: Path, device: str, rep_to_frame_dim: bool = False):
        self.name = name
        self.filename = filename
        self._device = device
        self.rep_to_frame_dim = rep_to_frame_dim

        self._read_dir: dict[int, np.ndarray] = {}
        self._phase_dir: dict[int, np.ndarray] = {}
        self._slice_dir: dict[int, np.ndarray] = {}
        self._position: dict[int, np.ndarray] = {}

        self.norm_orientation_rot: int | None = None
        self.norm_orientation_hor: int | None = None
        self.norm_orientation_ver: int | None = None

        self.noise_cov: np.ndarray | None = None

    def read_meta(self):
        self.read_header()
        self.read_ecg()

    @abstractmethod
    def read_header(self):
        raise NotImplementedError

    @abstractmethod
    def read_ecg(self):
        raise NotImplementedError

    def read_kdata(self, whiten: bool = True, select_slice: int | None = None):
        self._read_kdata(select_slice=select_slice)
        if self.rep_to_frame_dim:
            self._move_rep_to_frame_dim()
        self._slice_idx = select_slice
        self.device = self._device
        if whiten:
            self._whiten_kdata()

    @abstractmethod
    def _read_kdata(self, **kwargs):
        raise NotImplementedError

    def _move_rep_to_frame_dim(self):
        self.k_data = self._rep2frame(self.k_data)
        self.k_data_full = self._rep2frame(self.k_data_full)

    @staticmethod
    def _rep2frame(k_data: np.ndarray) -> np.ndarray:
        k_data = np.moveaxis(k_data, 7, 4)  # [kx, ky, kz, coil, rep, frame, set, slice, avg]
        k_data = np.expand_dims(k_data, 8)  # [kx, ky, kz, coil, rep, frame, set, slice, 1, avg]
        new_shape = (*k_data.shape[:4], -1, *k_data.shape[6:])
        return k_data.reshape(*new_shape)  # [kx, ky, kz, coil, rep*frame, set, slice, 1, avg]

    def _whiten_kdata(self):
        if self.noise_cov is None:
            return
        xp = get_ndarray_module(self.k_data)
        dtype = self.k_data.dtype

        if cp is not None:
            k_data_temp = cp.asnumpy(self.k_data)
            noise_cov_temp = cp.asnumpy(self.noise_cov)
        else:
            k_data_temp = self.k_data
            noise_cov_temp = self.noise_cov

        try:
            k_data_temp = np.moveaxis(k_data_temp, 3, 0)  # [coil, kx, ky, kz, ...]
            norm = np.linalg.norm(k_data_temp)
            k_data_temp = sigpy.mri.whiten(k_data_temp, noise_cov_temp)
            k_data_temp = k_data_temp / np.linalg.norm(k_data_temp) * norm  # type: ignore  # re-normalize
            k_data_temp = k_data_temp.astype(dtype)
            k_data_temp = np.moveaxis(k_data_temp, 0, 3)  # [kx, ky, kz, coil, ...]
            self.k_data = xp.asarray(k_data_temp)
        except np.linalg.LinAlgError:
            logging.warning('Whitening failed because covariance matrix is not positive definite, skipping')
        except Exception as e:
            logging.warning(f'Whitening failed due to unknown error ({e}), skipping')

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device):
        if 'cuda' in device:
            assert cp is not None, 'Setting device to `cuda` failed because cupy could not be imported'
            self.k_data = cp.asarray(self.k_data)
        elif device == 'cpu':
            if cp is not None:
                self.k_data = cp.asnumpy(self.k_data)
        else:
            raise RuntimeError(f'Unknown device {device}')
        self._device = device

    @property
    def n_slices(self) -> int:
        try:
            return self.hdr.encoding[0].encodingLimits.slice.maximum + 1  # type: ignore
        except AttributeError:
            raise RuntimeError('Number of slices could not be determined from ISMRMRD header')

    @property
    def slice_idx(self) -> int | None:
        return self._slice_idx

    def select_slice(self, slice_idx: int):
        try:
            self.k_data = self.k_data_full[:, :, :, :, :, :, slice_idx, None]
            self._slice_idx = slice_idx
            self.device = self._device
        except IndexError as e:
            raise IndexError(
                f'Slice index {slice_idx} out of range. The selected slice was probably not loaded. '
                f'Please select it using `read_kdata(select_slice=slice_idx)`'
            ) from e

    def get_read_dir(self, slice_idx: int | None) -> np.ndarray | None:
        if slice_idx in self._read_dir:
            return self._read_dir[slice_idx]
        elif slice_idx is None:
            return self._read_dir[0]
        return None

    @property
    def read_dir(self) -> np.ndarray | None:
        return self.get_read_dir(self.slice_idx)

    def get_phase_dir(self, slice_idx: int | None) -> np.ndarray | None:
        if slice_idx in self._phase_dir:
            return self._phase_dir[slice_idx]
        elif slice_idx is None:
            return self._phase_dir[0]
        return None

    @property
    def phase_dir(self) -> np.ndarray | None:
        return self.get_phase_dir(self.slice_idx)

    def get_slice_dir(self, slice_idx: int | None) -> np.ndarray | None:
        if slice_idx in self._slice_dir:
            return self._slice_dir[slice_idx]
        elif slice_idx is None:
            return self._slice_dir[0]
        return None

    @property
    def slice_dir(self) -> np.ndarray | None:
        return self.get_slice_dir(self.slice_idx)

    def get_position(self, slice_idx: int | None) -> np.ndarray | None:
        if slice_idx in self._position:
            return self._position[slice_idx]
        elif slice_idx is None:
            return self._position[0]
        return None

    @property
    def position(self) -> np.ndarray | None:
        return self.get_position(self.slice_idx)

    def set_norm_orientation(self, rot: int, hor: int, ver: int):
        self.norm_orientation_rot = rot
        self.norm_orientation_hor = hor
        self.norm_orientation_ver = ver

    def norm_orientation(self, img: ndarray, image_axes: tuple[int, int]) -> np.ndarray:
        xp = get_ndarray_module(img)
        if self.norm_orientation_rot is not None and self.norm_orientation_rot > 0:
            img = xp.rot90(img, self.norm_orientation_rot // 90, axes=image_axes)
        if self.norm_orientation_hor:
            img = xp.flip(img, axis=image_axes[1])
        if self.norm_orientation_ver:
            img = xp.flip(img, axis=image_axes[0])
        return img

    def estimate_noise(self, return_noise_data: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.k_data is None:
            raise RuntimeError('Noise can only be estimated after k-space data is read')

        noise_data = np.concatenate([
            self.k_data[:min(16, round(self.k_data.shape[0] / 8))],
            self.k_data[-min(16, round(self.k_data.shape[0] / 8)):]
        ], axis=0)
        noise_data = np.concatenate([
            noise_data[:, :round(self.k_data.shape[1] / 5)],
            noise_data[:, -round(self.k_data.shape[1] / 5):]
        ], axis=1)
        noise_data = np.concatenate([
            noise_data[:, :, :round(self.k_data.shape[2] / 5)],
            noise_data[:, :, -round(self.k_data.shape[2] / 5):]
        ], axis=2)

        noise_data = np.moveaxis(noise_data, 3, 0)
        noise_data = np.reshape(noise_data, (noise_data.shape[0], -1))
        noise_std = np.std(noise_data, axis=1, where=noise_data != 0)  # type: ignore

        if return_noise_data:
            noise_data = np.stack([n[np.abs(n) > 0] for n in noise_data], axis=0)
            return noise_std, noise_data
        return noise_std


class IsmrmrdDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_read_os = True
        self.dset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=False)

    def read_header(self):
        self.hdr = ismrmrd.xsd.CreateFromDocument(self.dset.read_xml_header())

    def read_ecg(self):
        ecg = []
        try:
            num_waveforms = self.dset.number_of_waveforms()
        except LookupError:
            num_waveforms = 0
        ecg_sample_time = None
        for i in range(num_waveforms):  # type: ignore
            w = self.dset.read_waveform(i)
            head = w.getHead()
            if head.waveform_id in [5, 8] and head.channels == 5:  # ecg signal
                if ecg_sample_time is None:
                    ecg_sample_time = head.sample_time_us
                if ecg_sample_time != head.sample_time_us:
                    logging.warning('aborting ECG signal extraction due to inconsistent sample time')
                    break
                ecg.append(w.data)
        self.ecg = np.concatenate(ecg, axis=1) if len(ecg) > 0 else np.array(0)

    def _read_kdata(self, select_slice: int | None = None):
        enc = self.hdr.encoding[0]
        n_x = enc.encodedSpace.matrixSize.x  # type: ignore
        n_y = enc.encodingLimits.kspace_encoding_step_1.maximum + 1  # no zero padding along ky  # type: ignore
        n_z = enc.encodedSpace.matrixSize.z  # type: ignore
        n_coils = self.dset.read_acquisition(0).data.shape[0]
        n_slices = enc.encodingLimits.slice.maximum + 1  # type: ignore
        n_reps = enc.encodingLimits.repetition.maximum + 1  # type: ignore
        n_phases = enc.encodingLimits.phase.maximum + 1  # type: ignore
        n_sets = enc.encodingLimits.set.maximum + 1  # type: ignore
        n_average = enc.encodingLimits.average.maximum + 1  # type: ignore

        first_acq = 0

        # look for noise scans
        noise_data = []
        noise_dwelltime_us = -1
        while True:
            acq = self.dset.read_acquisition(first_acq)
            head = acq.getHead()
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noise_data.append(acq.data)
                if noise_dwelltime_us == -1:
                    noise_dwelltime_us = head.sample_time_us
                assert noise_dwelltime_us == head.sample_time_us, 'sample_time_us is inconsistent'
                first_acq += 1
            else:
                break
        noise_data = np.array(noise_data)

        # asymmetric echo
        kx_pre = 0
        acq = self.dset.read_acquisition(first_acq)
        head = acq.getHead()
        if head.center_sample * 2 < n_x:
            kx_pre = n_x - head.number_of_samples

        acquisition_dwelltime_us = -1
        if select_slice is None:
            k_data = np.zeros((n_x, n_y, n_z, n_coils, n_phases, n_sets, n_slices, n_reps, n_average), dtype=np.complex64)
        else:
            k_data = np.zeros((n_x, n_y, n_z, n_coils, n_phases, n_sets, 1, n_reps, n_average), dtype=np.complex64)
        for i in tqdm(range(first_acq, self.dset.number_of_acquisitions()), leave=False):  # type: ignore
            acq = self.dset.read_acquisition(i)
            head = acq.getHead()

            if acquisition_dwelltime_us == -1:
                acquisition_dwelltime_us = head.sample_time_us
            assert acquisition_dwelltime_us == head.sample_time_us, 'sample_time_us is inconsistent'

            acq_idx: ismrmrd.EncodingCounters = acq.idx  # type: ignore
            y_idx = acq_idx.kspace_encode_step_1
            z_idx = acq_idx.kspace_encode_step_2
            phase_idx = acq_idx.phase
            set_idx = acq_idx.set
            slice_idx = acq_idx.slice
            rep_idx = acq_idx.repetition
            avg_idx = acq_idx.average

            if acq.isFlagSet(ismrmrd.ACQ_IS_REVERSE):
                raise NotImplementedError('Acquisition is reversed. Flip it before proceeding!')

            if select_slice is None:
                k_data[kx_pre:, y_idx, z_idx, :, phase_idx, set_idx, slice_idx, rep_idx, avg_idx] = np.transpose(acq.data)
            elif slice_idx == select_slice:
                k_data[kx_pre:, y_idx, z_idx, :, phase_idx, set_idx, 0, rep_idx, avg_idx] = np.transpose(acq.data)
            else:
                continue

            read_dir = np.array(head.read_dir)
            read_dir_prev = self.get_read_dir(slice_idx)
            if read_dir_prev is None:
                self._read_dir[slice_idx] = read_dir
            elif not np.allclose(read_dir, read_dir_prev):
                raise ValueError('read_dir is inconsistent')

            phase_dir = np.array(head.phase_dir)
            phase_dir_prev = self.get_phase_dir(slice_idx)
            if phase_dir_prev is None:
                self._phase_dir[slice_idx] = phase_dir
            elif not np.allclose(phase_dir, phase_dir_prev):
                raise ValueError('phase_dir is inconsistent')

            slice_dir = np.array(head.slice_dir)
            slice_dir_prev = self.get_slice_dir(slice_idx)
            if slice_dir_prev is None:
                self._slice_dir[slice_idx] = slice_dir
            elif not np.allclose(slice_dir, slice_dir_prev):
                raise ValueError('slice_dir is inconsistent')

            position = np.array(head.position)
            position_prev = self.get_position(slice_idx)
            if position_prev is None:
                self._position[slice_idx] = position
            elif not np.allclose(position, position_prev):
                raise ValueError('position is inconsistent')

        # discard pilot tone signal, if present
        if any(param.name == 'PilotTone' and param.value == 1 for param in self.hdr.userParameters.userParameterLong):  # type: ignore
            logging.info('Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line')
            k_data[kx_pre:kx_pre + 3] = 0
            k_data[-1] = 0

        # discard all but first average (don't use mean of averages, to preserve SNR of a single scan)
        self.k_data_full = k_data[..., 0, None]
        self.k_data = self.k_data_full

        if noise_data.size == 0:
            _, noise_data = self.estimate_noise(True)
        else:
            noise_data *= np.sqrt(noise_dwelltime_us / acquisition_dwelltime_us)
            noise_data = np.moveaxis(noise_data, 0, 1)
            noise_data = np.reshape(noise_data, (noise_data.shape[0], -1))

        self.noise_cov = sigpy.mri.get_cov(noise_data)


# --------------------------------------------------------------------------
# Preprocessing driver (helpers/preprocessing.py)
# --------------------------------------------------------------------------

class Preprocessing:
    REQ_CSV_COLS = [
        'file name', 'scn', 'smp', 'ech', 'dur', 'viw', 'sli', 'fov', 'sub', 'slices',  # OCMR columns
        'excluded slices',  # slices to exclude from processing (e.g. '0-2+4' for slices 0, 1, 2, 4)
        'mask type',  # mask
        'FOV x', 'FOV y', 'FOV z',  # field of view
        'frames', 'coils', 'averages', 'kx', 'ky acq', 'ky enc', 'ky rec',  # kspace dimensionality
        'TRes',  # temporal resolution
        'HR',  # heart rate
        'protocol name',  # protocol name
        'split',  # train/val/test set
        'viw 2',  # view (additional column for 2ch/3ch/4ch distinction)
        'rotation', 'flip horizontal', 'flip vertical',  # operations to be applied to fix image orientation
        'center x', 'center y', 'axis 1 x', 'axis 1 y', 'axis 2 x', 'axis 2 y',  # profiles
        'bbox x low', 'bbox y low', 'bbox x high', 'bbox y high',  # ROI bounding box
    ]

    def __init__(
            self,
            csv_file: Path,
            raw_dir: Path,
            target_dir: Path,
            csv_query: str | None = None,
            csv_idx_range: list[int | None] = [0, None],
            accelerations: list[int] | None = None,
            mask_types: list[str] | None = None,
            n_maps: int = 1,
            try_cuda: bool = True,
            seed: int = 42,
            persist_csv_updates: bool = True,
    ):
        """
        Args:
            csv_file: path to CardioFlow's ocmr.csv (metadata of samples to process)
            raw_dir: path to raw OCMR files, laid out as `raw_dir/ocmr/<file>.h5`
            target_dir: path to store processed data. Coil sensitivity maps
                are cached under `target_dir/coil_sens`, alongside the
                `ocmr_train`/`ocmr_val`/`ocmr_test_*` folders -- not a
                separately-configurable location.
            csv_query: pandas query to filter samples (see https://datagy.io/pandas-query/)
            csv_idx_range: [start] or [start, stop] row-index range to filter samples
            accelerations: acceleration rates for retrospective (GRO) undersampling
            mask_types: only 'gro' is vendored in this script
            n_maps: number of coil sensitivity maps to estimate (sigpy only supports 1)
            try_cuda: use GPU if available
            seed: RNG seed
            persist_csv_updates: write per-file header info back to csv_file
                as each file is processed. Set False when running several
                preprocessing processes in parallel against the same
                csv_file (each sharded via csv_idx_range) to avoid the
                processes clobbering each other's writes -- see
                `update_csv`'s docstring.
        """
        assert csv_file.is_file(), f'{csv_file} is not a file.'
        self.csv_file = csv_file
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.coil_sens_dir = target_dir / 'coil_sens'
        self.persist_csv_updates = persist_csv_updates
        self.accelerations = [] if accelerations is None else accelerations
        self.mask_types = [] if mask_types is None else mask_types
        self.n_maps = n_maps
        self.device = 'cpu'
        if try_cuda and cp is not None and cp.cuda.is_available():
            self.device = 'cuda'
        self.rng = np.random.RandomState(seed)

        self.df_full = pd.read_csv(csv_file, dtype={'excluded slices': str})
        missing_cols = [c for c in self.REQ_CSV_COLS if c not in self.df_full.columns]
        assert len(missing_cols) == 0, f'Dataframe misses the following cols: {missing_cols}'

        self.df = self.df_full.loc[csv_idx_range[0]:csv_idx_range[1]]
        if csv_query:
            self.df = self.df.query(csv_query)

    @staticmethod
    def update_csv(df: pd.DataFrame, dset: Dataset, csv_file: Path, persist: bool = True) -> dict[str, Any]:
        """
        Fills in per-file columns (frames, coils, FOV, ...) read from this
        file's own header. `persist=False` skips the `df.to_csv` write and
        only returns the updated row as a dict -- required when running
        multiple preprocessing processes in parallel against the same
        `csv_file`, since each process holds its own in-memory copy of the
        full dataframe and an unconditional write here would race: whichever
        process writes last clobbers every other process's updates for rows
        it isn't currently touching. The values being filled in are already
        correct in the upstream ocmr.csv (verified against the raw headers),
        so skipping the on-disk update costs nothing when running sharded.
        """
        row = df['file name'] == dset.filename.name
        assert sum(row) == 1

        hdr = dset.hdr
        enc = hdr.encoding[0]
        t_res = hdr.sequenceParameters.TR[0]  # type: ignore
        n_frames = enc.encodingLimits.phase.maximum + 1  # type: ignore

        df.loc[row, 'FOV x'] = enc.encodedSpace.fieldOfView_mm.x  # type: ignore
        df.loc[row, 'FOV y'] = enc.encodedSpace.fieldOfView_mm.y  # type: ignore
        df.loc[row, 'FOV z'] = enc.encodedSpace.fieldOfView_mm.z  # type: ignore
        df.loc[row, 'frames'] = n_frames
        df.loc[row, 'coils'] = hdr.acquisitionSystemInformation.receiverChannels  # type: ignore
        df.loc[row, 'averages'] = enc.encodingLimits.average.maximum + 1  # type: ignore
        df.loc[row, 'kx'] = enc.encodedSpace.matrixSize.x  # type: ignore
        df.loc[row, 'ky acq'] = enc.encodingLimits.kspace_encoding_step_1.maximum + 1  # type: ignore
        df.loc[row, 'ky enc'] = enc.encodedSpace.matrixSize.y  # type: ignore
        df.loc[row, 'ky rec'] = enc.reconSpace.matrixSize.y  # type: ignore
        df.loc[row, 'TRes'] = round(t_res, 2)
        if df.loc[row, 'smp'].item() in ['fs']:
            df.loc[row, 'HR'] = round(60 / (t_res * n_frames / 1000))
        df.loc[row, 'protocol name'] = hdr.measurementInformation.protocolName  # type: ignore
        if persist:
            df.to_csv(csv_file, index=False)

        return df.loc[row].iloc[0].to_dict()

    def load_dataset(self, raw_file: Path, data_attrs: dict[str, Any]) -> tuple[Dataset, dict[str, Any]]:
        """
        Unlike CineVN's own `load_dataset`, this never downloads a missing
        file from S3 -- it fails loudly instead, since CardioFlow always
        works off the already-extracted local OCMR copy.
        """
        req_keys = ('split', 'rotation', 'flip horizontal', 'flip vertical', 'dur', 'frames', 'sli')
        for key in req_keys:
            assert key in data_attrs, f'{key} not in data_attrs'

        if not raw_file.exists():
            raise FileNotFoundError(
                f'{raw_file} not found locally. This project relies on the pre-extracted OCMR dataset and does '
                f'not download files automatically -- check that --raw_dir points directly at the folder '
                f'containing the extracted .h5 files (e.g. G:\\OCMR_extracted\\OCMR_data).'
            )

        logging.info(f'Loading {raw_file}')
        dset = IsmrmrdDataset(name=raw_file.name, filename=raw_file, device=self.device)

        dset.split = str(data_attrs['split'])

        if all(pd.notna(data_attrs[k]) for k in ['rotation', 'flip horizontal', 'flip vertical']):
            dset.set_norm_orientation(
                rot=int(data_attrs['rotation']),
                hor=int(data_attrs['flip horizontal']),
                ver=int(data_attrs['flip vertical']),
            )

        logging.info('Reading metadata')
        dset.read_meta()

        logging.info(f'Updating {self.csv_file}')
        data_attrs = self.update_csv(self.df_full, dset, self.csv_file, persist=self.persist_csv_updates)

        if data_attrs['dur'] == 'shr' and float(data_attrs['frames']) < 50:
            logging.info('Reading k-space data')
            dset.read_kdata(whiten=True, select_slice=None)

        return dset, data_attrs

    def get_sampling_patterns(self, acq_sampling: str, acq_mask_type: str, exclude: bool) -> list[Sampling]:
        if acq_sampling.lower() == 'fs':
            sampling_patterns = [Sampling(name='fs', acceleration=1)]
            if not exclude:
                for mask_type, acc in itertools.product(self.mask_types, self.accelerations):
                    sampling_patterns.append(Sampling(name=f'{mask_type}_{acc:02d}', acceleration=acc, mask_type=mask_type))
        elif acq_sampling.lower() == 'pse':
            sampling_patterns = [Sampling(name='pse', mask_type=acq_mask_type)]
        else:
            raise ValueError(f'Unknown acq_sampling: {acq_sampling}')

        logging.info(f'Sampling Patterns: {[s.name for s in sampling_patterns]}')
        return sampling_patterns

    @staticmethod
    def phase_pad_dataset(dset: Dataset) -> tuple[Dataset, int, int]:
        enc = dset.hdr.encoding[0]
        enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
        enc_mat_y: int = enc.encodedSpace.matrixSize.y  # type: ignore
        enc_fov_x: float = enc.encodedSpace.fieldOfView_mm.x  # type: ignore
        enc_fov_y: float = enc.encodedSpace.fieldOfView_mm.y  # type: ignore
        if round((enc_fov_x / enc_fov_y) / (enc_mat_x / enc_mat_y)) == 2:
            enc_mat_y //= 2  # necessary for some OCMR Free.Max datasets
        enc_lim_center = enc.encodingLimits.kspace_encoding_step_1.center  # type: ignore
        enc_lim_max = enc.encodingLimits.kspace_encoding_step_1.maximum  # type: ignore
        phase_padding_left = enc_mat_y // 2 - enc_lim_center
        phase_padding_right = enc_mat_y - phase_padding_left - enc_lim_max - 1

        logging.info(f'Phase padding: {phase_padding_left}, {phase_padding_right}')
        if phase_padding_left < 0 or phase_padding_right < 0:
            logging.warning(
                'Phase padding is negative (i.e., phase resolution > 100%)! The additional k-space lines will be '
                'removed and ignored in the following.'
            )

        dset.k_data = apply_phase_padding(dset.k_data, phase_padding_left, phase_padding_right)
        return dset, phase_padding_left, phase_padding_right

    @staticmethod
    def get_recon_phase_size(dset: Dataset) -> int:
        enc = dset.hdr.encoding[0]
        enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
        rec_mat_x: int = enc.reconSpace.matrixSize.x  # type: ignore
        rec_mat_y: int = enc.reconSpace.matrixSize.y  # type: ignore
        if rec_mat_x == enc_mat_x:  # 2D interpolation on
            return rec_mat_y // 2
        return rec_mat_y  # 2D interpolation off

    def mask_kspace(self, k_data: ndarray, sampling: Sampling, pad_left: int, pad_right: int) -> tuple[ndarray, ndarray]:
        xp = get_ndarray_module(k_data)

        if sampling.name == 'fs':  # fully sampled
            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=xp.uint8)  # [ky, frame]
            kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

        elif sampling.name == 'pse':  # prospectively undersampled
            mask = (abs(xp.mean(xp.abs(k_data), axis=2)) > 0).astype(xp.uint8)  # [kx, ky, frame, slice]
            assert xp.all(mask.all(axis=3) | (1 - mask).all(axis=3)), 'Mask is not equal along slice dim'
            mask = mask[mask.shape[0] // 2, :, :, 0]  # [ky, frame]

            mask_unpadded = mask[max(0, pad_left): mask.shape[0] - max(0, pad_right), :]
            sampling.acceleration = int(mask_unpadded.size / mask_unpadded.sum())
            logging.info(f'Acceleration rate: {sampling.acceleration:.2f}')

            mask = xp.ones_like(mask)
            mask[max(0, pad_left): mask.shape[0] - max(0, pad_right), :] = mask_unpadded
            kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

        else:  # retrospective undersampling (GRO)
            num_cols = k_data.shape[1] - max(0, pad_right) - max(0, pad_left)
            assert sampling.mask_type is not None and sampling.acceleration is not None
            mask_unpadded = get_mask(  # [frame, ky]
                k_data.shape[3], num_cols, sampling.mask_type, sampling.acceleration, padding_left=pad_left,
                padding_right=pad_right, rng=self.rng,
            )
            if cp is not None and isinstance(k_data, cp.ndarray):
                mask_unpadded = cp.asarray(mask_unpadded)
            mask_unpadded = mask_unpadded.T  # [ky, frame]

            acc_eff = mask_unpadded.size / mask_unpadded.sum()
            logging.info(f'Effective Acceleration Rate: {acc_eff:5.2f}')

            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=mask_unpadded.dtype)  # [ky, frame]
            mask[max(0, pad_left): mask.shape[0] - max(0, pad_right), :] = mask_unpadded
            kdata_masked = k_data * mask[None, :, None, :, None]  # [kx, ky, coil, frame, slice]

        return kdata_masked, mask

    def get_coil_sensitivities(self, kdata_masked: ndarray, output_name: str) -> ndarray:
        coil_sens_fname = self.coil_sens_dir / output_name / 'coil_sens_avg.h5'

        try:
            coil_sens = coil_sens_read(coil_sens_fname, nmaps=self.n_maps)  # [x, y, coil, map, frame, slice]
            if cp is not None and isinstance(kdata_masked, cp.ndarray):
                coil_sens = cp.asarray(coil_sens)
            logging.info('Loaded previously computed coil sensitivity maps')
        except FileNotFoundError:
            logging.info('Estimating coil sensitivity maps using sigpy')
            coil_sens = coil_sens_estimate(kdata_masked, fname=coil_sens_fname, nmaps=self.n_maps)

        return coil_sens

    def save_data(
            self,
            output_name: str,
            dset: Dataset,
            sampling: Sampling,
            kdata_masked: ndarray,
            mask: ndarray,
            recons: dict[str, ndarray],
            data_attrs: dict[str, Any],
            slice_idx: int,
    ):
        partition = f'test_{sampling.name}' if dset.split == 'test' else f'{dset.split}'
        out_file = self.target_dir / f'ocmr_{partition}' / f'{output_name}.h5'
        out_file.parent.mkdir(exist_ok=True, parents=True)

        logging.info(f'Writing file {out_file}')

        with h5py.File(out_file, 'w') as hf:
            kspace = kdata_masked.astype(np.complex64)
            if cp is not None:
                kspace = cp.asnumpy(kspace)
            hf.create_dataset('kspace', data=kspace)

            for key, recon in recons.items():
                if key not in ['rss', 'weighted']:
                    continue
                dtype = np.complex64 if np.iscomplexobj(recon) else np.float32
                recon = recon.astype(dtype)
                if cp is not None:
                    recon = cp.asnumpy(recon)
                hf.create_dataset(f'reconstruction_{key}', data=recon)

            if dset.split == 'test':
                mask_ = mask.astype(np.uint8)
                if cp is not None:
                    mask_ = cp.asnumpy(mask_)
                hf.create_dataset('mask', data=mask_)
                hf.attrs['acceleration'] = sampling.acceleration
                hf.attrs['mask_type'] = sampling.mask_type if sampling.mask_type else data_attrs['mask type']

            if 'weighted' in recons:
                hf.attrs['norm'] = np.linalg.norm(np.abs(recons['weighted'])).item()
                hf.attrs['abs_max'] = np.max(np.abs(recons['weighted'])).item()
            hf.attrs['patient_id'] = dset.filename.stem
            hf.attrs['view'] = data_attrs['viw']
            hf.attrs['slice'] = slice_idx
            if dset.noise_cov is not None:
                hf.attrs['noise'] = dset.noise_cov

            hf.create_dataset('ismrmrd_header', data=dset.hdr.toXML('utf-8'))  # type: ignore
            if dset.ecg is not None:
                hf.create_dataset('ecg', data=dset.ecg)

            if 'center x' in data_attrs and not np.isnan(data_attrs['center x']):
                hf.attrs['center'] = (int(data_attrs['center x']), int(data_attrs['center y']))
                hf.attrs['axis1'] = (int(data_attrs['axis 1 x']), int(data_attrs['axis 1 y']))
                hf.attrs['axis2'] = (int(data_attrs['axis 2 x']), int(data_attrs['axis 2 y']))
                hf.attrs['bbox'] = (int(data_attrs['bbox x low']), int(data_attrs['bbox y low']),
                                    int(data_attrs['bbox x high']), int(data_attrs['bbox y high']))

            if dset.read_dir is not None:
                hf.attrs['read_dir'] = dset.read_dir
            if dset.phase_dir is not None:
                hf.attrs['phase_dir'] = dset.phase_dir
            if dset.slice_dir is not None:
                hf.attrs['slice_dir'] = dset.slice_dir
            if dset.position is not None:
                hf.attrs['position'] = dset.position

            if dset.norm_orientation_rot is not None:
                hf.attrs['rotation'] = dset.norm_orientation_rot
            if dset.norm_orientation_hor is not None:
                hf.attrs['flip_horizontal'] = dset.norm_orientation_hor
            if dset.norm_orientation_ver is not None:
                hf.attrs['flip_vertical'] = dset.norm_orientation_ver

    def process_slice_with_sampling(
            self,
            dset: Dataset,
            data_attrs: dict[str, Any],
            sampling: Sampling,
            output_name: str,
            exclude: bool,
            pad_left: int,
            pad_right: int,
            recon_phase_size: int,
    ):
        kdata_masked, mask = self.mask_kspace(dset.k_data, sampling, pad_left, pad_right)
        coil_sens = self.get_coil_sensitivities(kdata_masked, output_name)

        recons = {}
        if sampling.name != 'pse':
            logging.info('Computing reference reconstructions')
            recons['rss'] = rss(dset.k_data, fft_axes=(0, 1), coil_axis=2)  # [x, y, frame, slice]
            recons['weighted'] = sensitivity_weighted(  # [x, y, frame, slice]
                dset.k_data, coil_sens[:, :, :, 0], fft_axes=(0, 1), coil_axis=2, frame_axis=3,
            )

        # reorder dims to fastMRI format
        mask = mask.T  # [frame, ky]
        logging.info(f'New mask shape: {mask.shape} (frame, ky)')
        kdata_masked = kdata_masked.transpose(4, 2, 3, 0, 1)  # [slice, coil, frame, kx, ky]
        logging.info(f'New k-space shape: {kdata_masked.shape} (slice, coil, frame, kx, ky)')
        for k in recons:
            recons[k] = recons[k].transpose(3, 2, 0, 1)  # [slice, frame, x, y]
            logging.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        logging.info('Cropping phase oversampling')
        for k in recons:
            recons[k] = center_crop(recons[k], 3, recon_phase_size)
            logging.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        # save the sensitivity-weighted reconstruction as a GIF for a quick visual sanity check
        if sampling.name == 'fs' and 'weighted' in recons:
            recon = dset.norm_orientation(recons['weighted'], image_axes=(-2, -1))
            gif_dir = self.target_dir / 'ocmr_recons'
            if exclude:
                gif_dir /= 'excluded'
            gif_dir.mkdir(exist_ok=True, parents=True)
            if cp is not None:
                recon = cp.asnumpy(recon)
            tres = dset.hdr.sequenceParameters.TR[0]  # type: ignore
            save_movie(recon[0], gif_dir / output_name, clip=True, tres=tres)

        if (exclude
            or dset.split in [None, 'nan']
            or (dset.split == 'test' and sampling.name == 'fs')
            or (dset.split in ('train', 'val') and sampling.name != 'fs')
        ):
            return

        assert isinstance(dset.slice_idx, int)
        self.save_data(output_name, dset, sampling, kdata_masked, mask, recons, data_attrs, dset.slice_idx)

    def process_slice(self, dset: Dataset, data_attrs: dict[str, Any], slc_idx: int, output_name: str, exclude: bool):
        sampling_patterns = self.get_sampling_patterns(str(data_attrs['smp']), str(data_attrs['mask type']), exclude)

        if hasattr(dset, 'k_data_full') and dset.n_slices == dset.k_data_full.shape[6]:
            dset.select_slice(slc_idx)
        else:
            logging.info(f'Reading k-space data for slice {slc_idx + 1}')
            dset.read_kdata(whiten=True, select_slice=slc_idx)

        logging.info(f'K-space shape: {dset.k_data.shape} (kx, ky, kz, coil, frame, set, slice, rep, avg)')

        # the number of frames may differ between slices; check for last frame with data
        sum_per_frame = np.moveaxis(np.abs(dset.k_data), 4, 0).reshape(dset.k_data.shape[4], -1).sum(axis=1)
        last_nonzero = np.nonzero(sum_per_frame)[0][-1]
        dset.k_data = dset.k_data[:, :, :, :, :last_nonzero + 1, :, :, :, :]

        dset, pad_left, pad_right = self.phase_pad_dataset(dset)

        if data_attrs['ech'] == 'asy':
            if data_attrs['smp'] == 'fs':
                logging.info('Running POCS')
                dset.k_data = pocs(dset.k_data)
                left = max(0, pad_left)
                right = min(dset.k_data.shape[1], dset.k_data.shape[1] - pad_right)
                dset.k_data[:, :left] = 0
                dset.k_data[:, right:] = 0
            else:
                logging.warning('POCS only implemented for fully sampled data, skipping')

        if dset.has_read_os:  # crop readout oversampling -- assumed consistently 200%
            logging.info('Removing readout oversampling')
            dset.k_data = crop_readout_oversampling(dset.k_data)

        assert dset.k_data.shape[2] == 1, 'kz must be 1'
        assert dset.k_data.shape[5] == 1, 'set must be 1'
        assert dset.k_data.shape[7] == 1, 'rep must be 1'
        assert dset.k_data.shape[8] == 1, 'avg must be 1'
        dset.k_data = dset.k_data.squeeze(axis=(2, 5, 7, 8))  # [kx, ky, coil, frame, slice]
        logging.info(f'K-space shape: {dset.k_data.shape} (kx, ky, coil, frame, slice)')

        recon_phase_size = self.get_recon_phase_size(dset)

        for sampling in sampling_patterns:
            logging.info(f'-- Processing sampling \'{sampling.name}\' on slice {slc_idx + 1} of dataset {dset.name} --')
            self.process_slice_with_sampling(
                dset, data_attrs, sampling, output_name, exclude, pad_left, pad_right, recon_phase_size,
            )

    def process_sample(self, data_attrs: dict[str, Any]):
        raw_file = self.raw_dir / str(data_attrs['file name'])

        dset, data_attrs = self.load_dataset(raw_file, data_attrs)

        excluded_slices = []
        if not pd.isna(data_attrs['excluded slices']):
            for slc in str(data_attrs['excluded slices']).split('+'):
                if '-' in slc:
                    start, end = slc.split('-')
                    excluded_slices.extend(list(range(int(start), int(end) + 1)))
                else:
                    excluded_slices.append(int(slc))
        logging.info(f'Excluded slices: {excluded_slices}')

        for slc_idx in range(dset.n_slices):
            logging.info(f'---- Processing slice {slc_idx + 1} / {dset.n_slices} ----')
            output_name = f'{dset.filename.stem}_slice{slc_idx:02d}'
            exclude = slc_idx in excluded_slices
            self.process_slice(dset, data_attrs, slc_idx, output_name, exclude)

    def run(self):
        pbar = tqdm(self.df.iterrows(), total=len(self.df), desc='Files', unit='file')
        for count, (_, row) in enumerate(pbar):
            data_attrs = row.to_dict()
            pbar.set_postfix_str(str(data_attrs['file name']))
            logging.info('-' * 50)
            logging.info('')
            logging.info(f'Processing {data_attrs["file name"]} ({count + 1} / {len(self.df)})')
            logging.info('')
            self.process_sample(data_attrs)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DEFAULT_CSV = Path(__file__).parent / 'ocmr.csv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=Path, required=True,
                        help='Dir containing the extracted raw .h5 files (e.g. G:\\OCMR_extracted\\OCMR_data). '
                             'Required -- no machine-specific default.')
    parser.add_argument('--target_dir', type=Path, required=True,
                        help='Dir where processed data samples are stored (ocmr_train/ocmr_val/ocmr_test_* and a '
                             'coil_sens/ cache subfolder all live under here). Required -- no default.')
    parser.add_argument('--csv_query', nargs='*', type=str, default=None,
                        help='Pandas query to filter data set, e.g. --csv_query "smp==\'fs\'"')
    parser.add_argument('--csv_idx_range', nargs='*', type=int, default=[0, None],
                        help='Indices of data samples to process. (start) | (start, stop)')
    parser.add_argument('--accelerations', '--acceleration', nargs='+', type=int, default=None,
                        help='Acceleration rates to use for retrospective (GRO) undersampling')
    parser.add_argument('--mask_types', '--mask_type', nargs='+', type=str, default=None, choices=['gro'],
                        help='Types of k-space masks to use for retrospective undersampling (only GRO is vendored)')
    parser.add_argument('--n_maps', type=int, default=1,
                        help='Number of coil sensitivity maps to estimate (sigpy only supports 1)')
    parser.add_argument('--no_cuda', dest='try_cuda', action='store_false',
                        help='Disable cuda even if available')
    parser.add_argument('--no_csv_persist', dest='persist_csv_updates', action='store_false',
                        help='Do not write per-file header info back to ocmr.csv. Required when running multiple '
                             'preprocessing processes in parallel (each with a disjoint --csv_idx_range) against '
                             'the same ocmr.csv, to avoid them clobbering each other\'s writes.')
    args = parser.parse_args()

    if args.csv_query:
        args.csv_query = ' and '.join(args.csv_query)

    if len(args.csv_idx_range) == 0:
        args.csv_idx_range = [0, None]
    elif len(args.csv_idx_range) == 1:
        args.csv_idx_range = [args.csv_idx_range[0], None]
    elif len(args.csv_idx_range) != 2:
        raise ValueError(f'Invalid csv_idx_range: {args.csv_idx_range}')

    logging.info('Arguments: ')
    for k, v in vars(args).items():
        if '__' in k:
            continue
        logging.info(f'{k}: {v}')

    preprocessing = Preprocessing(
        DEFAULT_CSV,
        args.raw_dir,
        args.target_dir,
        csv_query=args.csv_query,
        csv_idx_range=args.csv_idx_range,
        accelerations=args.accelerations,
        mask_types=args.mask_types,
        n_maps=args.n_maps,
        try_cuda=args.try_cuda,
        persist_csv_updates=args.persist_csv_updates,
    )
    preprocessing.run()
