"""
Loading one undersampled cine clip from the preprocessing output.

Layout under `data_dir` (see preprocessing/preprocess_ocmr.py):
    ocmr_test_gro_08/<subject>_sliceNN.h5     (and _12, _16, _20)
    coil_sens/<subject>_sliceNN/coil_sens_avg.h5

Each test file stores the *already masked* multi-coil `kspace`, the `mask` that
produced it, and -- for scoring only -- `reconstruction_weighted`, the
fully-sampled coil-combined reference. That reference is byte-identical across the
four acceleration folders (checked), so it is genuinely the ground truth and not
something derived from the mask. Nothing but the scoring path may read it.
"""

from pathlib import Path

import h5py
import numpy as np
import torch

from kspace import fft2c


def find_files(data_dir: Path, acceleration: int, limit: int | None = None) -> list[Path]:
    split_dir = Path(data_dir) / f'ocmr_test_gro_{acceleration:02d}'
    if not split_dir.is_dir():
        available = sorted(p.name for p in Path(data_dir).glob('ocmr_test_gro_*'))
        raise FileNotFoundError(
            f'{split_dir} not found. Available test splits: {available or "none"}.'
        )
    files = sorted(split_dir.glob('*.h5'))
    if not files:
        raise FileNotFoundError(f'No .h5 files under {split_dir}.')
    return files[:limit] if limit else files


def _shift_point(point: list[int] | None, top: int, left: int,
                 height: int, width: int) -> list[int] | None:
    """Move a (row, col) landmark into crop coordinates, or drop it if it falls outside."""
    if point is None:
        return None
    row, col = point[0] - top, point[1] - left
    return [row, col] if 0 <= row < height and 0 <= col < width else None


def _shift_box(box: list[int] | None, top: int, left: int,
               height: int, width: int) -> list[int] | None:
    """Move a (row_low, col_low, row_high, col_high) box into crop coordinates, clipped."""
    if box is None:
        return None
    row_low, col_low = max(0, box[0] - top), max(0, box[1] - left)
    row_high, col_high = min(height, box[2] - top), min(width, box[3] - left)
    return [row_low, col_low, row_high, col_high] if row_low < row_high and col_low < col_high else None


def load_clip(
    path: Path,
    data_dir: Path,
    coil_mode: str = 'multicoil',
    max_frames: int | None = None,
    crop: tuple[int, int] | None = None,
) -> dict:
    """
    One clip, as tensors with a leading batch axis of 1.

    coil_mode:
      multicoil -- the real inverse problem: measured multi-coil k-space plus
                   ESPIRiT maps. This is what CineVN is benchmarked on.
      combined  -- a single-coil simulation: undersample the coil-combined
                   reference directly with the same k-t mask. Mask fusion is then
                   an exact orthogonal projection, which makes it the cleaner
                   testbed for the shared-noise ablation, at the cost of not being
                   the real acquisition model.

    `crop` is only meaningful in combined mode -- an image-domain crop is
    inconsistent with a k-space mask defined over the full FOV, so multicoil runs
    always use the full FOV. `max_frames` is valid in both, since the transform is
    a 2D FFT per frame and frames are independent under it.
    """
    with h5py.File(path, 'r') as handle:
        kspace = np.asarray(handle['kspace'][0])                      # [coil, frame, kx, ky]
        mask = np.asarray(handle['mask'])                             # [frame, ky]
        reference = np.asarray(handle['reconstruction_weighted'][0])  # [frame, kx, ky]
        attrs = {
            'acceleration': int(handle.attrs['acceleration']),
            'bbox': [int(v) for v in handle.attrs['bbox']] if 'bbox' in handle.attrs else None,
            # (row, col), same ordering as bbox -- verified against it on the
            # test split. Used for the x-t profiles ssim_xt scores.
            'center': [int(v) for v in handle.attrs['center']] if 'center' in handle.attrs else None,
            'patient_id': str(handle.attrs.get('patient_id', path.stem)),
            'view': str(handle.attrs.get('view', '')),
        }

    if max_frames:
        kspace = kspace[:, :max_frames]
        mask = mask[:max_frames]
        reference = reference[:max_frames]

    sens = None
    if coil_mode == 'multicoil':
        if crop:
            raise ValueError(
                'data.crop is only supported with data.coil_mode=combined: an image-domain '
                'crop is inconsistent with a k-space mask defined over the full FOV.'
            )
        sens_path = Path(data_dir) / 'coil_sens' / path.stem / 'coil_sens_avg.h5'
        if not sens_path.exists():
            raise FileNotFoundError(
                f'Coil maps not found at {sens_path}. Run with '
                f'--set data.coil_mode=combined to use the single-coil simulation instead.'
            )
        with h5py.File(sens_path, 'r') as handle:
            sens = np.asarray(handle['coil_sens'][0, 0, :, 0])        # [coil, kx, ky]

    elif coil_mode == 'combined':
        # The reference has phase oversampling removed, so it can be narrower than
        # the acquisition grid the mask is defined on (22 of 38 test clips). Since
        # the crop is central, centre-crop the mask to match before simulating.
        if mask.shape[-1] != reference.shape[-1]:
            offset = (mask.shape[-1] - reference.shape[-1]) // 2
            mask = mask[:, offset : offset + reference.shape[-1]]
        if crop:
            height, width = crop
            full_h, full_w = reference.shape[-2:]
            if height > full_h or width > full_w:
                raise ValueError(f'crop {tuple(crop)} exceeds image size {(full_h, full_w)}.')
            top, left = (full_h - height) // 2, (full_w - width) // 2
            reference = reference[:, top : top + height, left : left + width]
            mask = mask[:, left : left + width]
            # The heart annotations index the uncropped image, so they must move
            # with the crop -- and are dropped outright if they fall outside it,
            # rather than silently pointing at the wrong anatomy.
            attrs['center'] = _shift_point(attrs['center'], top, left, height, width)
            attrs['bbox'] = _shift_box(attrs['bbox'], top, left, height, width)
    else:
        raise ValueError(f'Unknown data.coil_mode {coil_mode!r}; expected multicoil or combined.')

    reference_t = torch.from_numpy(reference).to(torch.complex64).unsqueeze(0)  # [1, F, H, W]
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None, :, None, :]  # [1, 1, F, 1, W]

    if coil_mode == 'combined':
        # Simulate the acquisition: the same k-t mask, applied to the reference.
        kspace_t = mask_t * fft2c(reference_t).unsqueeze(1)                     # [1, 1, F, H, W]
        sens_t = None
    else:
        kspace_t = torch.from_numpy(kspace).to(torch.complex64).unsqueeze(0)    # [1, C, F, H, W]
        sens_t = torch.from_numpy(sens).to(torch.complex64)[None, :, None]      # [1, C, 1, H, W]
        # The stored k-space is already masked; re-applying is a cheap invariant check.
        kspace_t = mask_t * kspace_t

    return {
        'kspace': kspace_t,
        'mask': mask_t,
        'sens': sens_t,
        'reference': reference_t,
        'file_id': path.stem,
        **attrs,
    }
