"""Metrics, plots, and small helpers for the zero-shot sampler."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402


def resolve_device(requested: str) -> torch.device:
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA requested but unavailable, falling back to CPU.')
        return torch.device('cpu')
    return torch.device(requested)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def magnitude(x) -> np.ndarray:
    """complex [1, F, H, W] (or [F, H, W]) -> real magnitude [F, H, W]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.abs(x)
    return x[0] if x.ndim == 4 else x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# The four metrics below are deliberately defined to match CineVN's
# src/cinevn/metrics.py, because CineVN's published OCMR/GRO numbers are the
# baseline this project is compared against and a comparison table is only valid
# if both sides compute the same quantity. Two conventions matter:
#   * `maxval` is max(reference), NOT max - min.
#   * SSIM uses a 7x7 uniform (non-Gaussian) window, applied per frame and then
#     averaged -- CineVN reaches this through torchmetrics configured to match
#     skimage's defaults, which is what we call directly.
# CineVN does NOT histogram-equalise before scoring; equalize_adapthist appears
# only in its image-writing path.

def psnr(reference: np.ndarray, test: np.ndarray, maxval: float | None = None) -> float:
    maxval = float(reference.max()) if maxval is None else float(maxval)
    mse = float(np.mean((reference - test) ** 2))
    return float('inf') if mse == 0 else 20 * np.log10(maxval) - 10 * np.log10(mse)


def ssim(reference: np.ndarray, test: np.ndarray, maxval: float | None = None) -> float:
    maxval = float(reference.max()) if maxval is None else float(maxval)
    scores = [
        structural_similarity(
            reference[i], test[i], data_range=maxval or 1.0,
            win_size=7, gaussian_weights=False,
        )
        for i in range(reference.shape[0])
    ]
    return float(np.mean(scores))


def nmse(reference: np.ndarray, test: np.ndarray) -> float:
    denominator = float(np.sum(reference ** 2)) or 1.0
    return float(np.sum((reference - test) ** 2) / denominator)


def hfen(reference: np.ndarray, test: np.ndarray, sigma: float = 1.5) -> float:
    """
    High-Frequency Error Norm (Ravishankar & Bresler, 10.1109/TMI.2010.2090538).

    The L2 norm of a Laplacian-of-Gaussian filtered error image, so it responds to
    lost edges and fine texture rather than to bulk intensity error. Worth watching
    precisely because PSNR *rewards* blur: a flow prior that over-smooths at high
    acceleration can improve its PSNR while HFEN gets worse.

    Not normalised, matching CineVN -- the value scales with image intensity, so it
    is comparable across methods on the same data but not across datasets.
    """
    from scipy.ndimage import gaussian_laplace

    return float(np.linalg.norm(gaussian_laplace(reference - test, sigma=sigma)))


def ssim_xt(
    reference: np.ndarray,
    test: np.ndarray,
    center: tuple[int, int],
    n_profiles: int = 8,
    maxval: float | None = None,
) -> float:
    """
    SSIM on x-t profiles through the heart -- CineVN's spatiotemporal metric.

    Takes a horizontal and a vertical line through `center` at each of
    `n_profiles // 2` rotations of the clip, stacks each line over time into an x-t
    image, and scores those. Because every profile is a slice through *time*, this
    measures temporal fidelity directly, which makes it the natural published
    counterpart to this project's `bg_flicker`.

    `center` is (row, col), matching the `center` attribute's ordering in the
    preprocessed files (verified against `bbox` on the test split). CineVN indexes
    the transposed way because it applies an affine to its annotations first.

    Faithful to CineVN, the clip is rotated about its own centre while the profile
    coordinates stay fixed, so profiles at non-zero angles are chords near the
    heart rather than exactly through it.
    """
    from torchvision.transforms.functional import rotate

    if reference.ndim != 3:
        raise ValueError(f'ssim_xt expects [F, H, W], got {reference.shape}.')
    if n_profiles < 2 or n_profiles & (n_profiles - 1):
        raise ValueError(f'n_profiles must be a power of two >= 2, got {n_profiles}.')

    maxval = float(reference.max()) if maxval is None else float(maxval)
    height, width = reference.shape[1:]
    row, col = int(center[0]), int(center[1])
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f'ssim_xt centre {(row, col)} is outside the image {(height, width)}; '
            f'the landmark and the image are on different coordinate systems.'
        )
    radius = int(min(height, width) // 4)
    angle_increment = 360 / (2 * n_profiles)

    row_low, row_high = max(0, row - radius), min(height, row + radius)
    col_low, col_high = max(0, col - radius), min(width, col + radius)

    reference_t = torch.from_numpy(np.ascontiguousarray(reference))
    test_t = torch.from_numpy(np.ascontiguousarray(test))

    scores = []
    for i in range(n_profiles // 2):
        angle = i * angle_increment
        ref_rot = rotate(reference_t, angle).numpy()
        test_rot = rotate(test_t, angle).numpy()
        for ref_profile, test_profile in (
            (ref_rot[:, row, col_low:col_high], test_rot[:, row, col_low:col_high]),
            (ref_rot[:, row_low:row_high, col], test_rot[:, row_low:row_high, col]),
        ):
            if min(ref_profile.shape) < 7:  # SSIM needs a 7x7 window
                continue
            scores.append(structural_similarity(
                ref_profile, test_profile, data_range=maxval or 1.0,
                win_size=7, gaussian_weights=False,
            ))
    return float(np.mean(scores)) if scores else float('nan')


def temporal_tv(frames: np.ndarray) -> float:
    """
    Mean absolute frame-to-frame difference.

    Real cine has a healthy amount of this -- the heart is moving -- so the number
    on its own means nothing. What matters is the ratio against the reference:
    well above 1 means the reconstruction is varying in time more than the anatomy
    does, which is exactly what flicker is.
    """
    if frames.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(frames, axis=0))))


def _background_mask(shape: tuple[int, int], bbox: list[int] | None) -> np.ndarray:
    """
    Pixels that should be static over time: everything outside the heart bounding box.

    Falls back to a border ring when the bbox is missing or does not fit the image
    (e.g. after a combined-mode crop, which moves the coordinates).
    """
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    if bbox is not None and len(bbox) == 4:
        h_low, w_low, h_high, w_high = bbox
        if 0 <= h_low < h_high <= height and 0 <= w_low < w_high <= width:
            mask[h_low:h_high, w_low:w_high] = False
            return mask
    margin_h, margin_w = height // 4, width // 4
    mask[margin_h : height - margin_h, margin_w : width - margin_w] = False
    return mask


def background_flicker(frames: np.ndarray, reference: np.ndarray, bbox: list[int] | None) -> float:
    """
    Temporal standard deviation outside the heart, as a fraction of mean reference
    intensity.

    The background is anatomically static, so any temporal variation there is
    artefact rather than motion. This is the cleanest single number for the
    shared-vs-independent-noise ablation: unlike `temporal_tv` it cannot be
    confounded by genuine cardiac motion.
    """
    if frames.shape[0] < 2:
        return 0.0
    mask = _background_mask(frames.shape[-2:], bbox)
    scale = float(np.mean(reference)) or 1.0
    return float(np.mean(frames.std(axis=0)[mask]) / scale)


def _crop_bbox(frames: np.ndarray, bbox: list[int] | None) -> np.ndarray | None:
    """The heart region, as [F, h, w], or None if the bbox does not fit the image."""
    if bbox is None or len(bbox) != 4:
        return None
    row_low, col_low, row_high, col_high = bbox
    height, width = frames.shape[-2:]
    if not (0 <= row_low < row_high <= height and 0 <= col_low < col_high <= width):
        return None
    return frames[:, row_low:row_high, col_low:col_high]


def score(
    reconstruction,
    reference,
    bbox: list[int] | None = None,
    center: tuple[int, int] | None = None,
) -> dict:
    """
    All metrics for one clip, on magnitude images.

    Three groups:
      * CineVN-comparable -- psnr, ssim, nmse, hfen, ssim_xt (same definitions as
        the baseline, so the numbers can go in one table).
      * ROI -- the same fidelity metrics restricted to the heart bounding box.
        Whole-image numbers are diluted by a large, mostly-empty background that
        any method reconstructs well, so they overstate quality; the ROI numbers
        are what actually reflects diagnostic content. Scored against the global
        `maxval`, as CineVN does, so ROI and whole-image values stay on one scale.
      * temporal -- ttv_ratio and bg_flicker, this project's own diagnostics for
        the flicker the shared-noise correction targets.
    """
    rec, ref = magnitude(reconstruction), magnitude(reference)
    maxval = float(ref.max())
    ref_ttv = temporal_tv(ref)

    metrics = {
        'psnr': psnr(ref, rec, maxval),
        'ssim': ssim(ref, rec, maxval),
        'nmse': nmse(ref, rec),
        'hfen': hfen(ref, rec),
        'ttv': temporal_tv(rec),
        'ttv_ratio': temporal_tv(rec) / ref_ttv if ref_ttv else 0.0,
        'bg_flicker': background_flicker(rec, ref, bbox),
    }

    if center is not None:
        metrics['ssim_xt'] = ssim_xt(ref, rec, center, maxval=maxval)

    ref_roi, rec_roi = _crop_bbox(ref, bbox), _crop_bbox(rec, bbox)
    if ref_roi is not None and min(ref_roi.shape[-2:]) >= 7:
        metrics['psnr_roi'] = psnr(ref_roi, rec_roi, maxval)
        metrics['ssim_roi'] = ssim(ref_roi, rec_roi, maxval)
        metrics['nmse_roi'] = nmse(ref_roi, rec_roi)

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shared_vmax(reference: np.ndarray) -> float:
    return float(np.percentile(reference, 99.5)) or 1.0


def plot_reconstruction(reconstruction, reference, zero_filled, path: Path,
                        max_frames: int = 6, title: str = '') -> None:
    """
    Reference / reconstruction / zero-filled, one row each.

    Every row is displayed at the *reference's* window, not its own, so a
    reconstruction that came out too bright or too dim -- a scale-estimation
    failure -- is visible rather than normalised away.
    """
    rows = {
        'reference': magnitude(reference),
        'reconstruction': magnitude(reconstruction),
        'zero-filled': magnitude(zero_filled),
    }
    vmax = _shared_vmax(rows['reference'])
    n_frames = min(max_frames, rows['reference'].shape[0])
    picks = np.linspace(0, rows['reference'].shape[0] - 1, n_frames).round().astype(int)

    fig, axes = plt.subplots(len(rows), n_frames,
                             figsize=(1.7 * n_frames, 1.9 * len(rows)), squeeze=False)
    for row, (label, frames) in enumerate(rows.items()):
        for col, idx in enumerate(picks):
            ax = axes[row][col]
            ax.imshow(frames[idx], cmap='gray', vmin=0, vmax=vmax)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'frame {idx}', fontsize=8)
        axes[row][0].axis('on')
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        axes[row][0].set_ylabel(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_temporal_profile(clips: dict, path: Path) -> None:
    """
    y-t profiles through the image centre.

    Flicker -- the artefact shared-noise trajectory correction targets -- shows up
    here as vertical streaking: pixels changing frame to frame that should not.
    """
    fig, axes = plt.subplots(1, len(clips), figsize=(4 * len(clips), 3.4), squeeze=False)
    vmax = _shared_vmax(magnitude(next(iter(clips.values()))))
    for ax, (label, clip) in zip(axes[0], clips.items()):
        frames = magnitude(clip)
        ax.imshow(frames[:, :, frames.shape[2] // 2].T, cmap='gray', aspect='auto',
                  vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('frame')
        ax.set_ylabel('y')
    fig.suptitle('Temporal profile (centre column over time)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cine_gif(clip, path: Path, fps: int = 10, vmax: float | None = None) -> None:
    """
    Animate a clip.

    Pass `vmax` to window several clips identically -- comparing a reconstruction
    against the reference is only meaningful if both are displayed the same way,
    and flicker is far easier to see in an animation than in a still.
    """
    frames = magnitude(clip)
    vmax = vmax if vmax is not None else _shared_vmax(frames)
    images = [Image.fromarray((np.clip(f / vmax, 0, 1) * 255).astype(np.uint8)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)


def save_arrays(path: Path, arrays: dict, attrs: dict | None = None) -> None:
    """
    Write raw complex arrays to HDF5 so a run's numbers survive without re-running it.

    Recomputing a metric, cropping a region for a figure, or diffing two runs all
    need the actual values, not a rendered PNG -- and re-running the sampler to get
    them back costs far more than the disk. One clip is roughly 8 MB per array at
    complex64, so `output.save_array_inputs` is off by default: the reference is
    identical across every cell of a sweep and the zero-filled recon depends only
    on R, making them pure duplication once you have more than one arm.
    """
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        for name, value in arrays.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            handle.create_dataset(name, data=np.asarray(value), compression='gzip',
                                  compression_opts=4)
        for key, value in (attrs or {}).items():
            if value is not None:
                handle.attrs[key] = value


def plot_ablation(results: dict, path: Path, metric: str = 'psnr', ylabel: str = '') -> None:
    """
    One metric across accelerations, one line per noise mode.

    `results` maps noise mode -> {acceleration: (mean, std)}.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, by_r in sorted(results.items()):
        accelerations = sorted(by_r)
        means = [by_r[r][0] for r in accelerations]
        errors = [by_r[r][1] for r in accelerations]
        ax.errorbar(accelerations, means, yerr=errors, marker='o', capsize=3, label=mode)
    ax.set_xlabel('acceleration R')
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f'{metric} vs acceleration, by trajectory-correction noise')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
