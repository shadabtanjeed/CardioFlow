"""Training helpers: EMA, checkpoints, metrics, and plotting."""

import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print('[warn] CUDA requested but unavailable, falling back to CPU.')
        return torch.device('cpu')
    return torch.device(requested)


def rng_state() -> dict:
    """Captures every RNG stream the training pipeline draws from, so a resumed
    run reproduces the same augmentation/noise sequence it would have without
    interruption."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, active: bool = True) -> None:
        decay = self.decay if active else 0.0
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(decay).add_(m.detach(), alpha=1 - decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model,
    ema,
    optimizer,
    scaler,
    scheduler,
    best_val: float,
    best_epoch: int,
    history: dict,
    cfg: dict,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'ema': ema.shadow.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_val': best_val,
            'best_epoch': best_epoch,
            'history': history,
            'rng_state': rng_state(),
            'config': cfg,
        },
        path,
    )


def write_json(path: Path, payload: dict) -> None:
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_magnitude(clip) -> np.ndarray:
    """[C, F, H, W] -> [F, H, W] magnitude."""
    if isinstance(clip, torch.Tensor):
        clip = clip.detach().float().cpu().numpy()
    return np.sqrt(clip[0] ** 2 + clip[1] ** 2) if clip.shape[0] == 2 else np.abs(clip[0])


def _to_uint8(frames: np.ndarray, vmax: float | None = None) -> np.ndarray:
    vmax = vmax if vmax is not None else float(np.percentile(frames, 99.5)) or 1.0
    return (np.clip(frames / vmax, 0, 1) * 255).astype(np.uint8)


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    data_range = float(reference.max() - reference.min()) or 1.0
    mse = float(np.mean((reference - test) ** 2))
    return float('inf') if mse == 0 else 20 * float(np.log10(data_range / np.sqrt(mse)))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    data_range = float(reference.max() - reference.min()) or 1.0
    scores = [
        structural_similarity(reference[i], test[i], data_range=data_range)
        for i in range(reference.shape[0])
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curve(epochs, values, title: str, path: Path, color: str = 'tab:blue') -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, values, color=color, linewidth=1.2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('flow-matching loss')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_combined(history: dict, path: Path, best_epoch: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history['epoch'], history['train_loss'], label='train', color='tab:blue',
            alpha=0.6, linewidth=1.0)
    if history['val_epoch']:
        ax.plot(history['val_epoch'], history['val_loss'], label='val', color='tab:red',
                marker='o', markersize=3, linewidth=1.4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color='k', linestyle='--', linewidth=1, label=f'best ({best_epoch})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('flow-matching loss')
    ax.set_title('Training vs validation loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_t_loss(t_values, losses, path: Path, title: str = 'Loss vs flow time') -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_values, losses, marker='o', color='tab:purple')
    ax.set_xlabel('t   (0 = noise, 1 = data)')
    ax.set_ylabel('flow-matching loss')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cine_grid(clip, path: Path, title: str = '', max_frames: int = 8) -> None:
    frames = to_magnitude(clip)
    n = min(max_frames, frames.shape[0])
    picks = np.linspace(0, frames.shape[0] - 1, n).round().astype(int)
    vmax = float(np.percentile(frames, 99.5)) or 1.0

    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.2))
    for ax, idx in zip(np.atleast_1d(axes), picks):
        ax.imshow(frames[idx], cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(f't={idx}', fontsize=8)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cine_gif(clip, path: Path, fps: int = 10) -> None:
    frames = _to_uint8(to_magnitude(clip))
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 / max(fps, 1)), loop=0
    )


def plot_temporal_profile(clips: dict, path: Path) -> None:
    """
    y-t profiles through the image centre.

    Temporal flicker -- the artefact the project's shared-noise correction
    targets -- shows up here as horizontal striping.
    """
    fig, axes = plt.subplots(1, len(clips), figsize=(4 * len(clips), 3.4), squeeze=False)
    for ax, (label, clip) in zip(axes[0], clips.items()):
        frames = to_magnitude(clip)
        profile = frames[:, :, frames.shape[2] // 2].T  # [H, F]
        ax.imshow(profile, cmap='gray', aspect='auto',
                  vmin=0, vmax=float(np.percentile(frames, 99.5)) or 1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('frame')
        ax.set_ylabel('y')
    fig.suptitle('Temporal profile (centre column over time)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(rows: dict, path: Path, max_frames: int = 6, title: str = '') -> None:
    """rows: {label: clip} rendered as one row per entry, sharing a frame axis."""
    labels = list(rows)
    n_frames = min(max_frames, to_magnitude(rows[labels[0]]).shape[0])
    picks = np.linspace(0, to_magnitude(rows[labels[0]]).shape[0] - 1, n_frames).round().astype(int)

    fig, axes = plt.subplots(
        len(labels), n_frames, figsize=(1.7 * n_frames, 1.9 * len(labels)), squeeze=False
    )
    for row, label in enumerate(labels):
        frames = to_magnitude(rows[label])
        vmax = float(np.percentile(frames, 99.5)) or 1.0
        for col, idx in enumerate(picks):
            ax = axes[row][col]
            ax.imshow(frames[idx], cmap='gray', vmin=0, vmax=vmax)
            ax.axis('off')
            if row == 0:
                ax.set_title(f't={idx}', fontsize=8)
        axes[row][0].set_ylabel(label, fontsize=9)
        axes[row][0].axis('on')
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
