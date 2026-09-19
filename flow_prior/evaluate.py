"""
Evaluate a trained flow-matching prior on the val and test splits.

Run standalone against a finished run:
    python evaluate.py --exp exps/20260916-1200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import load_config
from dataset import build_dataset
from flow import FlowMatching, stratified_t
from unet import UNet3D
from utils import (
    plot_comparison,
    plot_per_t_loss,
    plot_temporal_profile,
    psnr,
    resolve_device,
    save_cine_gif,
    save_cine_grid,
    ssim,
    to_magnitude,
    write_json,
)


def build_flow(cfg: dict) -> FlowMatching:
    net = UNet3D(**cfg['model'])
    return FlowMatching(net, time_scale=cfg['flow']['time_scale'])


def build_loader(cfg: dict, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        build_dataset(cfg, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg['data']['num_workers'],
        drop_last=False,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate_loss(
    flow: FlowMatching,
    loader: DataLoader,
    device,
    t_bins: int = 8,
    max_batches: int | None = None,
    seed: int = 1234,
    progress: bool = False,
):
    """
    Deterministic flow-matching loss.

    Each clip is scored at a fixed grid of t values with fixed noise, so the
    number is comparable across checkpoints -- sampling t randomly makes the
    val curve too noisy to select a best model from.
    """
    flow.eval()
    t_grid = stratified_t(t_bins, device)
    totals = np.zeros(t_bins)
    counts = np.zeros(t_bins)

    batches = enumerate(loader)
    if progress:
        total = len(loader) if max_batches is None else min(max_batches, len(loader))
        batches = tqdm(batches, total=total, desc='eval', leave=False)

    for i, x1 in batches:
        if max_batches is not None and i >= max_batches:
            break
        x1 = x1.to(device)
        generator = torch.Generator().manual_seed(seed + i)
        x0 = torch.randn(x1.shape, generator=generator).to(device)
        for j in range(t_bins):
            t = t_grid[j].expand(x1.shape[0])
            per_sample = flow.loss(x1, t=t, x0=x0, reduce=False)
            totals[j] += float(per_sample.sum())
            counts[j] += x1.shape[0]

    per_t = totals / np.maximum(counts, 1)
    return float(per_t.mean()), t_grid.cpu().numpy(), per_t


@torch.no_grad()
def generate_samples(flow: FlowMatching, cfg: dict, device, out_dir: Path) -> torch.Tensor:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg['eval']
    shape = (
        eval_cfg['num_samples'],
        cfg['model']['in_channels'],
        cfg['data']['frames'],
        *cfg['data']['crop'],
    )
    generator = torch.Generator().manual_seed(cfg['train']['seed'])
    samples = flow.sample(shape, steps=eval_cfg['sample_steps'], device=device, generator=generator)

    for i, clip in enumerate(samples):
        save_cine_grid(clip, out_dir / f'sample_{i}.png', title=f'Unconditional sample {i}')
        save_cine_gif(clip, out_dir / f'sample_{i}.gif', fps=eval_cfg['gif_fps'])
    return samples


@torch.no_grad()
def restoration_check(
    flow: FlowMatching, loader: DataLoader, cfg: dict, device, out_dir: Path
) -> dict:
    """
    Noise real clips to t_start, integrate back to t=1, and score the result.

    Metrics are averaged over the whole split (or the first `eval.restore_clips`
    clips), with the spread reported alongside -- a single clip says very little.
    The saved figure shows the first clip only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_clips = cfg['eval'].get('restore_clips')
    metrics = {}

    for t_start in cfg['eval']['restore_t']:
        scores: list[tuple[float, float]] = []
        first: tuple[torch.Tensor, torch.Tensor] | None = None

        for batch_idx, batch in enumerate(tqdm(loader, desc=f'restore t={t_start}', leave=False)):
            if max_clips is not None and len(scores) >= max_clips:
                break
            reference = batch.to(device)
            generator = torch.Generator().manual_seed(cfg['train']['seed'] + batch_idx)
            restored = flow.restore(
                reference,
                t_start=float(t_start),
                steps=max(1, int(cfg['eval']['sample_steps'] * (1 - float(t_start)))),
                generator=generator,
            )
            for ref, out in zip(reference, restored):
                ref_mag, out_mag = to_magnitude(ref), to_magnitude(out)
                scores.append((psnr(ref_mag, out_mag), ssim(ref_mag, out_mag)))
                if first is None:
                    first = (ref, out)

        psnrs, ssims = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
        metrics[f't={t_start}'] = {
            'psnr': float(psnrs.mean()),
            'psnr_std': float(psnrs.std()),
            'ssim': float(ssims.mean()),
            'ssim_std': float(ssims.std()),
            'num_clips': len(scores),
        }
        if first is not None:
            plot_comparison(
                {'reference': first[0], f'restored (t0={t_start})': first[1]},
                out_dir / f'restore_t{t_start}.png',
                title=f'Partial-noise restoration from t={t_start} '
                      f'(split mean PSNR {psnrs.mean():.2f} dB over {len(scores)} clips)',
            )
    return metrics


def run_evaluation(
    flow: FlowMatching,
    cfg: dict,
    exp_dir: Path,
    device,
    splits=('val', 'test'),
) -> dict:
    flow.eval()
    eval_dir = exp_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print('\nGenerating unconditional samples...')
    samples = generate_samples(flow, cfg, device, eval_dir / 'samples')

    for split in splits:
        try:
            loader = build_loader(cfg, split, batch_size=cfg['train']['batch_size'])
        except FileNotFoundError as e:
            print(f'[warn] skipping {split}: {e}')
            continue

        split_dir = eval_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f'Scoring {split} split ({len(loader.dataset)} clips)...')
        mean_loss, t_values, per_t = evaluate_loss(
            flow, loader, device, t_bins=cfg['train']['val_t_bins'], progress=True
        )
        plot_per_t_loss(
            t_values, per_t, split_dir / 'loss_per_t.png', title=f'{split}: loss vs flow time'
        )

        batch = next(iter(loader))
        restore_metrics = restoration_check(flow, loader, cfg, device, split_dir)

        plot_temporal_profile(
            {'real': batch[0], 'generated': samples[0]}, split_dir / 'temporal_profile.png'
        )
        plot_comparison(
            {'real': batch[0], 'generated': samples[0]},
            split_dir / 'real_vs_generated.png',
            title='Real clip vs unconditional sample',
        )

        results[split] = {
            'num_clips': len(loader.dataset),
            'flow_loss': mean_loss,
            'loss_per_t': {f'{t:.3f}': float(v) for t, v in zip(t_values, per_t)},
            'restoration': restore_metrics,
        }
        print(f'  {split}: flow_loss={mean_loss:.5f}')

    # Val is scored by training and test only later, in a separate deliberate run, so
    # merge into any existing metrics instead of clobbering the other split's numbers.
    metrics_path = eval_dir / 'metrics.json'
    merged = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            merged = json.load(f)
    merged.update(results)
    write_json(metrics_path, merged)

    print(f'\nEvaluation written to {eval_dir}')
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained CardioFlow prior.')
    parser.add_argument('--exp', type=Path, required=True, help='Experiment dir under exps/.')
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    parser.add_argument('--data_dir', type=str, default=None, help='Override data.data_dir.')
    args = parser.parse_args()

    cfg = load_config(args.exp / 'config.yaml')
    if args.data_dir:
        cfg['data']['data_dir'] = args.data_dir

    device = resolve_device(cfg['train']['device'])
    checkpoint_path = args.exp / 'checkpoints' / args.checkpoint
    if not checkpoint_path.exists():
        available = sorted(p.name for p in (args.exp / 'checkpoints').glob('*.pt'))
        raise SystemExit(
            f'{checkpoint_path} not found. Available in this run: {available or "none"}'
        )
    # weights_only=False: this is our own trusted checkpoint, which also carries plain-Python
    # RNG state and a config dict that `weights_only=True` (the torch>=2.6 default) rejects.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    flow = build_flow(cfg).to(device)
    flow.velocity_fn.load_state_dict(checkpoint['ema'])
    print(f"Loaded {args.checkpoint} from epoch {checkpoint['epoch']} "
          f"(best val {checkpoint['best_val']:.5f})")

    run_evaluation(flow, cfg, args.exp, device, splits=tuple(args.splits))


if __name__ == '__main__':
    main()
