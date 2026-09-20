"""
Does the prior denoise frame-correlated noise as well as frame-i.i.d. noise?

This is the mechanism test behind the ablation result in README.md. The sweep showed
`independent` trajectory-correction noise beating `shared` on every metric at every
acceleration; the leading explanation was that a 3D U-Net's temporal convolutions
*attenuate* frame-i.i.d. noise but pass frame-constant noise through at full amplitude,
so a shared draw is far harder for the prior to remove. Nothing in the sweep tests that
directly -- too much else happens between a reset and the final image.

This does, with no sampler, no k-space and no data consistency in the way:

    x_t       = t * x1 + (1 - t) * noise          (the interpolation the prior trained on)
    estimate  = x_t + (1 - t) * velocity(x_t, t)  (one look-ahead, trajectory_correction.py)
    error     = ||estimate - x1|| / ||x1||

The only thing that varies is the noise's correlation along the frame axis:

    noise = sqrt(rho) * shared + sqrt(1 - rho) * independent

which has unit per-voxel variance and cross-frame correlation exactly `rho` for any
rho in [0, 1]. So rho=0 is `independent`, rho=1 is `shared`, and the magnitude of the
perturbation is identical throughout -- only its temporal structure changes. If error
rises monotonically with rho, the mechanism is confirmed, and the shape of the curve
says whether a partially-correlated compromise is worth trying.

Runs at the size the prior trained on (12 frames, 128x104, `normalize: std`) so the
test is in-distribution and cheap enough for CPU: ~2.3 s per forward pass.

    python noise_probe.py --checkpoint_dir <run> --data_dir <processed>
    python noise_probe.py ... --rho 0,0.5,1 --t 0.25,0.5,0.75 --seeds 5 --clips 4
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import find_files, load_clip  # noqa: E402
from kspace import complex_std, to_channels  # noqa: E402
from prior import load_prior  # noqa: E402
from sampler import make_velocity_fn  # noqa: E402
from utils import resolve_device, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe the prior's denoising vs temporal noise correlation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint_dir', required=True, type=Path)
    p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--weights', default='ema', choices=['ema', 'model'])
    p.add_argument('--data_dir', required=True, type=Path)
    p.add_argument('--acceleration', type=int, default=8,
                   help='which split to pull clips from (only the reference is used)')
    p.add_argument('--clips', type=int, default=3, help='how many clips to average over')
    p.add_argument('--rho', default='0,0.25,0.5,0.75,0.9,1.0',
                   help='cross-frame noise correlations to test; 0=independent, 1=shared')
    p.add_argument('--t', default='0.1,0.25,0.5,0.75,0.9',
                   help='flow times to probe (the schedule resets across this whole range)')
    p.add_argument('--seeds', type=int, default=3, help='noise draws averaged per cell')
    p.add_argument('--frames', type=int, default=12, help='clip length (training default 12)')
    p.add_argument('--crop', default='128,104', help='H,W crop (training default 128,104)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, default=Path('noise_probe'))
    return p


def correlated_noise(shape, rho: float, generator, device):
    """
    Unit-variance Gaussian noise with cross-frame correlation exactly `rho`.

    noise = sqrt(rho)*shared + sqrt(1-rho)*independent has variance rho + (1-rho) = 1
    and, for two different frames, covariance rho -- so the per-voxel marginal is
    identical to `randn` at every rho and only the temporal structure changes. At
    rho=0 and rho=1 this reduces exactly to noise.py's two modes.
    """
    batch, channels, frames, height, width = shape
    shared = torch.randn((batch, channels, 1, height, width), generator=generator)
    independent = torch.randn(shape, generator=generator)
    noise = np.sqrt(rho) * shared.expand(-1, -1, frames, -1, -1) + np.sqrt(1 - rho) * independent
    return noise.to(device)


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    rhos = [float(v) for v in args.rho.split(',')]
    times = [float(v) for v in args.t.split(',')]
    crop_h, crop_w = (int(v) for v in args.crop.split(','))

    flow, info = load_prior(args.checkpoint_dir, device,
                            checkpoint=args.checkpoint, weights=args.weights)
    print(f'Prior   : {info["checkpoint"]} (epoch {info["epoch"]}, val {info["best_val"]:.5f})')
    print(f'Device  : {device}')
    print(f'Grid    : {len(rhos)} rho x {len(times)} t x {args.seeds} seeds x {args.clips} clips '
          f'= {len(rhos)*len(times)*args.seeds*args.clips} forward passes\n')

    # --- clean clips, in the exact representation the prior trained on ----------
    targets = []
    for path in find_files(args.data_dir, args.acceleration, limit=args.clips):
        clip = load_clip(path, args.data_dir, coil_mode='multicoil')
        reference = clip['reference']
        if reference.shape[1] < args.frames:
            continue
        # centre crop in space, leading slab in time -- matching the training crop
        _, frames, height, width = reference.shape
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        reference = reference[:, :args.frames, top:top + crop_h, left:left + crop_w]
        x1 = to_channels(reference / complex_std(reference)).to(device)
        targets.append((clip['file_id'], x1))
    if not targets:
        raise SystemExit('No clip long enough; lower --frames.')

    velocity = make_velocity_fn(flow, 1, device, amp=False,
                                multiple=info['downsample_multiple'])

    # --- the probe --------------------------------------------------------------
    results: dict = {}
    started = time.time()
    for t in times:
        for rho in rhos:
            errors, kept = [], []
            for file_id, x1 in targets:
                for seed in range(args.seeds):
                    generator = torch.Generator().manual_seed(1000 * seed + 7)
                    noise = correlated_noise(tuple(x1.shape), rho, generator, device)
                    x_t = t * x1 + (1.0 - t) * noise
                    estimate = x_t + (1.0 - t) * velocity(x_t, t)
                    scale = x1.norm()
                    errors.append(float((estimate - x1).norm() / scale))
                    # how much of the perturbation actually got removed
                    kept.append(float((estimate - x1).norm() / (x_t - x1).norm()))
            results.setdefault(t, {})[rho] = {
                'error': float(np.mean(errors)), 'error_std': float(np.std(errors)),
                'kept': float(np.mean(kept)),
            }
        row = results[t]
        base = row[rhos[0]]['error']
        print(f't={t:<5.2f} ' + '  '.join(
            f'rho={r:<4g} {row[r]["error"]:.4f} ({row[r]["error"]/base:+.0%})' for r in rhos))
    print(f'\n{len(rhos)*len(times)*args.seeds*len(targets)} passes in '
          f'{time.time()-started:.0f}s')

    # --- verdict ----------------------------------------------------------------
    print('\n' + '=' * 74)
    print('relative error of the look-ahead estimate, independent (rho=0) -> shared (rho=1)')
    print('=' * 74)
    print(f'{"t":>6s}' + ''.join(f'{f"rho={r:g}":>12s}' for r in rhos) + f'{"shared/indep":>14s}')
    penalties = []
    for t in times:
        row = results[t]
        ratio = row[rhos[-1]]['error'] / row[rhos[0]]['error']
        penalties.append(ratio)
        print(f'{t:6.2f}' + ''.join(f'{row[r]["error"]:12.4f}' for r in rhos)
              + f'{ratio:13.2f}x')
    monotone = all(
        all(results[t][a]['error'] <= results[t][b]['error'] + 1e-6
            for a, b in zip(rhos, rhos[1:]))
        for t in times
    )
    print(f'\nmonotone in rho at every t : {monotone}')
    print(f'mean shared/independent    : {np.mean(penalties):.2f}x')
    print('CONFIRMED: the prior removes frame-correlated noise far less well.'
          if np.mean(penalties) > 1.05 else
          'NOT CONFIRMED: correlation does not meaningfully change denoising.')

    # --- outputs ----------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'noise_probe.json', {
        'results': {str(t): {str(r): v for r, v in row.items()} for t, row in results.items()},
        'rho': rhos, 't': times, 'clips': [f for f, _ in targets],
        'seeds': args.seeds, 'frames': args.frames, 'crop': [crop_h, crop_w],
        'prior': info, 'monotone_in_rho': monotone,
        'mean_shared_over_independent': float(np.mean(penalties)),
    })

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for t in times:
        left.plot(rhos, [results[t][r]['error'] for r in rhos], 'o-', label=f't = {t:g}')
    left.set_xlabel('cross-frame noise correlation  $\\rho$   (0 = independent, 1 = shared)')
    left.set_ylabel('relative error of look-ahead estimate')
    left.set_title('Prior denoises correlated noise worse')
    left.grid(alpha=0.3), left.legend(fontsize=8)

    right.plot(times, penalties, 'o-', color='crimson')
    right.axhline(1.0, color='grey', ls='--', lw=1)
    right.set_xlabel('flow time  $t$')
    right.set_ylabel('error(shared) / error(independent)')
    right.set_title('Penalty for sharing the noise')
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / 'noise_probe.png', dpi=180, bbox_inches='tight')
    print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
