"""
Publication figures from a finished sweep: undersampled input vs reconstruction vs
ground truth, for one clip.

Reads the reconstructions a sweep already wrote (`<cell>/clips/<id>.h5`) and pairs
them with the reference and zero-filled input recomputed from the dataset, so it
needs no GPU and no prior -- just the sweep output plus the preprocessed data.

Rows are whatever you ask for via `--order`, so the same script draws the plain
three-row figure and the arm-vs-arm comparison:

    # undersampled -> reconstruction -> ground truth
    python figure.py --ablation_dir <sweep> --data_dir <processed> --clip fs_0080_1_5T_slice00

    # compare both noise modes against the reference
    python figure.py ... --order zero_filled,shared,independent,reference

    # add per-row error maps, pick frames and acceleration
    python figure.py ... --acceleration 20 --frames 0,6,12,18 --error_maps

A note on the zero-filled row: the adjoint A^H y is not the inverse, so its overall
brightness is arbitrary and it renders nearly black at the reference's window --
which is why it is barely visible in the sampler's own per-clip PNGs. Here it is
fitted to the reference by a single least-squares scalar (`--zf_scale fit`, the
default) so the row shows the *aliasing structure* rather than a trivial brightness
mismatch. `--zf_scale none` restores the raw adjoint scale.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_clip  # noqa: E402
from kspace import KTOperator, center_crop  # noqa: E402
from utils import score  # noqa: E402

# Rows that are not a sweep arm; anything else in --order is looked up as one.
REFERENCE = 'reference'
ZERO_FILLED = 'zero_filled'
PRETTY = {
    ZERO_FILLED: 'Undersampled\n(zero-filled)',
    REFERENCE: 'Ground truth',
    'independent': 'Reconstruction\n(independent)',
    'shared': 'Reconstruction\n(shared)',
    'no-correction': 'Reconstruction\n(no correction)',
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Draw an undersampled / reconstruction / ground-truth figure.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ablation_dir', required=True, type=Path,
                   help='a finished sweep directory (the one holding R08_shared/ etc.)')
    p.add_argument('--data_dir', required=True, type=Path,
                   help='preprocessing output, holding ocmr_test_gro_XX/ and coil_sens/')
    p.add_argument('--clip', default=None,
                   help='file id to draw; default = the first clip in the split')
    p.add_argument('--acceleration', type=int, default=8, help='which R to draw')
    p.add_argument('--order', default=f'{ZERO_FILLED},independent,{REFERENCE}',
                   help='comma-separated rows, top to bottom; a name that is not '
                        f'{ZERO_FILLED!r}/{REFERENCE!r} is treated as a sweep arm')
    p.add_argument('--frames', default=None,
                   help='comma-separated frame indices; default = 6 spread over the clip')
    p.add_argument('--n_frames', type=int, default=6,
                   help='how many frames when --frames is not given')
    p.add_argument('--profile', dest='profile', action='store_true', default=True,
                   help='append a y-t temporal-profile column (flicker shows as streaks)')
    p.add_argument('--no_profile', dest='profile', action='store_false')
    p.add_argument('--error_maps', action='store_true',
                   help='add an error-map row under each non-reference row')
    p.add_argument('--error_gain', type=float, default=4.0,
                   help='brightness multiplier for error maps')
    p.add_argument('--zf_scale', choices=['fit', 'none'], default='fit',
                   help="'fit' least-squares-matches the zero-filled row to the reference")
    p.add_argument('--window', type=float, default=99.5,
                   help='intensity window as a percentile of the reference')
    p.add_argument('--metrics', dest='metrics', action='store_true', default=True,
                   help='annotate each row with PSNR/SSIM against the reference')
    p.add_argument('--no_metrics', dest='metrics', action='store_false')
    p.add_argument('--coil_mode', default='multicoil', choices=['multicoil', 'combined'])
    p.add_argument('--cmap', default='gray')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--title', default=None, help='override the figure title')
    p.add_argument('--out', type=Path, default=None,
                   help='output path; default = <ablation_dir>/figure_<clip>_R<R>.png')
    return p


def load_panels(args) -> tuple[dict, dict, np.ndarray]:
    """Returns (panels {name: (T,H,W) magnitude}, clip metadata, reference)."""
    split = args.data_dir / f'ocmr_test_gro_{args.acceleration:02d}'
    if args.clip is None:
        args.clip = sorted(p.stem for p in split.glob('*.h5'))[0]
    clip = load_clip(split / f'{args.clip}.h5', args.data_dir, coil_mode=args.coil_mode)

    reference = np.abs(np.asarray(clip['reference']))[0]  # (1,T,H,W) -> (T,H,W)
    target = reference.shape[-2:]

    panels: dict[str, np.ndarray] = {}
    for name in args.order.split(','):
        name = name.strip()
        if not name or name in panels:
            continue
        if name == REFERENCE:
            panels[name] = reference
        elif name == ZERO_FILLED:
            operator = KTOperator(clip['mask'], clip.get('sens'))
            zf = center_crop(operator.zero_filled(clip['kspace']), target)
            zf = np.abs(np.asarray(zf))[0]
            if args.zf_scale == 'fit':
                # A^H y is not the inverse, so its scale is arbitrary; fit the single
                # scalar that best matches the reference so the row shows aliasing
                # structure rather than a brightness mismatch.
                zf = zf * float((zf * reference).sum() / max((zf * zf).sum(), 1e-12))
            panels[name] = zf
        else:
            h5 = args.ablation_dir / f'R{args.acceleration:02d}_{name}' / 'clips' / f'{args.clip}.h5'
            if not h5.exists():
                raise SystemExit(f'No reconstruction for arm {name!r} at R={args.acceleration}:'
                                 f'\n  expected {h5}')
            with h5py.File(h5, 'r') as f:
                rec = np.abs(f['reconstruction'][()])[0]
            if rec.shape[-2:] != target:
                rec = center_crop(torch.from_numpy(rec)[None], target)[0].numpy()
            panels[name] = rec
    return panels, clip, reference


def row_metrics(panel: np.ndarray, clip: dict) -> str:
    rec = torch.from_numpy(panel.astype(np.complex64))[None]
    m = score(rec, clip['reference'], bbox=clip.get('bbox'), center=clip.get('center'))
    return f'PSNR {m["psnr"]:.2f} dB   SSIM {m["ssim"]:.3f}'


def main() -> None:
    args = build_parser().parse_args()
    panels, clip, reference = load_panels(args)
    names = [n.strip() for n in args.order.split(',') if n.strip()]

    n_t = reference.shape[0]
    if args.frames:
        frames = [int(f) % n_t for f in args.frames.split(',')]
    else:
        frames = list(np.linspace(0, n_t - 1, min(args.n_frames, n_t)).round().astype(int))

    vmax = float(np.percentile(reference, args.window)) or 1.0
    # centre column through the heart when the annotation is available
    centre = clip.get('center')
    col = int(centre[1]) if centre is not None else reference.shape[-1] // 2
    col = min(max(col, 0), reference.shape[-1] - 1)

    rows = []
    for name in names:
        rows.append((name, panels[name], False))
        if args.error_maps and name != REFERENCE:
            rows.append((name, np.abs(panels[name] - reference), True))

    n_cols = len(frames) + (1 if args.profile else 0)
    height = reference.shape[-2]
    width = reference.shape[-1]
    aspect = height / width
    fig, axes = plt.subplots(
        len(rows), n_cols, squeeze=False,
        figsize=(1.7 * n_cols + 1.1, 1.7 * aspect * len(rows) + 0.7),
    )

    for r, (name, volume, is_error) in enumerate(rows):
        for c, frame in enumerate(frames):
            ax = axes[r][c]
            ax.imshow(volume[frame], cmap=args.cmap, vmin=0,
                      vmax=vmax / args.error_gain if is_error else vmax)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f'frame {frame}', fontsize=9)
        if args.profile:
            ax = axes[r][len(frames)]
            # (T,H,W) -> (H,T): y down, time across. Flicker reads as vertical streaks.
            ax.imshow(volume[:, :, col].T, cmap=args.cmap, vmin=0,
                      vmax=vmax / args.error_gain if is_error else vmax, aspect='auto')
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title('y–t profile', fontsize=9)

        label = 'error x%g' % args.error_gain if is_error else PRETTY.get(name, name)
        axes[r][0].set_ylabel(label, fontsize=9)
        if args.metrics and not is_error and name != REFERENCE:
            axes[r][0].text(0.03, 0.03, row_metrics(panels[name], clip),
                            transform=axes[r][0].transAxes, fontsize=7.5, color='white',
                            va='bottom', ha='left',
                            bbox=dict(facecolor='black', alpha=0.55, pad=1.8,
                                      edgecolor='none'))

    title = args.title or f'{args.clip}   R = {args.acceleration}'
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or args.ablation_dir / f'figure_{args.clip}_R{args.acceleration:02d}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Written {out}')


if __name__ == '__main__':
    main()
