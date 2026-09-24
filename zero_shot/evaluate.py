"""
The ablation: does sharing the trajectory-correction noise across frames help?

Sweeps acceleration x trajectory-correction noise mode, reusing one loaded prior,
and writes a comparison table plus per-metric plots. The three arms are:

    no-correction  -- correction_steps = 0, so the sampler never resets and the
                      noise mode is irrelevant. The floor: shows what trajectory
                      correction is worth at all.
    independent    -- the naive per-frame port of Restora-Flow.
    shared         -- CardioFlow: one noise draw broadcast across frames.

`bg_flicker` is the metric to read first. PSNR/SSIM average over the whole clip
and are dominated by spatial fidelity, so they can move very little even when
flicker is obvious; background temporal deviation measures the artefact directly
and cannot be confounded by real cardiac motion.

    python evaluate.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed>
    python evaluate.py --set ablation.accelerations=[8] --set data.limit=4
"""

import copy
import time
from pathlib import Path

import torch
from tqdm import tqdm

from calibration import calibrate_scale
from config import build_parser, resolve_config, save_config
from data import find_files
from prior import load_prior
from sampler import run
from utils import plot_ablation, resolve_device, write_json

# (metric key, human label, which direction is better)
REPORTED = [
    ('psnr', 'PSNR (dB)', 'higher'),
    ('ssim', 'SSIM', 'higher'),
    ('nmse', 'NMSE', 'lower'),
    ('hfen', 'HFEN', 'lower'),
    ('ssim_xt', 'SSIM_XT (spatiotemporal)', 'higher'),
    ('psnr_roi', 'PSNR (dB), heart ROI', 'higher'),
    ('ssim_roi', 'SSIM, heart ROI', 'higher'),
    ('ttv_ratio', 'temporal TV ratio', 'closer to 1'),
    ('bg_flicker', 'background flicker', 'lower'),
]


def calls_per_clip(ode_steps: int, correction_steps: int) -> int:
    """
    Network-call cost of reconstructing one clip -- see schedule.py. Not `ode_steps`
    alone: a `correction_steps=1` cell makes ~3x the calls of a `correction_steps=0`
    one at the same `ode_steps`, which is why the outer sweep bar below weights by
    this instead of by a flat per-cell count -- a naive count would make the ETA
    swing wildly every time a cheap no-correction cell finishes.
    """
    return (2 * correction_steps + 1) * ode_steps - 2 * correction_steps


def arms(cfg: dict) -> list[dict]:
    """The sweep's arms, as (label, correction_steps, correction_noise) triples."""
    ablation = cfg['ablation']
    out = []
    if ablation['include_no_correction']:
        out.append({'label': 'no-correction', 'correction_steps': 0, 'correction_noise': 'shared'})
    for mode in ablation['correction_noise']:
        out.append({
            'label': mode,
            'correction_steps': cfg['sampler']['correction_steps'],
            'correction_noise': mode,
        })
    return out


def sweep(cfg: dict, out_dir: Path) -> dict:
    device = resolve_device(cfg['sampler']['device'])
    flow, prior_info = load_prior(
        Path(cfg['prior']['checkpoint_dir']), device,
        checkpoint=cfg['prior']['checkpoint'], weights=cfg['prior']['weights'],
    )

    accelerations = cfg['ablation']['accelerations']
    sweep_arms = arms(cfg)
    ode_steps = cfg['sampler']['ode_steps']

    # Number of clips actually reconstructed per acceleration (respects data.limit),
    # cached once so the cost-weighting below doesn't re-glob the test split per cell.
    n_clips = {
        r: len(find_files(Path(cfg['data']['data_dir']), r, cfg['data']['limit']))
        for r in accelerations
    }
    cells = [(r, arm) for r in accelerations for arm in sweep_arms]
    total_calls = sum(n_clips[r] * calls_per_clip(ode_steps, arm['correction_steps'])
                      for r, arm in cells)

    print(f'Prior          : {prior_info["checkpoint"]}')
    print(f'                 epoch {prior_info["epoch"]}, val {prior_info["best_val"]:.5f}')
    print(f'Device         : {device}')
    print(f'Sweep          : R in {accelerations} x {[a["label"] for a in sweep_arms]} '
          f'= {len(cells)} runs')
    print(f'Clips per run  : {cfg["data"]["limit"] or "all"}\n')

    # `auto` is calibrated once per distinct acceleration in the sweep -- not once
    # per cell -- since the ratio depends on (acceleration, coil_mode) only, and
    # coil_mode is fixed for the whole sweep. Every arm at a given R then reuses the
    # same value, exactly as `run()` would if it resolved `auto` itself per cell;
    # this just avoids recomputing it `len(sweep_arms)` times over.
    scale_cache: dict[int, tuple[float, int]] = {}

    results: dict = {}
    started = time.time()
    # Weighted by network-call count, not by cell count: a no-correction cell makes
    # ~1/3 the calls of a shared/independent one at the same ode_steps (see
    # calls_per_clip), so weighting by cells alone would give a jumpy, misleading
    # ETA every time a cheap cell finishes. position=0 so this stays the top line
    # while each cell's own per-clip bar (sampler.py, position=1) runs underneath it.
    # A manually-driven bar, not one wrapping `cells` as an iterable: iterating a
    # tqdm-wrapped sequence auto-increments it by 1 per item, which would double
    # count on top of the explicit cost-weighted `.update()` below and overshoot
    # `total` well before the sweep actually finishes.
    outer = tqdm(total=total_calls, unit='call', position=0, leave=True, desc='sweep')
    for acceleration, arm in cells:
        label = arm['label']
        outer.set_description(f'sweep (R={acceleration} {label})')

        cell_cfg = copy.deepcopy(cfg)
        cell_cfg['data']['acceleration'] = acceleration
        cell_cfg['sampler']['correction_steps'] = arm['correction_steps']
        cell_cfg['sampler']['correction_noise'] = arm['correction_noise']
        cell_cfg['sampler']['progress'] = False

        if cell_cfg['data']['scale_mode'] == 'auto':
            if acceleration not in scale_cache:
                scale_cache[acceleration] = calibrate_scale(
                    Path(cfg['data']['val_data_dir']), acceleration, cfg['data']['coil_mode'],
                    device, limit=cfg['data'].get('calib_limit'),
                )
                value, n_calib = scale_cache[acceleration]
                outer.write(f'  calibrated scale for R={acceleration}: {value:.4f} '
                            f'(from {n_calib} ocmr_val clips)')
            value, n_calib = scale_cache[acceleration]
            cell_cfg['data']['scale_mode'] = 'constant'
            cell_cfg['data']['scale_constant'] = value
            cell_cfg['data']['calibration_n_clips'] = n_calib

        cell_dir = out_dir / f'R{acceleration:02d}_{label}'
        cell_dir.mkdir(parents=True, exist_ok=True)

        payload = run(cell_cfg, cell_dir, flow=flow, prior_info=prior_info,
                      device=device, quiet=True, desc=f'R={acceleration} {label}', position=1)
        summary = payload['summary']
        results.setdefault(label, {})[acceleration] = summary
        outer.write(f'  R={acceleration:<3d} {label:<14s} PSNR {summary["psnr"]:6.2f}  '
                    f'SSIM {summary["ssim"]:.4f}  '
                    f'SSIM_XT {summary.get("ssim_xt", float("nan")):.4f}  '
                    f'bg-flicker {summary["bg_flicker"]:.5f}  ({summary["elapsed_s"]:.0f}s)')
        outer.update(n_clips[acceleration] * calls_per_clip(ode_steps, arm['correction_steps']))
    outer.close()

    table = format_table(results, accelerations)
    print('\n' + table)

    for metric, label, _ in REPORTED:
        plot_ablation(
            {arm: {r: (s[metric], s.get(f'{metric}_std', 0.0)) for r, s in by_r.items()}
             for arm, by_r in results.items()},
            out_dir / f'ablation_{metric}.png', metric=metric, ylabel=label,
        )

    payload = {
        'results': results,
        'accelerations': accelerations,
        'arms': [a['label'] for a in sweep_arms],
        'prior': prior_info,
        'config': cfg,
        'elapsed_s': round(time.time() - started, 1),
    }
    write_json(out_dir / 'ablation.json', payload)
    (out_dir / 'ablation.md').write_text(table, encoding='utf-8')

    print(f'\nWritten to {out_dir}')
    return payload


def format_table(results: dict, accelerations: list[int]) -> str:
    """One block per metric: arms down the side, accelerations across."""
    arm_labels = list(results)
    width = max(len(a) for a in arm_labels) + 2
    blocks = []
    for metric, label, direction in REPORTED:
        header = f'{label}  ({direction} is better)'
        columns = '  '.join(f'R={r:<10d}' for r in accelerations)
        lines = [header, f'{"":{width}s}{columns}']
        for arm in arm_labels:
            cells = []
            for r in accelerations:
                summary = results[arm].get(r, {})
                if metric in summary:
                    cells.append(f'{summary[metric]:.4f} +/-{summary.get(f"{metric}_std", 0):.3f}')
                else:
                    cells.append('-')
            lines.append(f'{arm:{width}s}' + '  '.join(f'{c:<12s}' for c in cells))
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def main() -> None:
    parser = build_parser('Ablate shared vs independent trajectory-correction noise.')
    args = parser.parse_args()
    cfg = resolve_config(args)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = cfg['output'].get('name')
    out_dir = Path(cfg['output']['dir']) / (
        f'{stamp}_{name}_ablation' if name else f'{stamp}_ablation'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / 'config.yaml')
    sweep(cfg, out_dir)


if __name__ == '__main__':
    main()
