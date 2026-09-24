"""
The zero-shot k-t sampler: Restora-Flow's algorithm, moved into k-t space.

One reconstruction is a walk from t = 0 (noise) to t = 1 (data) along the
schedule in schedule.py:

    forward step (t_last < t_cur)
        1. mask fusion  -- overwrite the measured k-space lines with the
                           observation re-noised to t_last (mask_fusion.py)
        2. Euler step   -- x <- x + (t_cur - t_last) * v_theta(x, t_last)

    backward jump (t_last > t_cur)
        trajectory correction -- look ahead to a one-shot estimate of the clean
        clip, then re-noise it back to t_cur (trajectory_correction.py)

The one deliberate divergence from Restora-Flow is which noise the backward jump
draws: per-frame ('independent', the naive port) or one draw shared across frames
('shared', this project's contribution). See noise.py.

    python sampler.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed> --acceleration 8
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibration import calibrate_scale
from config import build_parser, resolve_config, save_config
from data import find_files, load_clip
from kspace import (
    KTOperator,
    center_crop,
    estimate_scale,
    pad_to_multiple,
    to_complex,
    unpad,
)
from mask_fusion import fuse
from noise import draw_noise
from prior import load_prior
from schedule import time_pairs
from trajectory_correction import correct
from utils import (
    plot_reconstruction,
    plot_temporal_profile,
    resolve_device,
    save_arrays,
    save_cine_gif,
    score,
    write_json,
)


def make_velocity_fn(flow, batch_size: int, device: torch.device, amp: bool, multiple: int = 8):
    """
    Wraps the prior as a plain (x, t: float) -> velocity callable.

    Padding to a multiple of `multiple` happens here and is undone immediately, so
    the U-Net always sees a size it can downsample while every k-space operation
    outside this function stays on the true acquisition grid.
    """

    # autocast('cuda') is meaningless off CUDA, so gate on the device as train.py
    # does rather than on the config flag alone -- otherwise `amp: true` (the
    # default) misbehaves the moment anyone runs on CPU.
    use_amp = amp and device.type == 'cuda'

    def velocity(x: torch.Tensor, t: float) -> torch.Tensor:
        padded, pad = pad_to_multiple(x, multiple)
        t_tensor = torch.full((batch_size,), float(t), device=device)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out = flow.velocity(padded, t_tensor)
        return unpad(out.float(), pad)

    return velocity


@torch.no_grad()
def reconstruct(flow, clip: dict, cfg: dict, device: torch.device, generator=None,
                downsample_multiple: int = 8) -> dict:
    """
    Reconstruct one clip. Returns the complex image on the original data scale and
    cropped to the reference's FOV, plus the quantities worth logging.
    """
    sampler_cfg, data_cfg = cfg['sampler'], cfg['data']

    observation = clip['kspace'].to(device)
    mask = clip['mask'].to(device)
    sens = clip['sens'].to(device) if clip['sens'] is not None else None
    operator = KTOperator(mask, sens)

    zero_filled = operator.zero_filled(observation)

    # calibration.calibrate_scale measures its ratio on the zero-filled recon
    # cropped to the reference FOV, so crop before measuring here too, or the
    # statistic includes oversampled columns the reference does not have.
    target = tuple(clip['reference'].shape[-2:])
    scale = estimate_scale(
        center_crop(zero_filled, target),
        mode=data_cfg['scale_mode'],
        constant=data_cfg.get('scale_constant'),
        reference=clip['reference'].to(device) if data_cfg['scale_mode'] == 'reference' else None,
    )
    observation = observation / scale

    batch, frames, height, width = zero_filled.shape
    shape = (batch, 2, frames, height, width)
    velocity = make_velocity_fn(
        flow, batch, device, bool(sampler_cfg['amp']), multiple=downsample_multiple
    )

    # Restora-Flow initialises from pure noise, as unconditional sampling would.
    x = draw_noise(shape, sampler_cfg['init_noise'], device, generator)

    pairs = time_pairs(sampler_cfg['ode_steps'], sampler_cfg['correction_steps'])
    progress = tqdm(pairs, desc=clip['file_id'], leave=False, disable=not sampler_cfg['progress'])

    output = None
    first_forward = True
    n_fusions = n_corrections = 0

    for t_last, t_cur in progress:
        if t_last < t_cur:
            # Matching the reference implementation, the very first forward step
            # skips fusion: at t = 0 the re-noised observation is pure noise, so
            # injecting it would say nothing about the measured data.
            if not first_forward:
                x = fuse(
                    x, t_last, operator, observation,
                    noise_mode=sampler_cfg['fusion_noise'], generator=generator,
                )
                n_fusions += 1
            x = x + (t_cur - t_last) * velocity(x, t_last)
            output = x
            first_forward = False
        else:
            x = correct(
                x, t_last, t_cur, velocity,
                noise_mode=sampler_cfg['correction_noise'], generator=generator,
            )
            n_corrections += 1

    image = to_complex(output)

    if sampler_cfg['final_dc']:
        # Off by default: Restora-Flow has no final projection, and leaving it off
        # keeps this a faithful port. Enforcing the measured lines exactly at t = 1
        # is standard in MRI reconstruction and usually worth a little PSNR.
        k = operator.unmasked_forward(image)
        image = operator.adjoint(mask * observation + (1.0 - mask) * k)

    # The reference has phase oversampling removed, so it can be narrower than the
    # acquisition grid the sampler ran on. Crop to it for scoring and display.
    return {
        'reconstruction': center_crop(image * scale, target).cpu(),
        'zero_filled': center_crop(zero_filled, target).cpu(),
        'scale': float(scale),
        'n_fusions': n_fusions,
        'n_corrections': n_corrections,
        'n_velocity_calls': n_fusions + n_corrections + len(pairs),
    }


def run(cfg: dict, out_dir: Path, flow=None, prior_info: dict | None = None,
        device: torch.device | None = None, quiet: bool = False,
        desc: str | None = None, position: int = 0) -> dict:
    """
    Reconstruct `data.limit` clips at one acceleration and score them.

    `flow`/`prior_info`/`device` let a caller (evaluate.py) load the prior once and
    reuse it across a whole sweep instead of re-reading the checkpoint per cell.
    `desc`/`position` let a caller label this run's progress bar (e.g. with which
    sweep cell it is) and stack it under an outer bar instead of colliding with it.
    """
    device = device or resolve_device(cfg['sampler']['device'])
    if flow is None:
        flow, prior_info = load_prior(
            Path(cfg['prior']['checkpoint_dir']), device,
            checkpoint=cfg['prior']['checkpoint'], weights=cfg['prior']['weights'],
        )

    data_cfg = cfg['data']
    files = find_files(Path(data_cfg['data_dir']), data_cfg['acceleration'], data_cfg['limit'])

    # `auto` is resolved here, once per run (not per clip): calibrate fresh from
    # ocmr_val, then mutate data_cfg in place so `reconstruct()` -- and the saved
    # `metrics.json['config']`, which shares this same dict -- see plain
    # `constant` mode with the value actually used. evaluate.py pre-resolves this
    # itself (once per acceleration, cached across arms) before calling `run`, so
    # this branch only fires for a standalone `sampler.py` invocation.
    if data_cfg['scale_mode'] == 'auto':
        scale_value, n_calib = calibrate_scale(
            Path(data_cfg['val_data_dir']), data_cfg['acceleration'], data_cfg['coil_mode'],
            device, limit=data_cfg.get('calib_limit'),
        )
        data_cfg['scale_mode'] = 'constant'
        data_cfg['scale_constant'] = scale_value
        data_cfg['calibration_n_clips'] = n_calib
        if not quiet:
            print(f'Scale (auto)   : {scale_value:.4f}  (calibrated on {n_calib} ocmr_val '
                  f'clips, R={data_cfg["acceleration"]}, {data_cfg["coil_mode"]})')

    if not quiet:
        print(f'Prior          : {prior_info["checkpoint"]}')
        print(f'                 epoch {prior_info["epoch"]}, val {prior_info["best_val"]:.5f}, '
              f'{prior_info["weights"]} weights')
        print(f'Device         : {device}')
        print(f'Acceleration   : R={data_cfg["acceleration"]}  ({len(files)} clips, '
              f'{data_cfg["coil_mode"]})')
        print(f'Sampler        : {cfg["sampler"]["ode_steps"]} ODE steps, '
              f'{cfg["sampler"]["correction_steps"]} correction steps, '
              f'correction noise = {cfg["sampler"]["correction_noise"]}')

    per_clip, started = [], time.time()
    # Not gated on `quiet`: quiet suppresses the verbose header/summary text (so a
    # sweep doesn't reprint them once per cell), but a live "x/38 clips" bar is the
    # only feedback during a cell that can otherwise run silently for minutes --
    # `leave=False` clears it once done, so evaluate.py's one-line-per-cell summary
    # stays the permanent record instead of 12 stale finished bars.
    for path in tqdm(files, desc=desc or 'clips', unit='clip', leave=False, position=position):
        clip = load_clip(
            path, Path(data_cfg['data_dir']),
            coil_mode=data_cfg['coil_mode'],
            max_frames=data_cfg['max_frames'],
            crop=tuple(data_cfg['crop']) if data_cfg['crop'] else None,
        )
        generator = torch.Generator().manual_seed(cfg['sampler']['seed'])
        result = reconstruct(
            flow, clip, cfg, device, generator,
            downsample_multiple=prior_info['downsample_multiple'],
        )

        metrics = score(result['reconstruction'], clip['reference'],
                        bbox=clip['bbox'], center=clip['center'])
        baseline = score(result['zero_filled'], clip['reference'],
                         bbox=clip['bbox'], center=clip['center'])
        per_clip.append({
            'file_id': clip['file_id'],
            'view': clip['view'],
            'scale': result['scale'],
            **metrics,
            **{f'zf_{k}': v for k, v in baseline.items()},
        })

        clip_dir = out_dir / 'clips'
        if cfg['output']['save_clips']:
            clip_dir.mkdir(parents=True, exist_ok=True)
            plot_reconstruction(
                result['reconstruction'], clip['reference'], result['zero_filled'],
                clip_dir / f'{clip["file_id"]}.png',
                title=f'{clip["file_id"]}  R={clip["acceleration"]}  '
                      f'PSNR {metrics["psnr"]:.2f} dB (zero-filled {baseline["psnr"]:.2f})',
            )
            plot_temporal_profile(
                {'reference': clip['reference'], 'reconstruction': result['reconstruction'],
                 'zero-filled': result['zero_filled']},
                clip_dir / f'{clip["file_id"]}_temporal.png',
            )
            # All three animations share the reference's window, so a brightness
            # difference reads as a brightness difference rather than being
            # normalised away -- and flicker is easiest to judge side by side.
            window = float(np.percentile(np.abs(clip['reference'].numpy()), 99.5)) or 1.0
            for label, frames in (
                ('', result['reconstruction']),
                ('_reference', clip['reference']),
                ('_zero_filled', result['zero_filled']),
            ):
                save_cine_gif(frames, clip_dir / f'{clip["file_id"]}{label}.gif',
                              fps=cfg['output']['gif_fps'], vmax=window)

        if cfg['output']['save_arrays']:
            arrays = {'reconstruction': result['reconstruction'].numpy()}
            if cfg['output']['save_array_inputs']:
                arrays['reference'] = clip['reference'].numpy()
                arrays['zero_filled'] = result['zero_filled'].numpy()
            save_arrays(
                clip_dir / f'{clip["file_id"]}.h5', arrays,
                attrs={
                    'file_id': clip['file_id'],
                    'acceleration': clip['acceleration'],
                    'view': clip['view'],
                    'scale': result['scale'],
                    'correction_noise': cfg['sampler']['correction_noise'],
                    'correction_steps': cfg['sampler']['correction_steps'],
                    'ode_steps': cfg['sampler']['ode_steps'],
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'bg_flicker': metrics['bg_flicker'],
                },
            )

    summary = _summarise(per_clip)
    summary['elapsed_s'] = round(time.time() - started, 1)
    summary['num_clips'] = len(per_clip)
    summary['acceleration'] = data_cfg['acceleration']
    summary['correction_noise'] = cfg['sampler']['correction_noise']

    summary['correction_steps'] = cfg['sampler']['correction_steps']

    payload = {'summary': summary, 'per_clip': per_clip, 'prior': prior_info, 'config': cfg}
    write_json(out_dir / 'metrics.json', payload)

    if not quiet:
        print('\n' + _format_summary(summary))
        print(f'\nWritten to {out_dir}')
    return payload


def _summarise(per_clip: list[dict]) -> dict:
    """
    Mean and spread per metric.

    Keys are collected across every clip, not just the first: a clip missing the
    `center` landmark has no ssim_xt, and one whose bbox does not fit has no ROI
    metrics, so keying off clip 0 would silently drop a metric for the whole run.
    """
    if not per_clip:
        return {}
    keys = {k for clip in per_clip for k, v in clip.items() if isinstance(v, (int, float))}
    summary = {}
    for key in sorted(keys):
        values = [float(c[key]) for c in per_clip if isinstance(c.get(key), (int, float))]
        finite = torch.tensor([v for v in values if np.isfinite(v)])
        if not len(finite):
            continue
        summary[key] = float(finite.mean())
        summary[f'{key}_std'] = float(finite.std(unbiased=False))
        if len(finite) < len(per_clip):
            summary[f'{key}_n'] = len(finite)
    return summary


# (label, metric key, section header shown before it)
SUMMARY_ROWS = [
    ('PSNR (dB)', 'psnr', 'comparable with CineVN'),
    ('SSIM', 'ssim', None),
    ('NMSE', 'nmse', None),
    ('HFEN', 'hfen', None),
    ('SSIM_XT', 'ssim_xt', None),
    ('PSNR (dB), heart', 'psnr_roi', 'heart ROI only'),
    ('SSIM, heart', 'ssim_roi', None),
    ('NMSE, heart', 'nmse_roi', None),
    ('temporal TV ratio', 'ttv_ratio', 'temporal (this project)'),
    ('background flicker', 'bg_flicker', None),
]


def _format_summary(summary: dict) -> str:
    lines = [f'{"metric":22s} {"reconstruction":>22s} {"zero-filled":>16s}']
    for label, key, header in SUMMARY_ROWS:
        if key not in summary:
            continue
        if header:
            lines.append(f'-- {header} ' + '-' * max(0, 58 - len(header)))
        zero_filled = summary.get(f'zf_{key}')
        zf_text = f'{zero_filled:>16.4f}' if zero_filled is not None else f'{"-":>16s}'
        lines.append(
            f'{label:22s} {summary[key]:>10.4f} +/- {summary[f"{key}_std"]:<7.4f} {zf_text}'
        )
    return '\n'.join(lines)


def main() -> None:
    parser = build_parser('Zero-shot k-t reconstruction with a CardioFlow prior.')
    args = parser.parse_args()
    cfg = resolve_config(args)

    out_dir = Path(cfg['output']['dir']) / _run_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / 'config.yaml')
    run(cfg, out_dir)


def _run_name(cfg: dict) -> str:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = cfg['output'].get('name')
    tag = f'R{cfg["data"]["acceleration"]:02d}_{cfg["sampler"]["correction_noise"]}'
    return f'{stamp}_{name}_{tag}' if name else f'{stamp}_{tag}'


if __name__ == '__main__':
    main()
