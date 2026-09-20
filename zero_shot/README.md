# zero_shot — Phase 3: zero-shot k-t reconstruction

Reconstructs undersampled cine MRI by steering the trained `flow_prior` with the
measured k-space. It is Restora-Flow's algorithm (Hadzic et al., WACV 2026;
[imigraz/Restora-Flow](https://github.com/imigraz/Restora-Flow)) moved from static
pixel-space masks into **k-t space**, plus the one change this project exists to test.

Nothing is trained here. The prior is loaded from a `flow_prior` run and frozen.

## The algorithm

A reconstruction walks from `t = 0` (noise) to `t = 1` (data) along a schedule that
periodically jumps backwards:

| schedule step | what happens | file |
|---|---|---|
| forward (`t_last < t_cur`) | **mask fusion**, then one Euler step of the ODE | `mask_fusion.py` |
| backward (`t_last > t_cur`) | **trajectory correction** | `trajectory_correction.py` |

**Mask fusion** transforms the current state to k-space, overwrites the *measured*
phase-encode lines with the observation re-noised to the current time
(`t * y + (1 - t) * noise`, matching the interpolation the prior trained on), keeps
the model's own values on the unmeasured lines, and transforms back. Re-noising
matters: injecting clean data into a state that is still mostly noise would put the
trajectory off-distribution.

**Trajectory correction** asks the prior where the current state would land if it ran
straight to `t = 1`, then re-noises that estimate back to an earlier time and re-walks
the stretch, so earlier errors get a second chance.

### The contribution

Restora-Flow restores single images, so its reset draws `randn_like(x)`. Applied
frame-by-frame to a cine clip that draws an **independent** noise field per frame:
the anatomy barely changes between frames but each frame is pushed a different way,
and the reconstruction flickers. CardioFlow draws **one** field and broadcasts it
across frames (`noise.py`). Everything else is identical, which makes the two modes a
clean A/B:

```bash
python evaluate.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed>
```

Read the **temporal** metrics first, not PSNR/SSIM. `ablation.include_no_correction` adds
a `correction_steps=0` floor, so the table answers both "does correction help at all" and
"does sharing the noise help".

## Metrics

Three groups, all in `utils.score`.

**Comparable with CineVN** — `psnr`, `ssim`, `nmse`, `hfen`, `ssim_xt`, defined to match
`CineVN-main/src/cinevn/metrics.py` so the numbers can share a table with its published
OCMR/GRO results. `nmse`, `psnr` and `hfen` were cross-checked against CineVN's own
implementations and agree to 1e-5. Two conventions are easy to get wrong: `maxval` is
`max(reference)`, not `max - min`; and SSIM uses a 7x7 *uniform* window applied per frame
and averaged. CineVN does **not** histogram-equalise before scoring -- `equalize_adapthist`
appears only in its image-writing path.

`hfen` (high-frequency error norm) is worth watching because PSNR actively *rewards*
blur: a prior that over-smooths at high acceleration can post a better PSNR while losing
exactly the fine structure HFEN measures.

**Heart ROI** — `psnr_roi`, `ssim_roi`, `nmse_roi`: the same fidelity metrics restricted to
the annotated bounding box, scored against the global `maxval` as CineVN does. Whole-image
numbers are diluted by a large, mostly-empty background that every method reconstructs
well, so they overstate quality; the ROI numbers track diagnostic content.

**Temporal (this project)** — `ttv_ratio` and `bg_flicker`, the diagnostics for the
artefact the shared-noise correction targets. `bg_flicker` is temporal standard deviation
*outside* the heart, so unlike `ttv_ratio` it cannot be confounded by genuine cardiac motion.

### Why the temporal metrics are the ones to read

Injecting noise of identical magnitude into a real clip, differing only in whether it is
drawn per-frame or shared across frames:

| | SSIM | SSIM_XT | bg_flicker | ttv_ratio |
|---|---|---|---|---|
| independent (flickers) | 0.3884 | 0.2982 | 0.644 | 7.96 |
| shared (temporally coherent) | 0.3902 | 0.3750 | 0.104 | 0.98 |

SSIM is identical to three decimal places. Spatial metrics are structurally blind to this
artefact, so an evaluation built only on PSNR/SSIM would miss the contribution entirely
even if it worked perfectly.

`ssim_xt` needs at least 7 frames (SSIM's window must fit the time axis) and a `center`
landmark inside the image; it is omitted rather than faked when either is unavailable,
which is why it does not appear in short `--max_frames` smoke runs.

## Usage

```bash
# one acceleration, whole split
python sampler.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed> --acceleration 8

# the full ablation sweep (R x noise mode), one prior load
python evaluate.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed>
```

Every value in `config.yaml` is settable from the CLI, exactly as in `flow_prior`:
short flags for the common ones (`--acceleration`, `--ode_steps`, `--correction_noise`,
`--limit`, `--device`, ...) and `--set any.nested.key=value` for the rest.

## What a run writes

```
output/<timestamp>_<name>_R08_shared/
  config.yaml                      # the exact resolved config
  metrics.json                     # per-clip + summary numbers
  clips/
    <id>.png                       # reference / reconstruction / zero-filled, one row each
    <id>_temporal.png              # y-t profiles of all three -- flicker shows as streaking
    <id>.gif                       # reconstruction
    <id>_reference.gif             # ground truth
    <id>_zero_filled.gif           # undersampled input
    <id>.h5                        # raw complex arrays + metrics as attributes
```

Every image is rendered at the **reference's** intensity window, so a reconstruction
that came out too bright or too dim reads as exactly that rather than being normalised
away. The three GIFs share that window too, which is what makes them comparable --
flicker is far easier to judge in an animation than in a still frame.

`<id>.h5` exists so a run's numbers survive without re-running it: recomputing a metric,
cropping a region for a figure, or diffing two arms all need the values, not a PNG.

Measured on a real full-FOV clip (19 frames, 256x208), not extrapolated: PNGs ~820 KB,
the three GIFs ~2.7 MB combined, the `.h5` ~7.2 MB (`save_array_inputs: true` makes it
~22 MB, roughly 3x). **At the defaults (`save_clips` and `save_arrays` both on), that's
~11 MB/clip** -- ~415 MB for one acceleration's 38 clips, ~5 GB for the full 4-R x
3-arm x 38-clip sweep (456 clips). Real sizes vary with each clip's frame count and FOV
(17-28 frames, 160x120 up to 256x208 across the test split), so treat 5 GB as a rough
upper-ish estimate. `--save_arrays false` drops the sweep to ~1.6 GB (figures only);
`--save_clips false` drops it to ~3.3 GB (arrays only, no figures).

`--save_array_inputs true` adds the reference and zero-filled recon to each `.h5` (3x
its size) -- useful for a one-off run you want self-contained, wasteful across a sweep
where the reference is identical in every cell and the zero-filled recon varies only
with R.

## Data facts worth not re-deriving

Established by inspecting the preprocessing output (2026-09-19). Several of these are
easy to get silently wrong:

- **The test-folder k-space is already masked.** It cannot reproduce the fully-sampled
  reference; only `reconstruction_weighted` is the ground truth, and it is byte-identical
  across the four acceleration folders.
- **`reconstruction_weighted` is a centre crop** of the acquisition FOV (phase oversampling
  removed) for 22 of 38 test clips — verified by an exhaustive offset search. The sampler
  runs on the full acquisition grid and centre-crops only to score.
- **Several k-space widths are not divisible by 8** (126, 150, 174, ...), which the U-Net
  requires. `kspace.pad_to_multiple` pads around each *network call* only, so no k-space
  operation ever sees a distorted grid.
- **ESPIRiT maps satisfy `sum_c |S_c|^2 = 1` exactly**, so `sum_c conj(S_c) x_c` is the
  properly normalised SENSE adjoint and `A = M.F.S` needs no extra division.
- **The training scale is recoverable, but not with one constant.** The prior trained on
  std-normalised clips; that std does not exist at inference. `ZERO_FILLED_SCALE` holds
  ratios measured per `(coil_mode, R)` — a single pooled constant is off by up to 20%,
  because the ratio falls steadily with R and differs by coil mode. The multicoil estimate
  lands within 3% of the oracle. `data.scale_mode=reference` is an oracle escape hatch for
  telling "the scale estimate is off" apart from "the sampler is wrong".

## Coil modes

- `multicoil` (default) — the real inverse problem: measured multi-coil k-space plus
  ESPIRiT maps, fused with the standard data-consistency layer. This is the setting
  CineVN's published numbers are for.
- `combined` — a single-coil simulation: the same k-t mask applied to the coil-combined
  reference. Mask fusion is then an exact orthogonal projection, so it isolates the noise
  ablation from coil-model error. `data.crop` is only valid here, since an image-domain
  crop is inconsistent with a k-space mask over the full FOV.

## Files

| file | role |
|---|---|
| `sampler.py` | the sampling loop; CLI for one acceleration |
| `evaluate.py` | the ablation sweep; CLI for the whole comparison |
| `noise.py` | shared vs independent noise — **the contribution** |
| `mask_fusion.py` | k-space data injection |
| `trajectory_correction.py` | look-ahead + re-noise reset |
| `schedule.py` | RePaint-style jump schedule (matches the reference implementation exactly) |
| `kspace.py` | FFT, SENSE forward/adjoint, padding/cropping, scale estimation |
| `data.py` | loads one clip from the preprocessing output |
| `prior.py` | loads a frozen `flow_prior` checkpoint |
| `utils.py` | metrics (incl. temporal flicker) and plots |
