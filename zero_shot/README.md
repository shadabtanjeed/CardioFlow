# zero_shot — Phase 3: zero-shot k-t reconstruction

Reconstructs undersampled cine MRI by steering the trained `flow_prior` with the
measured k-space. It is Restora-Flow's algorithm (Hadzic et al., WACV 2026;
[imigraz/Restora-Flow](https://github.com/imigraz/Restora-Flow)) moved from static
pixel-space masks into **k-t space**, plus an ablation on how the trajectory-correction
noise is drawn across frames.

Nothing is trained here. The prior is loaded from a `flow_prior` run and frozen.

The adaptation works: trajectory correction is worth **+6.6 dB PSNR at R=20**. The
shared-noise idea the project set out to test does not -- see
[Ablation result](#ablation-result-shared-noise-does-not-help).

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

### The ablation

Restora-Flow restores single images, so its reset draws `randn_like(x)`. Applied
frame-by-frame to a cine clip that draws an **independent** noise field per frame. The
hypothesis this project was built on was that doing so would push each frame a different
way and make the reconstruction flicker, and that drawing **one** field and broadcasting
it across frames (`noise.py`) would fix that. Everything else is identical, which makes
the two modes a clean A/B:

```bash
python evaluate.py --checkpoint_dir ../flow_prior/output/<run> --data_dir <processed>
```

`ablation.include_no_correction` adds a `correction_steps=0` floor, so the table answers
both "does correction help at all" and "does sharing the noise help".

> **The hypothesis did not survive contact with the data.** See
> [Ablation result](#ablation-result-shared-noise-does-not-help) below before building on it.

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

### Why the temporal metrics are worth reporting

Spatial metrics can be structurally blind to temporal artefacts. Injecting noise of
identical magnitude directly into a finished clip, differing only in whether it is drawn
per-frame or shared across frames, moves SSIM by 0.002 while `bg_flicker` moves 6x:

| | SSIM | SSIM_XT | bg_flicker | ttv_ratio |
|---|---|---|---|---|
| independent | 0.3884 | 0.2982 | 0.644 | 7.96 |
| shared | 0.3902 | 0.3750 | 0.104 | 0.98 |

**This test does not describe the sampler, and it should not be read as motivation for
the shared-noise idea.** It adds noise to an already-finished image with no network in
the loop. The real algorithm injects noise at time `t` and then *denoises it* over the
remaining ODE steps (63 resets at a mean noise weight of 0.5 under the defaults), so the
artefact this table shows never reaches the output. Taking it as evidence that
per-frame noise causes flicker in the reconstruction was the project's central
methodological error -- the measured result below is the opposite.

Keep the table only for what it does establish: PSNR/SSIM alone cannot adjudicate a
temporal claim, so `bg_flicker`, `ttv_ratio` and `ssim_xt` are worth carrying.

`ssim_xt` needs at least 7 frames (SSIM's window must fit the time axis) and a `center`
landmark inside the image; it is omitted rather than faked when either is unavailable,
which is why it does not appear in short `--max_frames` smoke runs.

## Ablation result: shared noise does not help

Run `20260920-045106_ktflow-v1_ablation` -- all 38 test clips x R in {8,12,16,20} x
{no-correction, shared, independent}, `coil_mode: multicoil`, `scale_mode: auto` (no
oracle), `seed: 42`, 76 min on one GPU.

**`independent` beats `shared` on every metric at every acceleration**, including the
temporal ones the shared draw was meant to win:

| R | PSNR shared -> indep | SSIM shared -> indep | bg_flicker shared -> indep | SSIM_XT shared -> indep |
|---|---|---|---|---|
| 8 | 37.50 -> **38.89** | 0.924 -> **0.952** | 0.0820 -> **0.0797** | 0.911 -> **0.936** |
| 12 | 34.08 -> **36.12** | 0.873 -> **0.935** | 0.1007 -> **0.0825** | 0.835 -> **0.903** |
| 16 | 31.52 -> **34.29** | 0.819 -> **0.919** | 0.1259 -> **0.0884** | 0.757 -> **0.865** |
| 20 | 29.41 -> **32.57** | 0.760 -> **0.900** | 0.1560 -> **0.1045** | 0.683 -> **0.822** |

Systematic, not outlier-driven: `independent` wins SSIM on **38/38 clips at all four
accelerations**, and the gap widens with R. At R=8 the *temporal* metrics are close to a
tie (bg_flicker mean delta -0.0023, 28/38 clips); separation becomes decisive from R=12.

Three explanations were checked and ruled out:

- **Not an implementation bug.** The `[B,C,1,H,W]` draw plus `.expand()` over the frame
  axis in `noise.py` is correct, and `wiring_check.py`'s three assertions still pass.
- **Not "shared noise bakes in static hallucination".** Decomposing each arm's error
  against the reference into a frame-constant and a time-varying part gives a nearly
  identical split (R=20: shared 54.7% static, independent 58.1%). Shared's error is
  simply *larger in both components*.
- **Not *per-clip* oracle leakage.** `scale_mode: auto` throughout; verified in the
  saved config. (This run predates the fix below, though: its `auto` constants were
  measured on the test split itself -- a real, if less severe, leak. See
  [Scale calibration](#scale-calibration).)

Leading mechanism, **hypothesis only, not yet verified**: the prior trained exclusively
on `x0 = torch.randn_like(x1)`, i.i.d. across frames. A 3D U-Net's temporal convolutions
attenuate frame-i.i.d. noise but pass frame-constant noise through at full amplitude, so
a shared draw arrives at deeper layers far stronger than anything seen in training, and
the network's temporal-redundancy denoising strategy is exactly the one it defeats. That
also fits the widening gap: at higher R more of the result leans on the reset than on
measured data.

### What the run does establish

**Trajectory correction transfers to k-t space and is worth having.** At R=20 it buys
**+6.6 dB PSNR** (25.93 -> 32.57), lifts SSIM 0.683 -> 0.900 and nearly halves
background flicker (0.192 -> 0.105) versus `no-correction`. Note that `no-correction` at
R >= 12 is *worse* than the zero-filled input on `bg_flicker` (0.192 vs 0.110 at R=20):
correction is what rescues temporal behaviour, just not the sharing of its noise.

One caveat worth tracking: heart-ROI metrics lag the whole-image ones (R=20 independent:
32.57 global vs 31.54 ROI; for shared, 29.41 vs 26.73), and a few clips post
`nmse_roi > 1.0`. The background reconstructs better than the diagnostically relevant
region.

### Open threads

- **No baseline yet.** CineVN has pretrained OCMR/GRO checkpoints at the same four
  accelerations and these metrics were built to match its definitions; until that
  comparison exists the absolute numbers cannot be judged.
- **Single seed.** The 38/38 win rates make the effect unambiguous, but a seed repeat is
  cheap (~30 min for the two arms that matter at R={12,20}).
- **Mechanism unverified.** Needs a direct test of the prior's denoising behaviour on
  frame-shared vs frame-i.i.d. noise.
- A partial-correlation sweep (`x0 = sqrt(rho)*shared + sqrt(1-rho)*independent`) would
  turn the negative result into a characterisation for the cost of one sweep.
- **Rerun since the scale-calibration fix.** The table above used the old
  test-calibrated `ZERO_FILLED_SCALE` table (see below); it should be regenerated with
  the current `auto` mode before it goes in the paper. A 5-clip spot check put the new
  ratio within ~2-3% of the old one (R=8: 0.3284 vs 0.3202; R=12: 0.2630 vs 0.2587), so
  the ranking is very unlikely to move, but "very unlikely" is not "confirmed."

## Scale calibration

`data.scale_mode: auto` needs a divisor that recovers the scale `flow_prior` trained
at (`complex_std(fully_sampled_reference)`) from the measured data alone -- see
`kspace.estimate_scale`'s docstring. That divisor used to be a hardcoded
`(coil_mode, R)` lookup table in `kspace.py`, and its own comment said what it was:
*"measured on the local test split"* -- the same clips the headline numbers above are
scored on. Not the `scale_mode: reference` per-clip oracle (that was correctly never
used for a reported number), but a subtler, population-level version of the same
problem: a constant derived from the test split's ground truth, baked in, then applied
back to that split.

The fix (`calibration.py`) measures the same ratio fresh, every run, from `ocmr_val`
instead -- confirmed patient-disjoint from every `ocmr_test_gro_*` split (0 overlap
in patient IDs). `ocmr_val` is fully sampled and stores no mask, so `gro.py` generates
one on the fly with the same GRO generator the real preprocessing used. That generator
is a deterministic function of `(num_frames, num_cols, acceleration)` only -- no
randomness, no need to have run the actual ISMRMRD preprocessing pipeline -- verified
byte-identical against the stored test masks for several subjects and matrix sizes
before relying on it. `data.py::load_clip` and `find_files` grew a `split`/`acceleration`
path so the same loading and masking code serves both `ocmr_test_gro_*` (mask already
stored) and `ocmr_val` (mask generated) without duplication.

Point `--val_data_dir` at a root containing `ocmr_val/` and `coil_sens/` (can be the
same root as `--data_dir`, or a separate, smaller one -- see
[What a phase-3 data folder needs](#what-a-phase-3-data-folder-needs)). `sampler.py`
calibrates once per run; `evaluate.py` calibrates once per distinct acceleration in the
sweep and reuses it across arms, since the ratio depends on `(acceleration, coil_mode)`
only. The resolved value and how many val clips it came from are both printed and
recorded in the run's `metrics.json`/`ablation.json`, so every reported number carries
its own calibration provenance instead of a shared, silent constant.

## Usage

```bash
# one acceleration, whole split -- --val_data_dir is only read for scale_mode: auto
# calibration (ocmr_val), --data_dir is the test split that actually gets scored
python sampler.py --checkpoint_dir ../flow_prior/output/<run> \
    --data_dir <processed> --val_data_dir <processed> --acceleration 8

# the full ablation sweep (R x noise mode), one prior load, one calibration per R
python evaluate.py --checkpoint_dir ../flow_prior/output/<run> \
    --data_dir <processed> --val_data_dir <processed>
```

`--data_dir` and `--val_data_dir` can point at the same root (if it has both
`ocmr_test_gro_*/` and `ocmr_val/`) or at two different ones -- see
[What a phase-3 data folder needs](#what-a-phase-3-data-folder-needs) for the minimal
folder that has exactly what both need and nothing else. Every value in `config.yaml`
is settable from the CLI, exactly as in `flow_prior`: short flags for the common ones
(`--acceleration`, `--ode_steps`, `--correction_noise`, `--limit`, `--calib_limit`,
`--device`, ...) and `--set any.nested.key=value` for the rest.

## What a phase-3 data folder needs

Phase 3 only ever touches three things under a processed-data root: `ocmr_val/`
(calibration only, via `--val_data_dir`), `ocmr_test_gro_{08,12,16,20}/` (the split
actually reconstructed and scored, via `--data_dir`), and `coil_sens/<id>/` for
whichever subject IDs appear in those two splits. It never reads `ocmr_train`,
`ocmr_recons`, or `logs` — those exist only for `flow_prior` training and its own
sanity-checking. A folder that has exactly this (see `OCMR_data_processed_phase3`
next to the full `OCMR_data_processed`) is enough to run `sampler.py`/`evaluate.py`
end to end, at roughly half the size of the full preprocessing output.

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

## Figures

`figure.py` draws the undersampled / reconstruction / ground-truth comparison from a
finished sweep. It reads the reconstructions already on disk and recomputes the
reference and zero-filled input from the dataset, so it needs no GPU and no prior.

```bash
# undersampled -> reconstruction -> ground truth, with a y-t profile column
python figure.py --ablation_dir <sweep> --data_dir <processed> \
    --clip fs_0040_3T_slice00 --acceleration 12

# any rows, any order: 'zero_filled', 'reference', or any arm name
python figure.py ... --order zero_filled,no-correction,shared,independent,reference

# error maps under each reconstruction row
python figure.py ... --error_maps --error_gain 4
```

One display choice worth stating in a caption: `A^H y` is the adjoint, not the inverse,
so its overall brightness is arbitrary and it renders nearly black at the reference's
window -- which is why the zero-filled row is barely visible in the sampler's own
per-clip PNGs. `--zf_scale fit` (the default) fits a single least-squares scalar to the
reference so the row shows aliasing *structure* rather than a brightness mismatch;
`--zf_scale none` restores the raw adjoint scale.

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
- **The training scale is recoverable, but not with one constant, and not from test.**
  The prior trained on std-normalised clips; that std does not exist at inference, and a
  single pooled constant across R is off by up to 20% (the ratio falls steadily with R
  and differs by coil mode) — so it is calibrated per `(coil_mode, R)`, fresh from
  `ocmr_val`, every run. See [Scale calibration](#scale-calibration).
  `data.scale_mode=reference` is a separate, per-clip oracle escape hatch for telling
  "the scale estimate is off" apart from "the sampler is wrong" — never used for a
  reported number.
- **`ocmr_val`'s `kspace` is fully sampled, unlike the test folders' — no stored `mask`.**
  It was only ever prepped as flow-prior training/validation data. `gro.py` masks it
  on the fly for calibration; nothing in phase 3 reconstructs or scores it.

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
| `noise.py` | shared vs independent noise — **the variable the ablation flips** |
| `mask_fusion.py` | k-space data injection |
| `trajectory_correction.py` | look-ahead + re-noise reset |
| `schedule.py` | RePaint-style jump schedule (matches the reference implementation exactly) |
| `kspace.py` | FFT, SENSE forward/adjoint, padding/cropping, scale estimation |
| `gro.py` | GRO undersampling mask, generated on the fly (used to mask `ocmr_val` for calibration) |
| `calibration.py` | measures `data.scale_mode: auto`'s scale ratio from `ocmr_val`, on the fly |
| `data.py` | loads one clip from the preprocessing output (`ocmr_test_gro_*`, already masked, or `ocmr_val`, masked on the fly) |
| `prior.py` | loads a frozen `flow_prior` checkpoint |
| `utils.py` | metrics (incl. temporal flicker) and plots |
| `figure.py` | publication figures from a finished sweep (no GPU, no prior needed) |
