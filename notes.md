### To start preprocessing:

```bash
python preprocess_ocmr.py \
  --raw_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data \
  --target_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --csv_query "smp=='fs'" --accelerations 8 12 16 20 --mask_types gro

```

### To start training:

```bash
cd flow_prior
python train.py --num_workers 4 --name prior-v1

```

### To run zero-shot reconstruction (Phase 3):

```bash
cd zero_shot
python sampler.py \
  --checkpoint_dir ../flow_prior/output/20260917-062808_prior-v1 \
  --data_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --val_data_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --acceleration 8 --name ktflow-v1

```

### To run the shared-vs-independent noise ablation:

```bash
cd zero_shot
python evaluate.py \
  --checkpoint_dir ../flow_prior/output/20260917-062808_prior-v1 \
  --data_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --val_data_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --name ktflow-v1

```

Sweeps R={8,12,16,20} x {no-correction, independent, shared}, 38 clips each.
76 min on one GPU, ~5 GB of output at the default save settings. Every
config.yaml key is CLI-settable (`--set a.b=c`).

Needs the test split, `ocmr_val` (calibration only -- see below, never scored),
and the coil sensitivities for both -- not `ocmr_train` or `ocmr_recons`. See
`zero_shot/README.md#what-a-phase-3-data-folder-needs`;
`F:\Dataset\OCMR\OCMR_data_processed_phase3` is exactly that subset.

**`--data_dir` and `--val_data_dir` are two different roles, even when you pass the
same path for both.** `--data_dir` is the split that gets reconstructed and scored.
`--val_data_dir` is read only to calibrate `data.scale_mode: auto`'s zero-filled scale
constant, from `ocmr_val` -- patient-disjoint from every `ocmr_test_gro_*` split -- and
is never reconstructed or scored itself. Before 2026-09-21 this constant
(`kspace.ZERO_FILLED_SCALE`) was a hardcoded table measured on the test split itself,
which is a real (population-level, not per-clip) leak; `zero_shot/calibration.py` now
recomputes it fresh from val on every run instead. See
`zero_shot/README.md#scale-calibration`.

### To draw the figures:

```bash
cd zero_shot
python figure.py \
  --ablation_dir <sweep dir> \
  --data_dir <processed> \
  --clip fs_0040_3T_slice00 --acceleration 12
```

No GPU and no prior needed -- it reads the sweep's saved `.h5` reconstructions.
`--order` takes any row sequence, `--error_maps` adds error rows.

### Result so far (run `20260920-045106_ktflow-v1_ablation`)

**The shared-noise hypothesis was refuted.** `independent` beats `shared` on every
metric at every acceleration -- including `bg_flicker` and `ssim_xt`, the temporal
metrics the shared draw was supposed to win -- and wins SSIM on 38/38 clips at all
four R. The gap widens with R (PSNR +1.4 dB at R=8, +3.2 dB at R=20).

What *did* work is the adaptation itself: trajectory correction in k-t space is
worth **+6.6 dB PSNR at R=20** over no correction, and nearly halves background
flicker. That is the result to build on.

Do not trust the old synthetic justification test (the `bg_flicker` 0.644 vs 0.104
table): it added noise to a finished clip with no network in the loop, whereas the
real sampler denoises the injected noise over the remaining ODE steps. See
`zero_shot/README.md` for the full table, the three explanations that were ruled
out, and the leading (still unverified) mechanism.

Next: a CineVN baseline on the same 38 clips -- without it the absolute numbers
cannot be judged.

## Licensing and credits

Repo license: CC BY-NC 4.0 (see `LICENSE`). Full attribution for every
third-party repo/paper used, per-file provenance, and license terms are in
`CREDITS.md` -- including confirmation (2026-09-20) that non-commercial
redistribution of the CineVN-derived files in
`preprocessing/preprocess_ocmr.py` was cleared with the author.
