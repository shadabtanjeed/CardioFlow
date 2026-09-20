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
  --acceleration 8 --name ktflow-v1

```

### To run the shared-vs-independent noise ablation:

```bash
cd zero_shot
python evaluate.py \
  --checkpoint_dir ../flow_prior/output/20260917-062808_prior-v1 \
  --data_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --name ktflow-v1

```

Sweeps R={8,12,16,20} x {no-correction, independent, shared}. Read `bg_flicker`
first -- PSNR/SSIM are dominated by spatial fidelity and can barely move while
flicker is obvious. Every config.yaml key is CLI-settable (`--set a.b=c`).

## Third-party repos used in this project

### `D:\Projects\CineVN-main` (Vornehm et al., MRM 2025)
Source of `preprocessing/preprocess_ocmr.py`: ISMRMRD/OCMR reading, ESPIRiT
coil sensitivities, k-space utilities (fftc/reconstruct/utils), and the GRO
retrospective undersampling mask -- all vendored (adapted, not imported) into
one self-contained file so this project doesn't depend on the sibling repo.
Also the source of the pretrained baseline checkpoints (GRO mask, R=8/12/16/20)
we compare against.
- **License: cleared for public release (2026-09-20).** `CineVN-main/LICENSE.md`
  originally stated most of the repo (including everything we vendored,
  copyright Marc Vornehm) was "currently unlicensed." Emailed him 2026-09-19
  disclosing our intended public release and listing every file we adapted;
  he replied 2026-09-20: *"Please feel free to use and modify the codebase
  for any non-commercial purposes. Redistribution of the files in the
  helpers directory is allowed without further restrictions. So you should
  be able to use any license you want for your repository if you only
  redistribute the mentioned files."* Everything we vendored (`preprocessing.py`,
  `datasets/`, `mri/`, `save.py`) lives under `src/helpers/`, so this covers
  all of it -- no remaining blocker. Keep this email if the report needs an
  attribution/permissions statement.
- Exception: the GRO mask generator specifically is under a separate OSU
  academic license (see below) -- CineVN itself vendored it from OSU-CMR.

### `github.com/OSU-CMR/GRO-CAVA` (Ahmad et al.)
Original MatLab implementation of the GRO sampling pattern, reached via
CineVN's Python port. License: free for educational/research/not-for-profit
use, provided the copyright notice + two disclaimer paragraphs + author
attribution are kept in all copies (commercial use needs OSU's Office of
Technology Commercialization). Notice reproduced in
`preprocessing/preprocess_ocmr.py` directly above `gro_sampling_pattern()`.
Note: the method itself is patent-pending (US 16/984,351) -- not a blocker
for research, but relevant if this were ever commercialized.

### `github.com/MLI-lab/pddr` (PDDR, arXiv 2607.03299)
Design reference for `flow_prior/unet.py`: their spatiotemporal U-Net
(spatial-only downsampling so the temporal axis stays full-resolution
throughout, scale-shift timestep conditioning) and training recipe
(channel_mult, lr, EMA decay, temporal-slab augmentation with circular
padding). Reimplemented from scratch, not copied -- ~150 lines vs their 1087,
diffusion swapped for flow matching. Also our closest competitor: cardiac
cine + generative prior + undersampled k-space + OCMR, but diffusion-based
inside a variational (optimization) scheme rather than gradient-free
flow-matching mask fusion. Cite the paper regardless of no code being copied.
License: BSD 2-Clause (permissive, no copyleft) -- not the constraint here,
proper attribution in the writeup is.

### `github.com/imigraz/Restora-Flow` (Hadzic et al., WACV 2026)
The base paper's official code. Reference for Phase 3 (mask fusion +
trajectory correction, to be ported from pixel space to k-t space) --
sampling/inference only, no training code, so irrelevant to Phase 2.
License not stated in the repo as of this writing.
