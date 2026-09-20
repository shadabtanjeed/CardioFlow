# Credits

CardioFlow builds directly on the following work. This file exists so every
reused piece of code, architecture, or data is traceable to its source and
license — please cite the papers below if you build on this repository.

## Base paper

**Restora-Flow** — Hadzic et al., WACV 2026.
Zero-shot image restoration via mask fusion + trajectory correction on a
pretrained flow-matching prior. CardioFlow extends its algorithm from static
pixel-space masking to k-t space masking for dynamic cardiac cine MRI (see
`zero_shot/`).
- Code: [imigraz/Restora-Flow](https://github.com/imigraz/Restora-Flow) (sampling/inference only, no training code)
- Used for: the mask-fusion + trajectory-correction algorithm structure,
  reimplemented for k-t space in `zero_shot/mask_fusion.py` and
  `zero_shot/trajectory_correction.py`; the RePaint-style jump schedule in
  `zero_shot/schedule.py` matches their reference implementation exactly.
- License: not stated in the repository as of this writing.

## Data pipeline

**CineVN** — Vornehm et al., *"CineVN: Variational network reconstruction for
rapid functional cardiac cine MRI,"* Magnetic Resonance in Medicine, 2025.
Source of `preprocessing/preprocess_ocmr.py`: ISMRMRD/OCMR reading, ESPIRiT
coil-sensitivity estimation, k-space utilities (FFT, phase padding, readout
oversampling removal, POCS), and reference reconstruction (RSS +
sensitivity-weighted). Also the source of the pretrained baseline checkpoints
(GRO mask, R=8/12/16/20) used for comparison.
- Repo: `D:\Projects\CineVN-main` (not included in this repository)
- Used for: `preprocessing/preprocess_ocmr.py` is a self-contained adaptation
  of `src/helpers/preprocessing.py`, `src/helpers/datasets/`,
  `src/helpers/mri/`, and `src/helpers/save.py`, with the S3 auto-download
  removed (this project uses a local pre-extracted dataset copy) and the
  BART/PICS compressed-sensing reconstruction path dropped (unused here).
- License: these files were originally unlicensed; **written permission for
  non-commercial redistribution was obtained from the author on 2026-09-20.**

**GRO (Golden Ratio offset) sampling pattern** — Ahmad, Jin, Simonetti, Liu,
Rich. *"Cartesian sampling for dynamic magnetic resonance imaging (MRI),"*
U.S. Patent Application No. 16/984,351 (pub. 2021-02-04).
Retrospective undersampling mask generator used for training/eval acceleration
rates R=8/12/16/20.
- Original implementation: [OSU-CMR/GRO-CAVA](https://github.com/OSU-CMR/GRO-CAVA)
  (MatLab), reached via CineVN's Python port
  (`src/sampling_patterns/dynamic/gro.py`)
- Used for: `gro_sampling_pattern()` and `GROParam` in
  `preprocessing/preprocess_ocmr.py`
- License: Ohio State University academic license — free for educational,
  research, and not-for-profit use with attribution retained (full notice is
  in the source file above `gro_sampling_pattern()`). Note the underlying
  method is patent-pending; the patent does not restrict research use.
- Author: Rizwan Ahmad (ahmad.46@osu.edu)

**`read_ocmr`** — reference ISMRMRD reader from the official OCMR repository,
modified by Chong Chen (Chong.Chen@osumc.edu), itself based on
[ismrmrd/ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools)'s
`recon_ismrmrd_dataset.py`. Used in the EDA notebook (`eda/read_ocmr.py`,
`eda/example_ocmr.ipynb`) only — the preprocessing pipeline uses CineVN's
independent ISMRMRD reader instead.

## Model architecture reference

**PDDR** — *"Piecewise Dynamic Diffusion Regularization for Reconstruction of
Cardiac Cine MRI,"* arXiv:2607.03299.
Design reference for `flow_prior/unet.py`: the spatial-only downsampling
pattern (temporal axis stays full-resolution through the whole network),
scale-shift timestep conditioning, and training recipe (channel multipliers,
learning rate, EMA decay, temporal-slab augmentation with circular padding
around the periodic cardiac cycle).
- Code: [MLI-lab/pddr](https://github.com/MLI-lab/pddr)
- Used for: architectural and hyperparameter reference only —
  `flow_prior/unet.py` is an independent ~150-line reimplementation (their
  U-Net is 1087 lines), and the diffusion objective is replaced with flow
  matching throughout. No code was copied.
- License: BSD 2-Clause.
- Also our closest related work: cardiac cine reconstruction with a
  generative spatiotemporal prior on OCMR/CMRxRecon, but diffusion-based
  inside a variational (optimization) reconstruction scheme, rather than
  this project's gradient-free flow-matching mask fusion.

## Dataset

**OCMR** — Chen, C., Liu, Y., Schniter, P., Tong, M., Zareba, K., Simonetti,
O., Potter, L., Ahmad, R. *"OCMR (v1.0) — Open-Access Multi-Coil k-Space
Dataset for Cardiovascular Magnetic Resonance Imaging,"* arXiv:2008.03410,
2020.
- Dataset site: [ocmr.info](https://ocmr.info)
- Not included in this repository (see `.gitignore`) — must be obtained
  separately.

## Flow matching background

**Flow Matching for Generative Modeling** — Lipman et al., ICLR 2023, and
[facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
(CC BY-NC). Conceptual/notational reference for the training objective in
`flow_prior/flow.py`; no code was used from this repository.
