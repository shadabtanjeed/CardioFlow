"""
Cine clips from the preprocessing output (see preprocessing/preprocess_ocmr.py).

Expected layout under `data_dir`:
    ocmr_train/<subject>_sliceNN.h5
    ocmr_val/<subject>_sliceNN.h5
    ocmr_test_gro_08/<subject>_sliceNN.h5   (and _12, _16, _20)

Each file stores `reconstruction_weighted` as [slice=1, frame, x, y] complex64,
plus an `abs_max` attribute. Only the coil-combined reference image is needed
to train the prior -- k-space and masks stay unused until the Phase 3 zero-shot
sampler.

`normalize` picks what each clip is divided by, which sets the data scale
relative to the unit-variance noise the flow interpolates against. See
`_scale` -- this choice materially changes training dynamics.
"""

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class OCMRCineDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        frames: int = 12,
        crop: tuple[int, int] = (128, 128),
        key: str = 'reconstruction_weighted',
        normalize: str = 'max',
        normalize_percentile: float = 99.0,
        random_temporal_start: bool = False,
        random_crop: bool = False,
        crop_jitter: int = 16,
        crop_heart_prob: float = 0.5,
        flip: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.frames = frames
        self.crop = tuple(crop)
        self.key = key
        self.normalize = normalize
        self.normalize_percentile = normalize_percentile
        self.random_temporal_start = random_temporal_start
        self.random_crop = random_crop
        self.crop_jitter = crop_jitter
        self.crop_heart_prob = crop_heart_prob
        self.flip = flip
        self.seed = seed

        self.files = self._collect_files()
        if not self.files:
            raise FileNotFoundError(
                f'No .h5 files found for split {split!r} under {self.data_dir}. '
                f'Run preprocessing/preprocess_ocmr.py first.'
            )

    def _collect_files(self) -> list[Path]:
        if self.split == 'test':
            # Every ocmr_test_gro_XX folder holds the same fully-sampled reference
            # per slice (only the mask differs), so dedupe by filename to avoid
            # counting each clip once per acceleration rate.
            dirs = sorted(self.data_dir.glob('ocmr_test_*'))
            seen: dict[str, Path] = {}
            for d in dirs:
                for f in sorted(d.glob('*.h5')):
                    seen.setdefault(f.name, f)
            return [seen[k] for k in sorted(seen)]
        return sorted((self.data_dir / f'ocmr_{self.split}').glob('*.h5'))

    def __len__(self) -> int:
        return len(self.files)

    @property
    def _augmenting(self) -> bool:
        return self.random_temporal_start or self.random_crop or self.flip

    def _rng(self, idx: int):
        # Augmented: the global RNG, which set_seed() controls and which DataLoader
        # reseeds per worker -- so runs stay reproducible either way.
        # Not augmented: a fixed per-clip stream, so val/test clips never change.
        return random if self._augmenting else random.Random(self.seed + idx)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        with h5py.File(path, 'r') as hf:
            image = np.asarray(hf[self.key][0])  # [frame, x, y]
            abs_max = float(hf.attrs['abs_max']) if 'abs_max' in hf.attrs else None
            center = self._anchor(hf.attrs)

        image = image / self._scale(image, abs_max)

        rng = self._rng(idx)

        # temporal slab, wrapping around the (periodic) cardiac cycle
        n_frames = image.shape[0]
        start = rng.randrange(n_frames) if self.random_temporal_start else 0
        image = image[(start + np.arange(self.frames)) % n_frames]

        clip = (
            np.stack([image.real, image.imag], axis=0)
            if np.iscomplexobj(image)
            else image[None]
        ).astype(np.float32)

        clip = self._crop_or_pad(clip, rng, center)

        if self.flip:
            if rng.random() < 0.5:
                clip = clip[..., ::-1]
            if rng.random() < 0.5:
                clip = clip[..., ::-1, :]

        return torch.from_numpy(np.ascontiguousarray(clip))

    @staticmethod
    def _anchor(attrs) -> tuple[int, int] | None:
        """
        Where to centre the crop: the middle of the annotated heart bounding box.

        Preferred over the `center` attribute, which is an anatomical landmark (the LV
        centre, what CineVN uses for its x-t profile crosshairs) sitting a median 10px
        and up to 22px away from the bbox centre. Anchoring on the bbox instead raises
        the share of jittered crops containing the whole heart from 84% to 88%.
        """
        if 'bbox' in attrs:
            x_low, y_low, x_high, y_high = (int(v) for v in attrs['bbox'])
            return (x_low + x_high) // 2, (y_low + y_high) // 2
        if 'center' in attrs:
            return tuple(int(v) for v in attrs['center'])  # type: ignore[return-value]
        return None

    def _scale(self, image: np.ndarray, abs_max: float | None) -> float:
        """
        Divisor that puts a clip on the scale the prior trains at. Computed over
        the whole clip before cropping, so it is a per-file constant rather than
        something the random crop shifts around.

        Flow matching interpolates the data against x0 ~ N(0, 1), so this choice
        decides where signal overtakes noise along the trajectory: at
        t = 1/(1 + std(data)).
          'max'        -- the brightest pixel (the stored abs_max). Cine images are
                          mostly dark background, so std lands around 0.1-0.25 and
                          ~85% of the trajectory is noise-dominated.
          'std'        -- unit variance, putting the crossover near t = 0.5.
          'percentile' -- robust alternative to 'max', ignoring the brightest outliers.
        'std' and 'percentile' are also computable from an undersampled
        reconstruction, which 'max' via the stored attribute is not -- the Phase 3
        sampler has no fully-sampled reference to read abs_max from.
        """
        magnitude = np.abs(image)
        if self.normalize == 'max':
            scale = abs_max if abs_max else float(magnitude.max())
        elif self.normalize == 'std':
            components = np.stack([image.real, image.imag]) if np.iscomplexobj(image) else magnitude
            scale = float(components.std())
        elif self.normalize == 'percentile':
            scale = float(np.percentile(magnitude, self.normalize_percentile))
        else:
            raise ValueError(
                f'Unknown data.normalize {self.normalize!r}; expected max, std, or percentile.'
            )
        return scale or 1.0

    def _crop_or_pad(self, clip: np.ndarray, rng, center: tuple[int, int] | None = None) -> np.ndarray:
        """
        clip: [C, F, H, W] -> [C, F, crop_h, crop_w], zero-padding when too small.

        Training crops are a mixture: with probability `crop_heart_prob` the window
        anchors on the heart centre (see `_anchor`, plus
        +/-`crop_jitter` px of translation), otherwise it lands uniformly anywhere in
        the FOV. Both halves matter. A uniform 128x128 window over OCMR's 256x208 FOV
        keeps the whole heart only 9-29% of the time (measured on the pilot slices),
        so pure uniform sampling wastes most of a ~194-clip pool; but the Phase 3
        sampler runs at full FOV, so a prior that has only seen heart-centred crops
        would never have learned chest wall, lung or background appearance.

        Val/test crops are always heart-centred and deterministic, so the metric does
        not wander. Files without a `center` attribute fall back to uniform-random
        (train) or image-centre (val/test) cropping.
        """
        target_h, target_w = self.crop
        _, _, h, w = clip.shape

        pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
        if pad_h or pad_w:
            clip = np.pad(
                clip,
                ((0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            )
            _, _, h, w = clip.shape
            if center is not None:
                center = (center[0] + pad_h // 2, center[1] + pad_w // 2)

        anchor_on_heart = center is not None and (
            not self.random_crop or rng.random() < self.crop_heart_prob
        )
        if anchor_on_heart:
            top, left = center[0] - target_h // 2, center[1] - target_w // 2
            if self.random_crop and self.crop_jitter:
                top += rng.randint(-self.crop_jitter, self.crop_jitter)
                left += rng.randint(-self.crop_jitter, self.crop_jitter)
        elif self.random_crop:
            top, left = rng.randint(0, h - target_h), rng.randint(0, w - target_w)
        else:
            top, left = (h - target_h) // 2, (w - target_w) // 2

        top = int(np.clip(top, 0, h - target_h))
        left = int(np.clip(left, 0, w - target_w))
        return clip[:, :, top : top + target_h, left : left + target_w]


def build_dataset(cfg: dict, split: str) -> OCMRCineDataset:
    data_cfg = cfg['data']
    augment = data_cfg.get('augment', {}) if split == 'train' else {}
    return OCMRCineDataset(
        data_dir=data_cfg['data_dir'],
        split=split,
        frames=data_cfg['frames'],
        crop=data_cfg['crop'],
        key=data_cfg['key'],
        normalize=data_cfg['normalize'],
        normalize_percentile=data_cfg.get('normalize_percentile', 99.0),
        random_temporal_start=augment.get('random_temporal_start', False),
        random_crop=augment.get('random_crop', False),
        crop_jitter=augment.get('crop_jitter', 16),
        crop_heart_prob=augment.get('crop_heart_prob', 0.5),
        flip=augment.get('flip', False),
        seed=cfg['train']['seed'],
    )
