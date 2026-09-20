"""
Loading the trained flow-matching prior from a flow_prior run.

flow_prior/ uses flat module imports (`from unet import UNet3D`), so it goes on
sys.path rather than being imported as a package. Nothing is duplicated here: the
network and the flow-matching wrapper are exactly the ones that were trained.
"""

import sys
from pathlib import Path

import torch

FLOW_PRIOR_DIR = Path(__file__).resolve().parent.parent / 'flow_prior'
# Appended, never prepended: flow_prior has its own `utils` and `config` modules
# that would otherwise shadow this package's. The two files actually imported here
# (flow.py, unet.py) depend on nothing but torch, so there is no deeper collision.
if str(FLOW_PRIOR_DIR) not in sys.path:
    sys.path.append(str(FLOW_PRIOR_DIR))

from flow import FlowMatching  # noqa: E402
from unet import UNet3D  # noqa: E402


def resolve_checkpoint(checkpoint_dir: Path, checkpoint: str = 'best.pt') -> Path:
    path = Path(checkpoint_dir) / 'checkpoints' / checkpoint
    if not path.exists():
        available = sorted(p.name for p in (Path(checkpoint_dir) / 'checkpoints').glob('*.pt'))
        raise SystemExit(f'{path} not found. Available in this run: {available or "none"}')
    return path


def load_prior(
    checkpoint_dir: Path,
    device: torch.device,
    checkpoint: str = 'best.pt',
    weights: str = 'ema',
) -> tuple[FlowMatching, dict]:
    """
    Returns the prior in eval mode plus the config it was trained with.

    `weights='ema'` matches how flow_prior/evaluate.py scores a run; 'model' loads
    the raw (non-averaged) weights instead.

    weights_only=False: this is our own checkpoint, and it carries a config dict
    and plain-Python RNG state that torch>=2.6's default rejects.
    """
    if weights not in ('ema', 'model'):
        raise ValueError(f"Unknown prior.weights {weights!r}; expected 'ema' or 'model'.")

    path = resolve_checkpoint(checkpoint_dir, checkpoint)
    state = torch.load(path, map_location=device, weights_only=False)
    train_cfg = state['config']

    net = UNet3D(**train_cfg['model'])
    net.load_state_dict(state[weights])

    flow = FlowMatching(net, time_scale=train_cfg['flow']['time_scale']).to(device)
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    info = {
        'checkpoint': str(path),
        'epoch': state['epoch'],
        'best_val': float(state['best_val']),
        'weights': weights,
        'train_normalize': train_cfg['data']['normalize'],
        'train_crop': train_cfg['data']['crop'],
        'train_frames': train_cfg['data']['frames'],
        # H and W must be divisible by this for the U-Net's spatial downsampling;
        # the sampler pads around each network call to satisfy it.
        'downsample_multiple': 2 ** (len(train_cfg['model']['channel_mult']) - 1),
    }
    return flow, info
