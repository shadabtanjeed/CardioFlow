"""
Config loading for the zero-shot sampler: YAML is the source of truth, CLI overrides it.

Same contract as flow_prior/config.py, so the two halves of the project are driven
the same way:
    python sampler.py --acceleration 12 --correction_noise shared
    python sampler.py --set sampler.ode_steps=128 --set data.limit=5
Every entry in config.yaml is reachable through --set, and the ones tweaked most
often also have a short flag.

Two flags are easy to mix up:
    --checkpoint_dir  -- INPUT.  A flow_prior run directory (its checkpoints/best.pt
                         is what gets loaded). Read-only; never written to.
    --output          -- OUTPUT. Where THIS run's own timestamped result folders
                         get created. On Kaggle: --checkpoint_dir under
                         /kaggle/input/..., --output under /kaggle/working/....
--checkpoint (no _dir) is a third, unrelated flag: which *file* inside
--checkpoint_dir to load ('best.pt' or 'last.pt').
"""

import argparse
import ast
import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).parent / 'config.yaml'

# Flags whose value is always a literal string, never parsed as a Python literal.
RAW_STRING_FLAGS = {
    'data_dir', 'output', 'name', 'device', 'checkpoint_dir', 'checkpoint', 'weights',
    'coil_mode', 'correction_noise', 'fusion_noise', 'init_noise', 'scale_mode',
}

# CLI flag -> dotted config path.
SHORTCUTS = {
    'data_dir': 'data.data_dir',
    'acceleration': 'data.acceleration',
    'limit': 'data.limit',
    'coil_mode': 'data.coil_mode',
    'max_frames': 'data.max_frames',
    'scale_mode': 'data.scale_mode',
    'checkpoint_dir': 'prior.checkpoint_dir',
    'checkpoint': 'prior.checkpoint',
    'weights': 'prior.weights',
    'ode_steps': 'sampler.ode_steps',
    'correction_steps': 'sampler.correction_steps',
    'correction_noise': 'sampler.correction_noise',
    'fusion_noise': 'sampler.fusion_noise',
    'init_noise': 'sampler.init_noise',
    'final_dc': 'sampler.final_dc',
    'seed': 'sampler.seed',
    'device': 'sampler.device',
    'amp': 'sampler.amp',
    'name': 'output.name',
    'output': 'output.dir',
    'save_clips': 'output.save_clips',
    'save_arrays': 'output.save_arrays',
    'save_array_inputs': 'output.save_array_inputs',
}


def _parse_value(value: str):
    """'1e-4' -> float, '[128, 128]' -> list, 'true' -> bool, anything else -> str."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered in ('true', 'false'):
            return lowered == 'true'
        if lowered in ('null', 'none'):
            return None
        return value


def set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split('.')
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split('.'):
        node = node[key]
    return node


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Path to config YAML.')
    parser.add_argument(
        '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
        help='Override any config entry, e.g. --set sampler.ode_steps=128 (repeatable).',
    )
    for flag, dotted in SHORTCUTS.items():
        parser.add_argument(f'--{flag}', default=None, help=f'Overrides {dotted}')
    parser.add_argument('--crop', nargs=2, type=int, default=None,
                        help='Overrides data.crop (H W); combined coil_mode only.')
    return parser


def resolve_relative_paths(cfg: dict, base: Path) -> None:
    """Anchor relative paths to the config file's directory, not the shell's cwd."""
    for dotted in ('data.data_dir', 'output.dir', 'prior.checkpoint_dir'):
        try:
            value = get_by_path(cfg, dotted)
        except KeyError:
            continue
        if value and not Path(value).is_absolute():
            set_by_path(cfg, dotted, str((base / value).resolve()))


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for flag, dotted in SHORTCUTS.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        if flag in RAW_STRING_FLAGS or not isinstance(value, str):
            set_by_path(cfg, dotted, value)
        else:
            set_by_path(cfg, dotted, _parse_value(value))

    if getattr(args, 'crop', None):
        set_by_path(cfg, 'data.crop', list(args.crop))

    for override in args.overrides:
        if '=' not in override:
            raise ValueError(f'Malformed --set {override!r}, expected KEY=VALUE')
        key, _, value = override.partition('=')
        set_by_path(cfg, key.strip(), _parse_value(value.strip()))

    return cfg


def validate(cfg: dict) -> dict:
    """Fail fast on combinations that would otherwise go wrong deep inside a run."""
    from noise import MODES

    sampler_cfg, data_cfg = cfg['sampler'], cfg['data']
    for key in ('correction_noise', 'fusion_noise', 'init_noise'):
        if sampler_cfg[key] not in MODES:
            raise SystemExit(f'sampler.{key} must be one of {MODES}, got {sampler_cfg[key]!r}.')
    if data_cfg['coil_mode'] not in ('multicoil', 'combined'):
        raise SystemExit(f'data.coil_mode must be multicoil or combined, got {data_cfg["coil_mode"]!r}.')
    if data_cfg['crop'] and data_cfg['coil_mode'] != 'combined':
        raise SystemExit(
            'data.crop requires data.coil_mode=combined -- an image-domain crop is '
            'inconsistent with a k-space mask over the full FOV.'
        )
    if not cfg['prior']['checkpoint_dir']:
        raise SystemExit(
            'prior.checkpoint_dir is required: point it at a flow_prior run directory '
            '(--checkpoint_dir).'
        )
    return cfg


def resolve_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args)
    resolve_relative_paths(cfg, Path(args.config).resolve().parent)
    return validate(cfg)


def save_config(cfg: dict, path: Path) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
