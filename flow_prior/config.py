"""Config loading: YAML file as the source of truth, CLI flags override it."""

import argparse
import ast
import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).parent / 'config.yaml'

# Flags whose value is always a literal string, never parsed as a Python literal
# (a directory called "2024" must stay a string, not become an int).
RAW_STRING_FLAGS = {'data_dir', 'exp_root', 'name', 'device'}

# CLI flag -> dotted config path, for the settings tweaked most often.
SHORTCUTS = {
    'data_dir': 'data.data_dir',
    'frames': 'data.frames',
    'num_workers': 'data.num_workers',
    'epochs': 'train.epochs',
    'batch_size': 'train.batch_size',
    'lr': 'train.lr',
    'val_every': 'train.val_every',
    'seed': 'train.seed',
    'device': 'train.device',
    'name': 'output.name',
    'exp_root': 'output.exp_root',
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
        help='Override any config entry, e.g. --set train.lr=1e-4 (repeatable).',
    )
    for flag in SHORTCUTS:
        parser.add_argument(f'--{flag}', default=None, help=f'Overrides {SHORTCUTS[flag]}')
    parser.add_argument('--crop', nargs=2, type=int, default=None, help='Overrides data.crop (H W).')
    return parser


def resolve_relative_paths(cfg: dict, base: Path) -> None:
    """
    Anchor relative paths to `base` (normally the config file's directory, not
    the working directory), so a run behaves the same however it was launched.
    """
    for dotted in ('data.data_dir', 'output.exp_root'):
        try:
            value = get_by_path(cfg, dotted)
        except KeyError:
            continue
        if value and not Path(value).is_absolute():
            set_by_path(cfg, dotted, str((base / value).resolve()))


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Applies --<shortcut>, --crop, and --set KEY=VALUE CLI overrides to a loaded config."""
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


def resolve_config(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args)
    resolve_relative_paths(cfg, Path(args.config).resolve().parent)
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
