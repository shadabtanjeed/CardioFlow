"""
Train the CardioFlow flow-matching prior on OCMR cine clips.

    python train.py                                  # uses config.yaml
    python train.py --data_dir D:/data/processed --epochs 500
    python train.py --set train.lr=1e-4 --set model.num_res_blocks=4
    python train.py --resume exps/20260916-1200      # continues an interrupted run
    python train.py --resume exps/20260916-1200 --set train.epochs=2000   # ...or extends it

Each run writes to exps/<timestamp>/ with the resolved config, checkpoints,
loss plots, and an evaluation of the best model on the val split. The test
split is deliberately not scored here -- run `evaluate.py --splits test` once
when reporting, so routine runs cannot leak it into model selection. One
epoch is one pass over the train split; `last.pt` is overwritten after every
epoch so a run can always be resumed from where it left off, and `best.pt`
tracks the lowest validation loss seen so far. Training stops either after
`train.epochs` epochs or after `train.early_stop_patience` epochs without a
validation improvement, whichever comes first.

--resume restores the model, optimizer, scaler, LR scheduler, EMA, RNG streams,
and history exactly, and works both for a run interrupted partway (crash,
preemption) and for extending a finished one via `--set train.epochs=N`: the
schedule is rebuilt from the current config on every start (see
`build_scheduler`) rather than replayed from the checkpoint.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import apply_overrides, build_parser, load_config, resolve_config, resolve_relative_paths, save_config
from dataset import build_dataset
from evaluate import build_flow, build_loader, evaluate_loss, run_evaluation
from flow import FlowMatching
from utils import EMA, plot_loss_combined, plot_loss_curve, resolve_device, save_checkpoint, set_rng_state, set_seed, write_json


def make_exp_dir(cfg: dict) -> Path:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    name = cfg['output'].get('name')
    exp_dir = Path(cfg['output']['exp_root']) / (f'{stamp}_{name}' if name else stamp)
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'plots').mkdir(parents=True, exist_ok=True)
    return exp_dir


def resolve_resume(resume_arg: Path) -> tuple[Path, Path]:
    """`resume_arg` may be an exp_dir or a checkpoint file directly."""
    if resume_arg.is_dir():
        exp_dir = resume_arg
        checkpoint_path = exp_dir / 'checkpoints' / 'last.pt'
    else:
        checkpoint_path = resume_arg
        exp_dir = checkpoint_path.parent.parent
    if not checkpoint_path.exists():
        raise SystemExit(f'--resume checkpoint not found: {checkpoint_path}')
    return exp_dir, checkpoint_path


def build_scheduler(optimizer, train_cfg: dict):
    """
    Epoch-indexed LR schedule with optional linear warmup.

    LambdaLR on purpose: `state_dict` does not serialize the lambda, so resuming
    re-derives the schedule from the *current* config instead of replaying the old
    one. That is what makes `--resume --set train.epochs=N` work; OneCycleLR bakes
    its length into the checkpoint and raises once the step count passes it.
    """
    kind = train_cfg['scheduler']
    epochs, warmup = train_cfg['epochs'], train_cfg['warmup_epochs']
    if kind == 'none' and not warmup:
        return None

    def factor(epoch: int) -> float:  # 0-indexed
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if kind == 'cosine':
            progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup))
            floor = train_cfg['lr_min_factor']
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        if kind == 'step':
            return train_cfg['lr_gamma'] ** ((epoch - warmup) // train_cfg['lr_step_epochs'])
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(flow, loader, optimizer, scaler, ema, device, use_amp, grad_accum, grad_clip,
              global_step: int, ema_start_step: int) -> tuple[list[float], int]:
    """
    One epoch of optimizer updates, grouping `grad_accum` batches per step. A
    trailing partial group (fewer than grad_accum batches left in the epoch)
    is dropped rather than applied at reduced strength, so every logged loss
    point reflects the same effective batch size.
    """
    flow.train()
    losses = []
    batch_iter = iter(loader)
    n_updates = len(loader) // grad_accum
    for _ in range(n_updates):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x1 = next(batch_iter).to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                loss = flow.loss(x1)
            scaler.scale(loss / grad_accum).backward()
            step_loss += loss.item() / grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        ema.update(flow.velocity_fn, active=global_step >= ema_start_step)
        losses.append(step_loss)
    return losses, global_step


def main() -> None:
    parser = build_parser('Train the CardioFlow flow-matching prior.')
    parser.add_argument(
        '--resume', type=Path, default=None,
        help='Resume from a checkpoint file or an exp_dir (uses its checkpoints/last.pt). '
             'Keeps that run\'s resolved config; --set/--<shortcut> overrides still apply, '
             'including raising train.epochs to train an existing run for longer.',
    )
    args = parser.parse_args()

    resuming = args.resume is not None
    if resuming:
        exp_dir, checkpoint_path = resolve_resume(args.resume)
        cfg = load_config(exp_dir / 'config.yaml')
        cfg = apply_overrides(cfg, args)
        resolve_relative_paths(cfg, Path.cwd())
    else:
        cfg = resolve_config(args)
        exp_dir = make_exp_dir(cfg)
    save_config(cfg, exp_dir / 'config.yaml')

    train_cfg = cfg['train']
    set_seed(train_cfg['seed'])
    device = resolve_device(train_cfg['device'])
    use_amp = bool(train_cfg['amp']) and device.type == 'cuda'

    train_ds = build_dataset(cfg, 'train')
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        drop_last=len(train_ds) >= train_cfg['batch_size'],
        pin_memory=device.type == 'cuda',
    )
    steps_per_epoch = len(train_loader) // train_cfg['grad_accum']
    if steps_per_epoch == 0:
        raise SystemExit(
            f'Not enough training batches ({len(train_loader)}) for grad_accum='
            f'{train_cfg["grad_accum"]}; lower grad_accum or batch_size.'
        )

    try:
        val_loader = build_loader(cfg, 'val', batch_size=train_cfg['batch_size'])
    except FileNotFoundError as e:
        print(f'[warn] no validation split ({e}); selecting the final model instead of the best.')
        val_loader = None

    flow = build_flow(cfg).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f'Run directory  : {exp_dir}')
    print(f'Device         : {device}')
    print(f'Train clips    : {len(train_ds)} ({steps_per_epoch} optimizer steps/epoch)'
          + (f' | val clips: {len(val_loader.dataset)}' if val_loader else ''))
    print(f'Parameters     : {n_params / 1e6:.2f}M')

    optimizer = torch.optim.Adam(flow.parameters(), lr=train_cfg['lr'])
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    ema = EMA(flow.velocity_fn, decay=train_cfg['ema_decay'])
    eval_flow = FlowMatching(ema.shadow, time_scale=cfg['flow']['time_scale']).to(device)

    start_epoch = 1
    global_step = 0
    best_val, best_epoch = math.inf, -1
    history: dict[str, list] = {'epoch': [], 'train_loss': [], 'val_epoch': [], 'val_loss': []}

    if resuming:
        # weights_only=False: our own trusted checkpoint carries plain-Python RNG state and a
        # config dict that `weights_only=True` (the torch>=2.6 default) rejects.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        flow.velocity_fn.load_state_dict(checkpoint['model'])
        ema.shadow.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        set_rng_state(checkpoint['rng_state'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val = checkpoint['best_val']
        best_epoch = checkpoint['best_epoch']
        history = checkpoint['history']
        print(f'Resumed from   : {checkpoint_path} (epoch {checkpoint["epoch"]} done, '
              f'best val {best_val:.5f} @ epoch {best_epoch})')

    if start_epoch > train_cfg['epochs']:
        raise SystemExit(
            f'Resumed run already completed epoch {start_epoch - 1} >= train.epochs='
            f"{train_cfg['epochs']}; nothing to do. To train it further, raise the budget: "
            f'--resume <exp_dir> --set train.epochs={train_cfg["epochs"] * 2}'
        )

    patience = train_cfg.get('early_stop_patience')
    ema_start_step = steps_per_epoch * train_cfg['ema_start_epoch']
    stopped_early = False

    epoch_bar = tqdm(range(start_epoch, train_cfg['epochs'] + 1), desc='epoch', unit='ep', dynamic_ncols=True)
    for epoch in epoch_bar:
        losses, global_step = run_epoch(
            flow, train_loader, optimizer, scaler, ema, device, use_amp,
            train_cfg['grad_accum'], train_cfg['grad_clip'], global_step, ema_start_step,
        )
        if scheduler is not None:
            scheduler.step()
        epoch_loss = float(np.mean(losses))
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_loss)

        if val_loader is not None and (epoch % train_cfg['val_every'] == 0 or epoch == train_cfg['epochs']):
            val_loss, _, _ = evaluate_loss(
                eval_flow, val_loader, device,
                t_bins=train_cfg['val_t_bins'], max_batches=train_cfg['val_batches'],
            )
            history['val_epoch'].append(epoch)
            history['val_loss'].append(val_loss)

            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                save_checkpoint(
                    exp_dir / 'checkpoints' / 'best.pt', epoch, global_step, flow.velocity_fn, ema,
                    optimizer, scaler, scheduler, best_val, best_epoch, history, cfg,
                )

        save_checkpoint(
            exp_dir / 'checkpoints' / 'last.pt', epoch, global_step, flow.velocity_fn, ema,
            optimizer, scaler, scheduler, best_val, best_epoch, history, cfg,
        )

        epoch_bar.set_postfix(
            loss=f'{epoch_loss:.4f}',
            val=f'{history["val_loss"][-1]:.4f}' if history['val_loss'] else 'n/a',
            best=f'{best_val:.4f}@{best_epoch}' if best_epoch > 0 else 'n/a',
        )

        if patience is not None and val_loader is not None and best_epoch > 0 and (epoch - best_epoch) >= patience:
            print(f'\nEarly stopping: no val improvement for {epoch - best_epoch} epochs '
                  f'(early_stop_patience={patience}).')
            stopped_early = True
            break

    # ---- loss plots -------------------------------------------------------
    plot_loss_curve(
        history['epoch'], history['train_loss'], 'Training loss (mean per epoch)',
        exp_dir / 'plots' / 'loss_train.png',
    )
    if history['val_loss']:
        plot_loss_curve(
            history['val_epoch'], history['val_loss'], 'Validation loss',
            exp_dir / 'plots' / 'loss_val.png', color='tab:red',
        )
    plot_loss_combined(
        history, exp_dir / 'plots' / 'loss_combined.png',
        best_epoch=best_epoch if best_epoch > 0 else None,
    )
    write_json(exp_dir / 'history.json', history)

    # ---- final evaluation on the best checkpoint --------------------------
    best_path = exp_dir / 'checkpoints' / 'best.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        eval_flow.velocity_fn.load_state_dict(checkpoint['ema'])
        print(f'\nBest checkpoint: epoch {checkpoint["epoch"]} (val loss {checkpoint["best_val"]:.5f})')
    else:
        print('\nNo best checkpoint recorded; evaluating the final EMA weights.')

    # Val only: the test split stays untouched by routine training runs so it does not
    # inform hyperparameter choices. Score it deliberately, once, when reporting:
    #   python evaluate.py --exp <exp_dir> --splits test
    run_evaluation(eval_flow, cfg, exp_dir, device, splits=('val',))
    print(f'\n{"Stopped early" if stopped_early else "Done"}. Everything is under {exp_dir}')
    print(f'Test split not scored. When you are ready to report: '
          f'python evaluate.py --exp {exp_dir} --splits test')


if __name__ == '__main__':
    main()
