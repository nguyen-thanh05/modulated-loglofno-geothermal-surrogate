"""Autoregressive single-trajectory evaluation for heterogeneous models."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from losses.functional_losses import LpLoss
from training.constants import CHANNEL_NAMES, WELL_COORDS
from training.dataset import ARDataset
from training.model_adapters import create_adapter
from training.model_factory import create_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = (5, 42, 2026)
SLICE_SPECS = (
    ('d10', 'depth=10', lambda vol: vol[10, :, :]),
    ('h15', 'H=15', lambda vol: vol[:, 15, :]),
    ('w4', 'W=4', lambda vol: vol[:, :, 4]),
)
SNAPSHOT_TIMES = (0, 50, 100, 155)


def _inject_seed_into_paths(cfg, seed):
    ckpt = cfg['checkpoints']
    seed_dir = f'seed{seed}'

    ckpt['running_dir'] = os.path.join(ckpt['running_dir'], seed_dir, '')

    final_dir, final_name = os.path.split(ckpt['final_path'])
    ckpt['final_path'] = os.path.join(final_dir, seed_dir, final_name)

    if 'resume_path' in ckpt:
        resume_name = os.path.basename(ckpt['resume_path'])
        ckpt['resume_path'] = os.path.join(ckpt['running_dir'], resume_name)


def _resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _parse_seeds(raw):
    seeds = [int(s.strip()) for s in raw.split(',') if s.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError('At least one seed is required')
    return seeds


def _mask_wells(action):
    """Zero action at well cells for depth slices 0:2. action: (B, D, H, W)."""
    for well in WELL_COORDS:
        action[:, 0:2, well[0], well[1]] = 0.


def load_hetero_dataset(data_path, k_max=10):
    data_path = str(_resolve_path(data_path))
    mode = 'r'
    dataset = ARDataset(
        np.load(os.path.join(data_path, 'all_temp_formation.npy'), mmap_mode=mode),
        np.load(os.path.join(data_path, 'all_temp_frac.npy'), mmap_mode=mode),
        np.load(os.path.join(data_path, 'all_pres_formation.npy'), mmap_mode=mode),
        np.load(os.path.join(data_path, 'all_pres_frac.npy'), mmap_mode=mode),
        np.load(os.path.join(data_path, 'all_action.npy'), mmap_mode=mode),
        k_max=k_max,
        heterogeneous=True,
        por_matrix=np.load(os.path.join(data_path, 'all_por_matrix.npy'), mmap_mode=mode),
        por_frac=np.load(os.path.join(data_path, 'all_por_frac.npy'), mmap_mode=mode),
        perm_matrix=np.load(os.path.join(data_path, 'all_perm_matrix.npy'), mmap_mode=mode),
        perm_frac=np.load(os.path.join(data_path, 'all_perm_frac.npy'), mmap_mode=mode),
    )
    return dataset


def load_model(cfg, seed, device):
    cfg_seed = copy.deepcopy(cfg)
    _inject_seed_into_paths(cfg_seed, seed)
    final_path = _resolve_path(cfg_seed['checkpoints']['final_path'])
    if not final_path.is_file():
        raise FileNotFoundError(f'Missing checkpoint for seed {seed}: {final_path}')

    model_cfg = cfg_seed['model']
    model_type = model_cfg['type']
    model = create_model(model_cfg, model_type).to(device)
    adapter = create_adapter(model_type, heterogeneous=True)

    ckpt = torch.load(final_path, map_location=device, weights_only=False)
    if 'ema_model' not in ckpt:
        raise KeyError(f'Checkpoint {final_path} has no ema_model weights')
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    return model, adapter, final_path


@torch.no_grad()
def rollout_trajectory(model, adapter, dataset, traj_idx, device, store_preds=False):
    """Autoregressive rollout. Returns rel_l2 (T, 4) and optional preds (T, 4, D, H, W)."""
    l2_rel = LpLoss(d=3, p=2, reduction='none', measure=[0.25, 1., 0.5])
    n_steps = dataset.n_timesteps
    n_channels = len(CHANNEL_NAMES)

    y = dataset._state_at(traj_idx, 0).unsqueeze(0).to(device)
    static = dataset._static_at(traj_idx).unsqueeze(0).to(device)

    rel_errors = torch.zeros(n_steps, n_channels, device=device)
    preds = [] if store_preds else None

    for t in range(n_steps):
        action = dataset._action_at(traj_idx, t).unsqueeze(0).to(device)
        _mask_wells(action)

        model_input = adapter.build_model_input(y, action, static)
        y_pred = adapter.forward(model, model_input)

        target = dataset._state_at(traj_idx, t + 1).unsqueeze(0).to(device)
        # (1, 4, D, H, W) -> per-channel relative L2 -> (4,)
        rel_errors[t] = l2_rel(y_pred, target).reshape(n_channels)

        if store_preds:
            preds.append(y_pred.squeeze(0).cpu())
        y = y_pred

    rel_np = rel_errors.cpu().numpy()
    if store_preds:
        return rel_np, torch.stack(preds, dim=0).numpy()
    return rel_np, None


def plot_rel_l2(rel_l2, out_path):
    """rel_l2: (n_seeds, T, 4). Mean line + min/max band per channel."""
    mean = rel_l2.mean(axis=0)
    lo = rel_l2.min(axis=0)
    hi = rel_l2.max(axis=0)
    t = np.arange(rel_l2.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, mean[:, i], color='C0', label='mean')
        ax.fill_between(t, lo[:, i], hi[:, i], color='C0', alpha=0.25, label='min/max')
        ax.set_title(CHANNEL_NAMES[i])
        ax.set_ylabel('Relative L2')
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.set_xlabel('Timestep')
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Autoregressive relative L2 vs timestep', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _extract_slice(volume, slice_fn):
    """volume: (4, D, H, W) -> list of 2D arrays per channel."""
    return [slice_fn(volume[c]) for c in range(volume.shape[0])]


def plot_slice_comparison(gt_traj, pred_traj, slice_key, slice_label, slice_fn, out_path):
    """
    gt_traj / pred_traj: (T, 4, D, H, W) for predicted frames (targets at t+1).
    One figure: for each snapshot time, 4 channels x 3 cols (GT | Pred | |Error|).
    """
    n_times = len(SNAPSHOT_TIMES)
    n_channels = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(
        n_times * n_channels, 3,
        figsize=(10, 2.2 * n_times * n_channels),
        squeeze=False,
    )

    for ti, t in enumerate(SNAPSHOT_TIMES):
        gt_slices = _extract_slice(gt_traj[t], slice_fn)
        pred_slices = _extract_slice(pred_traj[t], slice_fn)

        for c in range(n_channels):
            row = ti * n_channels + c
            truth = gt_slices[c]
            pred = pred_slices[c]
            err = np.abs(pred - truth)

            vmin = min(truth.min(), pred.min())
            vmax = max(truth.max(), pred.max())
            emax = err.max() or 1.0

            ax0, ax1, ax2 = axes[row]
            im0 = ax0.imshow(truth, vmin=vmin, vmax=vmax, aspect='auto')
            ax0.set_ylabel(f't={t}\n{CHANNEL_NAMES[c]}')
            ax0.set_xticks([])
            ax0.set_yticks([])

            ax1.imshow(pred, vmin=vmin, vmax=vmax, aspect='auto')
            ax1.set_xticks([])
            ax1.set_yticks([])

            im2 = ax2.imshow(err, cmap='magma', vmin=0.0, vmax=emax, aspect='auto')
            ax2.set_xticks([])
            ax2.set_yticks([])

            if c == 0:
                ax0.set_title(f'GT (t={t})')
                ax1.set_title(f'Pred (t={t})')
                ax2.set_title(f'|Error| (t={t})')

            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f'Pred vs GT — slice {slice_label}', fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def gather_gt_trajectory(dataset, traj_idx):
    """Ground-truth frames at t+1 for each rollout step: (T, 4, D, H, W)."""
    frames = [
        dataset._state_at(traj_idx, t + 1).numpy()
        for t in range(dataset.n_timesteps)
    ]
    return np.stack(frames, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Autoregressive single-trajectory eval (heterogeneous models)',
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to heterogeneous YAML config',
    )
    parser.add_argument(
        '--traj', type=int, required=True,
        help='Trajectory index in [300, 399]',
    )
    parser.add_argument(
        '--seeds', type=_parse_seeds, default=list(DEFAULT_SEEDS),
        help='Comma-separated seeds (default: 5,42,2026)',
    )
    parser.add_argument(
        '--viz-seed', type=int, default=42,
        help='Seed used for spatial pred/GT panels (default: 42)',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='cuda / cpu (default: cuda if available)',
    )
    parser.add_argument(
        '--output-root', type=str, default='eval_results/single_trajectory',
        help='Root directory for outputs',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 300 <= args.traj <= 399:
        raise ValueError(f'--traj must be in [300, 399], got {args.traj}')
    if args.viz_seed not in args.seeds:
        raise ValueError(f'--viz-seed {args.viz_seed} must be one of --seeds {args.seeds}')

    config_path = _resolve_path(args.config)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if not cfg.get('data', {}).get('heterogeneous', False):
        raise ValueError('This script expects a heterogeneous config (data.heterogeneous: true)')

    model_type = cfg['model']['type']
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    out_dir = _resolve_path(args.output_root) / model_type / str(args.traj)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Config: {config_path}')
    print(f'Model: {model_type} | traj={args.traj} | seeds={args.seeds} | device={device}')
    print(f'Output: {out_dir}')

    k_max = cfg.get('training', {}).get('pushforward_k_max', 10)
    dataset = load_hetero_dataset(cfg['data']['path'], k_max=k_max)
    if args.traj >= dataset.n_trajectories:
        raise IndexError(
            f'traj {args.traj} out of range for dataset with '
            f'{dataset.n_trajectories} trajectories'
        )

    n_seeds = len(args.seeds)
    n_steps = dataset.n_timesteps
    rel_l2 = np.zeros((n_seeds, n_steps, len(CHANNEL_NAMES)), dtype=np.float64)
    viz_preds = None

    for i, seed in enumerate(args.seeds):
        print(f'\n=== Seed {seed} ===')
        model, adapter, ckpt_path = load_model(cfg, seed, device)
        print(f'Loaded EMA weights from {ckpt_path}')
        store = seed == args.viz_seed
        errors, preds = rollout_trajectory(
            model, adapter, dataset, args.traj, device, store_preds=store,
        )
        rel_l2[i] = errors
        if store:
            viz_preds = preds
        mean_err = errors.mean()
        print(f'Mean rel-L2 over trajectory: {mean_err:.6f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    np.savez(
        out_dir / 'metrics.npz',
        rel_l2=rel_l2,
        seeds=np.array(args.seeds, dtype=np.int64),
        traj=np.int64(args.traj),
        channel_names=np.array(CHANNEL_NAMES),
    )
    print(f'Saved {out_dir / "metrics.npz"}')

    plot_rel_l2(rel_l2, out_dir / 'rel_l2_vs_timestep.png')
    print(f'Saved {out_dir / "rel_l2_vs_timestep.png"}')

    gt_traj = gather_gt_trajectory(dataset, args.traj)
    for slice_key, slice_label, slice_fn in SLICE_SPECS:
        out_path = out_dir / f'slice_{slice_key}.png'
        plot_slice_comparison(
            gt_traj, viz_preds, slice_key, slice_label, slice_fn, out_path,
        )
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
