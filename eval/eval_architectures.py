"""Evaluate configured architectures with full autoregressive rollouts."""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.constants import CHANNEL_NAMES, WELL_COORDS
from training.dataset import ARDataset
from training.model_adapters import create_adapter
from training.model_factory import create_model


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "heterogeneous_dataset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation_results" / "rollouts"
DEFAULT_SEEDS = (5, 42, 2026)
CHANNEL_UNITS_PHYSICAL = ("degC", "degC", "kPa", "kPa")

METRIC_NAMES = (
    "relative_l2",
    "rmse_normalized",
    "rmse_physical",
    "absolute_error_max_normalized",
    "absolute_error_min_normalized",
    "absolute_error_max_physical",
    "absolute_error_min_physical",
)


@dataclass(frozen=True)
class EvaluationSpec:
    config_path: Path
    config: Mapping
    output_name: str
    checkpoint_paths: Tuple[Path, ...]


def _resolve_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _expand_config_paths(raw_paths: Optional[Sequence[str]]) -> List[Path]:
    if raw_paths is None:
        paths = sorted(DEFAULT_CONFIG_DIR.glob("*.yml"))
        if not paths:
            raise FileNotFoundError(
                f"No default YAML configs found in {DEFAULT_CONFIG_DIR}"
            )
        return [path.resolve() for path in paths]

    expanded: List[Path] = []
    for raw_path in raw_paths:
        if glob.has_magic(raw_path):
            pattern = Path(raw_path)
            if not pattern.is_absolute():
                pattern = REPO_ROOT / pattern
            matches = sorted(Path(match).resolve() for match in glob.glob(str(pattern)))
            if not matches:
                raise FileNotFoundError(
                    f"Config pattern did not match any files: {raw_path}"
                )
            expanded.extend(matches)
        else:
            expanded.append(_resolve_path(raw_path))

    unique_paths: List[Path] = []
    seen = set()
    for path in expanded:
        if path in seen:
            raise ValueError(f"Duplicate config path: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        if path.suffix.lower() not in (".yml", ".yaml"):
            raise ValueError(f"Config must be YAML: {path}")
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _read_config(config_path: Path) -> Mapping:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    for key in ("data", "model", "checkpoints"):
        if key not in config or not isinstance(config[key], dict):
            raise KeyError(f"Config {config_path} is missing mapping '{key}'")
    if "path" not in config["data"]:
        raise KeyError(f"Config {config_path} is missing data.path")
    if "type" not in config["model"]:
        raise KeyError(f"Config {config_path} is missing model.type")
    if "final_path" not in config["checkpoints"]:
        raise KeyError(f"Config {config_path} is missing checkpoints.final_path")
    return config


def _inject_seed_into_paths(config: Mapping, seed: int) -> Dict:
    seeded_config = copy.deepcopy(config)
    checkpoints = seeded_config["checkpoints"]
    seed_dir = f"seed{seed}"

    if "running_dir" in checkpoints:
        checkpoints["running_dir"] = os.path.join(
            checkpoints["running_dir"], seed_dir, ""
        )

    final_dir, final_name = os.path.split(checkpoints["final_path"])
    checkpoints["final_path"] = os.path.join(final_dir, seed_dir, final_name)

    if "resume_path" in checkpoints and "running_dir" in checkpoints:
        resume_name = os.path.basename(checkpoints["resume_path"])
        checkpoints["resume_path"] = os.path.join(
            checkpoints["running_dir"], resume_name
        )
    return seeded_config


def _build_specs(
    config_paths: Sequence[Path],
    seeds: Sequence[int],
) -> List[EvaluationSpec]:
    specs: List[EvaluationSpec] = []
    output_names = set()
    missing_checkpoints = []

    for config_path in config_paths:
        config = _read_config(config_path)
        output_name = config_path.stem
        if output_name in output_names:
            raise ValueError(
                f"Config stem '{output_name}' is not unique; output paths would collide"
            )
        output_names.add(output_name)

        checkpoint_paths = []
        for seed in seeds:
            seeded_config = _inject_seed_into_paths(config, seed)
            checkpoint_path = _resolve_path(
                seeded_config["checkpoints"]["final_path"]
            )
            checkpoint_paths.append(checkpoint_path)
            if not checkpoint_path.is_file():
                missing_checkpoints.append(
                    f"  config={config_path}, seed={seed}: {checkpoint_path}"
                )

        specs.append(
            EvaluationSpec(
                config_path=config_path,
                config=config,
                output_name=output_name,
                checkpoint_paths=tuple(checkpoint_paths),
            )
        )

    if missing_checkpoints:
        details = "\n".join(missing_checkpoints)
        raise FileNotFoundError(
            "Missing evaluation checkpoints:\n"
            f"{details}\n"
            "Checkpoint paths are derived from each config's final_path and seed."
        )
    return specs


def _load_array(data_path: Path, filename: str) -> np.ndarray:
    path = data_path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset array: {path}")
    return np.load(path, mmap_mode="r")


def _load_dataset(config: Mapping) -> Tuple[ARDataset, Path]:
    data_config = config["data"]
    data_path = _resolve_path(data_config["path"])
    heterogeneous = bool(data_config.get("heterogeneous", False))
    k_max = config.get("training", {}).get("pushforward_k_max", 1)

    extra_kwargs = {}
    if heterogeneous:
        extra_kwargs = {
            "heterogeneous": True,
            "por_matrix": _load_array(data_path, "all_por_matrix.npy"),
            "por_frac": _load_array(data_path, "all_por_frac.npy"),
            "perm_matrix": _load_array(data_path, "all_perm_matrix.npy"),
            "perm_frac": _load_array(data_path, "all_perm_frac.npy"),
        }

    dataset = ARDataset(
        _load_array(data_path, "all_temp_formation.npy"),
        _load_array(data_path, "all_temp_frac.npy"),
        _load_array(data_path, "all_pres_formation.npy"),
        _load_array(data_path, "all_pres_frac.npy"),
        _load_array(data_path, "all_action.npy"),
        k_max=k_max,
        **extra_kwargs,
    )
    return dataset, data_path


def _validate_dataset(
    dataset: ARDataset,
    start_index: int,
    end_index: int,
    context: str,
) -> None:
    if start_index < 0:
        raise ValueError(f"{context}: start index must be non-negative")
    if end_index < start_index:
        raise ValueError(
            f"{context}: end index {end_index} precedes start index {start_index}"
        )
    if end_index >= dataset.n_trajectories:
        raise IndexError(
            f"{context}: end index {end_index} is outside dataset with "
            f"{dataset.n_trajectories} trajectories"
        )

    state_arrays = {
        "all_temp_formation": dataset.temp_formation,
        "all_temp_frac": dataset.temp_frac,
        "all_pres_formation": dataset.pres_formation,
        "all_pres_frac": dataset.pres_frac,
    }
    expected_spatial_shape = None
    for name, array in state_arrays.items():
        if array.ndim != 5:
            raise ValueError(f"{context}: {name} must have 5 dimensions, got {array.shape}")
        if array.shape[0] <= end_index:
            raise ValueError(
                f"{context}: {name} has only {array.shape[0]} trajectories"
            )
        if array.shape[1] < dataset.n_timesteps + 1:
            raise ValueError(
                f"{context}: {name} needs at least {dataset.n_timesteps + 1} "
                f"states, got {array.shape[1]}"
            )
        if expected_spatial_shape is None:
            expected_spatial_shape = array.shape[2:]
        elif array.shape[2:] != expected_spatial_shape:
            raise ValueError(
                f"{context}: {name} spatial shape {array.shape[2:]} does not "
                f"match {expected_spatial_shape}"
            )

    action = dataset.action
    if action.ndim != 5:
        raise ValueError(
            f"{context}: all_action must have 5 dimensions, got {action.shape}"
        )
    if action.shape[0] <= end_index or action.shape[1] < dataset.n_timesteps:
        raise ValueError(
            f"{context}: all_action shape {action.shape} cannot evaluate "
            f"index {end_index} for {dataset.n_timesteps} timesteps"
        )
    if action.shape[2:] != expected_spatial_shape:
        raise ValueError(
            f"{context}: action spatial shape {action.shape[2:]} does not "
            f"match state shape {expected_spatial_shape}"
        )

    if dataset.heterogeneous:
        static_arrays = {
            "all_por_matrix": dataset.por_matrix,
            "all_por_frac": dataset.por_frac,
            "all_perm_matrix": dataset.perm_matrix,
            "all_perm_frac": dataset.perm_frac,
        }
        expected_static_shape = (dataset.n_trajectories,) + expected_spatial_shape
        for name, array in static_arrays.items():
            if array.shape != expected_static_shape:
                raise ValueError(
                    f"{context}: {name} shape {array.shape} does not match "
                    f"{expected_static_shape}"
                )


def _load_model(
    spec: EvaluationSpec,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    model_type = spec.config["model"]["type"]
    model = create_model(spec.config["model"], model_type)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "ema_model" not in checkpoint:
        raise KeyError(f"Checkpoint has no ema_model weights: {checkpoint_path}")

    ema_state = checkpoint["ema_model"]
    if "_metadata" in ema_state:
        ema_state = ema_state.copy()
        del ema_state["_metadata"]
    model.load_state_dict(ema_state)
    del ema_state
    del checkpoint

    model = model.to(device)
    model.eval()
    adapter = create_adapter(
        model_type,
        heterogeneous=bool(spec.config["data"].get("heterogeneous", False)),
    )
    return model, adapter


def _mask_wells(action: torch.Tensor) -> None:
    """Zero action at well cells for depth slices 0:2."""
    for h_index, w_index in WELL_COORDS:
        action[:, 0:2, h_index, w_index] = 0.0


def _stack_states(
    dataset: ARDataset,
    trajectory_indices: Sequence[int],
    timestep: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [dataset._state_at(index, timestep) for index in trajectory_indices],
        dim=0,
    ).to(device)


def _stack_actions(
    dataset: ARDataset,
    trajectory_indices: Sequence[int],
    timestep: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [dataset._action_at(index, timestep) for index in trajectory_indices],
        dim=0,
    ).to(device)


def _stack_static(
    dataset: ARDataset,
    trajectory_indices: Sequence[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not dataset.heterogeneous:
        return None
    return torch.stack(
        [dataset._static_at(index) for index in trajectory_indices],
        dim=0,
    ).to(device)


def _physical_scales(
    dataset: ARDataset,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            dataset._temp_range,
            dataset._temp_range,
            dataset._pres_range,
            dataset._pres_range,
        ],
        dtype=torch.float32,
        device=device,
    )


def _compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    physical_scales: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    difference = prediction - target
    flattened_difference = difference.flatten(start_dim=2)
    flattened_target = target.flatten(start_dim=2)

    relative_l2 = torch.linalg.vector_norm(
        flattened_difference, ord=2, dim=2
    ) / torch.linalg.vector_norm(flattened_target, ord=2, dim=2)

    squared_error = difference.square()
    rmse_normalized = torch.sqrt(squared_error.mean(dim=(2, 3, 4)))
    absolute_error = difference.abs()
    absolute_error_max_normalized = absolute_error.amax(dim=(2, 3, 4))
    absolute_error_min_normalized = absolute_error.amin(dim=(2, 3, 4))

    return {
        "relative_l2": relative_l2,
        "rmse_normalized": rmse_normalized,
        "rmse_physical": rmse_normalized * physical_scales,
        "absolute_error_max_normalized": absolute_error_max_normalized,
        "absolute_error_min_normalized": absolute_error_min_normalized,
        "absolute_error_max_physical": (
            absolute_error_max_normalized * physical_scales
        ),
        "absolute_error_min_physical": (
            absolute_error_min_normalized * physical_scales
        ),
    }


def _raise_for_nonfinite(
    tensor: torch.Tensor,
    metric_name: str,
    seed: int,
    trajectory_indices: Sequence[int],
    timestep: int,
) -> None:
    bad_indices = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)
    if bad_indices.numel() == 0:
        return

    batch_index, channel_index = bad_indices[0].tolist()
    raise FloatingPointError(
        f"Non-finite {metric_name}: seed={seed}, "
        f"trajectory={trajectory_indices[batch_index]}, timestep={timestep}, "
        f"channel={CHANNEL_NAMES[channel_index]}"
    )


@torch.inference_mode()
def _evaluate_checkpoint(
    model: torch.nn.Module,
    adapter: object,
    dataset: ARDataset,
    trajectory_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, np.ndarray]:
    n_trajectories = len(trajectory_indices)
    n_timesteps = dataset.n_timesteps
    n_channels = len(CHANNEL_NAMES)
    metrics = {
        name: np.empty(
            (n_trajectories, n_timesteps, n_channels),
            dtype=np.float32,
        )
        for name in METRIC_NAMES
    }
    physical_scales = _physical_scales(dataset, device)

    n_batches = (n_trajectories + batch_size - 1) // batch_size
    progress_interval = max(1, n_batches // 10)
    for batch_number, batch_start in enumerate(
        range(0, n_trajectories, batch_size),
        start=1,
    ):
        batch_stop = min(batch_start + batch_size, n_trajectories)
        batch_indices = trajectory_indices[batch_start:batch_stop]
        state = _stack_states(dataset, batch_indices, 0, device)
        static = _stack_static(dataset, batch_indices, device)

        for timestep in range(n_timesteps):
            action = _stack_actions(dataset, batch_indices, timestep, device)
            _mask_wells(action)
            model_input = adapter.build_model_input(state, action, static)
            prediction = adapter.forward(model, model_input)
            target = _stack_states(
                dataset,
                batch_indices,
                timestep + 1,
                device,
            )

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Model output shape {tuple(prediction.shape)} does not "
                    f"match target shape {tuple(target.shape)} at seed={seed}, "
                    f"trajectory={batch_indices[0]}, timestep={timestep}"
                )

            step_metrics = _compute_metrics(
                prediction,
                target,
                physical_scales,
            )
            for name, values in step_metrics.items():
                _raise_for_nonfinite(
                    values,
                    name,
                    seed,
                    batch_indices,
                    timestep,
                )
                metrics[name][batch_start:batch_stop, timestep] = (
                    values.cpu().numpy()
                )
            state = prediction

        if batch_number % progress_interval == 0 or batch_number == n_batches:
            print(
                f"    completed {batch_stop}/{n_trajectories} trajectories",
                flush=True,
            )

    return metrics


def _allocate_config_metrics(
    n_seeds: int,
    n_trajectories: int,
    n_timesteps: int,
) -> Dict[str, np.ndarray]:
    return {
        name: np.empty(
            (n_seeds, n_trajectories, n_timesteps, len(CHANNEL_NAMES)),
            dtype=np.float32,
        )
        for name in METRIC_NAMES
    }


def _save_metrics(
    output_path: Path,
    metrics: Mapping[str, np.ndarray],
    spec: EvaluationSpec,
    seeds: Sequence[int],
    trajectory_indices: Sequence[int],
    data_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.npz")

    payload = dict(metrics)
    payload.update(
        {
            "seeds": np.asarray(seeds, dtype=np.int64),
            "trajectory_indices": np.asarray(
                trajectory_indices,
                dtype=np.int64,
            ),
            "rollout_timesteps": np.arange(
                metrics["relative_l2"].shape[2],
                dtype=np.int64,
            ),
            "target_state_indices": np.arange(
                1,
                metrics["relative_l2"].shape[2] + 1,
                dtype=np.int64,
            ),
            "channel_names": np.asarray(CHANNEL_NAMES),
            "channel_units_physical": np.asarray(CHANNEL_UNITS_PHYSICAL),
            "config_path": np.asarray(str(spec.config_path)),
            "data_path": np.asarray(str(data_path)),
            "checkpoint_paths": np.asarray(
                [str(path) for path in spec.checkpoint_paths]
            ),
            "model_type": np.asarray(spec.config["model"]["type"]),
            "heterogeneous": np.asarray(
                bool(spec.config["data"].get("heterogeneous", False))
            ),
            "relative_l2_definition": np.asarray(
                "||prediction-target||_2 / ||target||_2 over spatial axes"
            ),
            "rmse_normalized_definition": np.asarray(
                "sqrt(mean((prediction-target)^2)) over spatial axes"
            ),
            "physical_error_units": np.asarray(
                "Temperature errors are degC; pressure errors are kPa"
            ),
        }
    )

    np.savez_compressed(temporary_path, **payload)
    os.replace(temporary_path, output_path)


def _release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _resolve_device(raw_device: Optional[str]) -> torch.device:
    device = torch.device(
        raw_device
        if raw_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
    return device


def _validate_cli_values(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.start_index < 0:
        raise ValueError(
            f"--start-index must be non-negative, got {args.start_index}"
        )
    if args.end_index < args.start_index:
        raise ValueError(
            f"--end-index {args.end_index} must be >= "
            f"--start-index {args.start_index}"
        )
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"--seeds contains duplicates: {args.seeds}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full autoregressive rollouts for one or more model configs. "
            "By default, all top-level heterogeneous composite-loss configs are "
            "evaluated."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "YAML config paths or glob patterns. Default: every top-level *.yml "
            "in configs/heterogeneous_dataset (excludes mse_only and ablations)."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Checkpoint seeds (default: 5 42 2026).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=300,
        help="First trajectory index, inclusive (default: 300).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=399,
        help="Last trajectory index, inclusive (default: 399).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of trajectories rolled out together (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device such as cuda, cuda:1, or cpu (default: auto).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _validate_cli_values(args)

    config_paths = _expand_config_paths(args.configs)
    specs = _build_specs(config_paths, args.seeds)
    device = _resolve_device(args.device)
    output_root = _resolve_path(args.output_root)
    trajectory_indices = list(range(args.start_index, args.end_index + 1))

    print(f"Device: {device}")
    print(
        f"Trajectories: {args.start_index}-{args.end_index} "
        f"({len(trajectory_indices)} total)"
    )
    print(f"Seeds: {args.seeds}")
    print(f"Configs: {len(specs)}")
    print(f"Output root: {output_root}")

    for config_number, spec in enumerate(specs, start=1):
        print(
            f"\n[{config_number}/{len(specs)}] {spec.output_name} "
            f"({spec.config['model']['type']})"
        )
        dataset, data_path = _load_dataset(spec.config)
        _validate_dataset(
            dataset,
            args.start_index,
            args.end_index,
            str(spec.config_path),
        )
        config_metrics = _allocate_config_metrics(
            len(args.seeds),
            len(trajectory_indices),
            dataset.n_timesteps,
        )

        for seed_index, (seed, checkpoint_path) in enumerate(
            zip(args.seeds, spec.checkpoint_paths)
        ):
            print(
                f"  Seed {seed}: loading EMA weights from {checkpoint_path}",
                flush=True,
            )
            model = None
            adapter = None
            try:
                model, adapter = _load_model(spec, checkpoint_path, device)
                seed_metrics = _evaluate_checkpoint(
                    model,
                    adapter,
                    dataset,
                    trajectory_indices,
                    args.batch_size,
                    device,
                    seed,
                )
                for name in METRIC_NAMES:
                    config_metrics[name][seed_index] = seed_metrics[name]
                del seed_metrics
            finally:
                if model is not None:
                    del model
                if adapter is not None:
                    del adapter
                _release_device_memory(device)

        output_path = output_root / spec.output_name / "metrics.npz"
        _save_metrics(
            output_path,
            config_metrics,
            spec,
            args.seeds,
            trajectory_indices,
            data_path,
        )
        print(f"  Saved {output_path}")

        del config_metrics
        del dataset
        _release_device_memory(device)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
