"""Plot architecture-level rollout errors from saved evaluation metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation_results" / "rollouts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation_results" / "plots"
AUTO_LOG_DYNAMIC_RANGE = 1_000.0


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    axis_label: str
    uses_physical_units: bool


@dataclass(frozen=True)
class ArchitectureResult:
    name: str
    label: str
    source_path: Path
    seeds: np.ndarray
    trajectory_indices: np.ndarray
    rollout_timesteps: np.ndarray
    channel_names: np.ndarray
    channel_units_physical: np.ndarray
    metrics: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RunExclusion:
    architecture_name: str
    seed: int
    trajectory: int


METRIC_SPECS = (
    MetricSpec(
        key="relative_l2",
        title="Autoregressive relative L2 error",
        axis_label="Relative L2 error",
        uses_physical_units=False,
    ),
    MetricSpec(
        key="rmse_physical",
        title="Autoregressive physical RMSE",
        axis_label="RMSE",
        uses_physical_units=True,
    ),
    MetricSpec(
        key="absolute_error_max_physical",
        title="Autoregressive maximum physical absolute error",
        axis_label="Maximum absolute error",
        uses_physical_units=True,
    ),
)

ARCHITECTURE_LABELS = {
    "fno_m4x16x8_h128_hetero": "FNO (m4x16x8, h128)",
    "fno_m8x32x16_h64_hetero": "FNO (m8x32x16, h64)",
    "modulated_loglo_hetero": "Modulated LOGLO-FNO",
    "ufno_hetero": "U-FNO",
    "unet_hetero": "U-Net",
    "uno_hetero": "UNO",
    "vanilla_loglo_hetero": "Vanilla LOGLO-FNO",
}

ARCHITECTURE_ORDER = (
    "modulated_loglo_hetero",
    "vanilla_loglo_hetero",
    "fno_m4x16x8_h128_hetero",
    "fno_m8x32x16_h64_hetero",
    "ufno_hetero",
    "unet_hetero",
    "uno_hetero",
)


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _parse_run_exclusion(raw_value: str) -> RunExclusion:
    parts = raw_value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Run exclusions must use ARCHITECTURE:SEED:TRAJECTORY"
        )
    architecture_name, raw_seed, raw_trajectory = parts
    if not architecture_name:
        raise argparse.ArgumentTypeError("Run exclusion architecture is empty")
    try:
        seed = int(raw_seed)
        trajectory = int(raw_trajectory)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Run exclusion seed and trajectory must be integers"
        ) from error
    return RunExclusion(architecture_name, seed, trajectory)


def _architecture_sort_key(path: Path) -> Tuple[int, str]:
    try:
        index = ARCHITECTURE_ORDER.index(path.parent.name)
    except ValueError:
        index = len(ARCHITECTURE_ORDER)
    return index, path.parent.name


def _discover_metric_paths(results_root: Path) -> List[Path]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    metric_paths = sorted(
        results_root.glob("*/metrics.npz"),
        key=_architecture_sort_key,
    )
    if not metric_paths:
        raise FileNotFoundError(
            f"No architecture metrics found under {results_root}; expected "
            "<architecture>/metrics.npz"
        )
    return metric_paths


def _load_result(path: Path) -> ArchitectureResult:
    metric_names = tuple(spec.key for spec in METRIC_SPECS)
    metadata_names = (
        "seeds",
        "trajectory_indices",
        "rollout_timesteps",
        "channel_names",
        "channel_units_physical",
    )
    required_names = set(metric_names + metadata_names)

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required_names.difference(archive.files))
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")

        seeds = np.asarray(archive["seeds"]).copy()
        trajectory_indices = np.asarray(archive["trajectory_indices"]).copy()
        rollout_timesteps = np.asarray(archive["rollout_timesteps"]).copy()
        channel_names = np.asarray(archive["channel_names"]).astype(str)
        channel_units = np.asarray(archive["channel_units_physical"]).astype(str)
        metrics: Dict[str, np.ndarray] = {
            name: np.asarray(archive[name]).copy() for name in metric_names
        }

    expected_shape = (
        seeds.size,
        trajectory_indices.size,
        rollout_timesteps.size,
        channel_names.size,
    )
    if channel_units.shape != channel_names.shape:
        raise ValueError(
            f"{path}: channel unit shape {channel_units.shape} does not match "
            f"channel name shape {channel_names.shape}"
        )

    for name, values in metrics.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{path}: {name} shape {values.shape} does not match metadata "
                f"shape {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{path}: {name} contains NaN or Inf")
        if np.any(values < 0.0):
            raise ValueError(f"{path}: {name} contains negative errors")

    architecture_name = path.parent.name
    return ArchitectureResult(
        name=architecture_name,
        label=ARCHITECTURE_LABELS.get(
            architecture_name,
            architecture_name.replace("_", " "),
        ),
        source_path=path,
        seeds=seeds,
        trajectory_indices=trajectory_indices,
        rollout_timesteps=rollout_timesteps,
        channel_names=channel_names,
        channel_units_physical=channel_units,
        metrics=metrics,
    )


def _validate_compatible(results: Sequence[ArchitectureResult]) -> None:
    reference = results[0]
    metadata = (
        "seeds",
        "trajectory_indices",
        "rollout_timesteps",
        "channel_names",
        "channel_units_physical",
    )
    labels = set()

    for result in results:
        if result.label in labels:
            raise ValueError(f"Duplicate architecture label: {result.label}")
        labels.add(result.label)

        for field_name in metadata:
            reference_values = getattr(reference, field_name)
            result_values = getattr(result, field_name)
            if not np.array_equal(reference_values, result_values):
                raise ValueError(
                    f"{result.source_path}: {field_name} does not match "
                    f"{reference.source_path}"
                )


def _build_inclusion_masks(
    results: Sequence[ArchitectureResult],
    exclusions: Sequence[RunExclusion],
) -> Dict[str, np.ndarray]:
    results_by_name = {result.name: result for result in results}
    masks = {
        result.name: np.ones(
            (result.seeds.size, result.trajectory_indices.size),
            dtype=bool,
        )
        for result in results
    }

    for exclusion in exclusions:
        if exclusion.architecture_name not in results_by_name:
            available = ", ".join(sorted(results_by_name))
            raise ValueError(
                f"Unknown excluded architecture '{exclusion.architecture_name}'. "
                f"Available architectures: {available}"
            )

        result = results_by_name[exclusion.architecture_name]
        seed_matches = np.flatnonzero(result.seeds == exclusion.seed)
        trajectory_matches = np.flatnonzero(
            result.trajectory_indices == exclusion.trajectory
        )
        if seed_matches.size != 1:
            raise ValueError(
                f"{result.name}: excluded seed {exclusion.seed} was not found"
            )
        if trajectory_matches.size != 1:
            raise ValueError(
                f"{result.name}: excluded trajectory {exclusion.trajectory} "
                "was not found"
            )

        seed_index = int(seed_matches[0])
        trajectory_index = int(trajectory_matches[0])
        if not masks[result.name][seed_index, trajectory_index]:
            raise ValueError(
                f"Duplicate run exclusion: {result.name}:{exclusion.seed}:"
                f"{exclusion.trajectory}"
            )
        masks[result.name][seed_index, trajectory_index] = False

    for result in results:
        retained_per_seed = masks[result.name].sum(axis=1)
        if np.any(retained_per_seed == 0):
            bad_seed_index = int(np.flatnonzero(retained_per_seed == 0)[0])
            raise ValueError(
                f"{result.name}: exclusions remove every trajectory for seed "
                f"{result.seeds[bad_seed_index]}"
            )
    return masks


def _aggregate_seed_curves(
    values: np.ndarray,
    inclusion_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and seed range after averaging trajectories per seed."""
    if inclusion_mask.shape != values.shape[:2]:
        raise ValueError(
            f"Inclusion mask shape {inclusion_mask.shape} does not match "
            f"metric sample shape {values.shape[:2]}"
        )
    per_seed = np.stack(
        [
            values[seed_index, inclusion_mask[seed_index]].mean(
                axis=0,
                dtype=np.float64,
            )
            for seed_index in range(values.shape[0])
        ],
        axis=0,
    )
    return (
        per_seed.mean(axis=0),
        per_seed.min(axis=0),
        per_seed.max(axis=0),
    )


def _positive_range(curves: Sequence[np.ndarray]) -> Tuple[float, float]:
    combined = np.concatenate([np.ravel(curve) for curve in curves])
    positive = combined[combined > 0.0]
    if positive.size == 0:
        return 0.0, 0.0
    return float(positive.min()), float(positive.max())


def _select_axis_scale(
    requested_scale: str,
    curves: Sequence[np.ndarray],
) -> Tuple[str, float, float]:
    positive_min, positive_max = _positive_range(curves)
    if requested_scale != "auto":
        return requested_scale, positive_min, positive_max
    if (
        positive_min > 0.0
        and positive_max / positive_min > AUTO_LOG_DYNAMIC_RANGE
    ):
        return "log", positive_min, positive_max
    return "linear", positive_min, positive_max


def _print_global_extremum(
    results: Sequence[ArchitectureResult],
    metric_name: str,
    inclusion_masks: Mapping[str, np.ndarray],
) -> None:
    maximum = -np.inf
    maximum_result: Optional[ArchitectureResult] = None
    maximum_index: Optional[Tuple[int, ...]] = None

    for result in results:
        values = result.metrics[metric_name]
        eligible = inclusion_masks[result.name][:, :, None, None]
        included_values = np.where(eligible, values, -np.inf)
        flat_index = int(np.argmax(included_values))
        index = np.unravel_index(flat_index, values.shape)
        value = float(values[index])
        if value > maximum:
            maximum = value
            maximum_result = result
            maximum_index = index

    if maximum_result is None or maximum_index is None:
        raise RuntimeError(f"Could not locate maximum for {metric_name}")

    seed_index, trajectory_index, timestep_index, channel_index = maximum_index
    print(
        f"  Global maximum: {maximum:.6e} | "
        f"architecture={maximum_result.label} | "
        f"seed={maximum_result.seeds[seed_index]} | "
        f"trajectory={maximum_result.trajectory_indices[trajectory_index]} | "
        f"timestep={maximum_result.rollout_timesteps[timestep_index]} | "
        f"channel={maximum_result.channel_names[channel_index]}"
    )


def _plot_metric(
    results: Sequence[ArchitectureResult],
    metric_spec: MetricSpec,
    output_dir: Path,
    requested_scale: str,
    inclusion_masks: Mapping[str, np.ndarray],
    exclusion_caption: str,
) -> Path:
    reference = results[0]
    aggregated = {
        result.name: _aggregate_seed_curves(
            result.metrics[metric_spec.key],
            inclusion_masks[result.name],
        )
        for result in results
    }
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=False,
    )
    legend_handles = []
    legend_labels = []

    print(f"\n{metric_spec.key}")
    _print_global_extremum(results, metric_spec.key, inclusion_masks)

    for channel_index, axis in enumerate(axes.ravel()):
        scale_curves = []
        for result in results:
            mean, lower, upper = aggregated[result.name]
            scale_curves.extend(
                (
                    mean[:, channel_index],
                    lower[:, channel_index],
                    upper[:, channel_index],
                )
            )
        axis_scale, positive_min, positive_max = _select_axis_scale(
            requested_scale,
            scale_curves,
        )

        for architecture_index, result in enumerate(results):
            mean, lower, upper = aggregated[result.name]
            color = colors(architecture_index % colors.N)
            highlighted = result.name == "modulated_loglo_hetero"
            (line,) = axis.plot(
                reference.rollout_timesteps,
                mean[:, channel_index],
                color=color,
                linewidth=2.8 if highlighted else 1.7,
                label=result.label,
                zorder=4 if highlighted else 2,
            )
            axis.fill_between(
                reference.rollout_timesteps,
                lower[:, channel_index],
                upper[:, channel_index],
                color=color,
                alpha=0.20 if highlighted else 0.12,
                linewidth=0.0,
                zorder=3 if highlighted else 1,
            )
            if channel_index == 0:
                legend_handles.append(line)
                legend_labels.append(result.label)

        channel_name = reference.channel_names[channel_index]
        axis.set_title(channel_name)
        if metric_spec.uses_physical_units:
            unit = reference.channel_units_physical[channel_index]
            axis.set_ylabel(f"{metric_spec.axis_label} ({unit})")
        else:
            axis.set_ylabel(metric_spec.axis_label)
        axis.set_xlabel("Rollout timestep")
        axis.set_yscale(axis_scale)
        axis.grid(True, which="both", alpha=0.25)
        print(
            f"  {channel_name}: yscale={axis_scale}, "
            f"positive plotted range=[{positive_min:.6e}, {positive_max:.6e}]"
        )

    source_root = reference.source_path.parent.parent
    try:
        source_label = source_root.relative_to(REPO_ROOT)
    except ValueError:
        source_label = source_root

    exclusion_note = (
        f" Excluded: {exclusion_caption}." if exclusion_caption else ""
    )
    fig.suptitle(metric_spec.title, fontsize=15, y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.text(
        0.5,
        0.015,
        (
            "Lines: per-seed trajectory mean, then seed mean. Shading: seed "
            "min/max. Auto log threshold: positive plotted range > 10^3."
            f"{exclusion_note} "
            f"Source: {source_label}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))

    output_path = output_dir / f"{metric_spec.key}_vs_timestep.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare architecture rollout errors over time from metrics.npz "
            "files produced by eval_architectures.py."
        )
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=(
            "Directory containing <architecture>/metrics.npz files "
            f"(default: {DEFAULT_RESULTS_ROOT})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for comparison plots (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--yscale",
        choices=("auto", "linear", "log"),
        default="auto",
        help=(
            "Y-axis scale. Auto uses log when positive plotted values span "
            "more than three orders of magnitude (default: auto)."
        ),
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=[],
        type=_parse_run_exclusion,
        metavar="ARCHITECTURE:SEED:TRAJECTORY",
        help=(
            "Exclude one architecture checkpoint/trajectory rollout from all "
            "aggregates. May be repeated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results_root = _resolve_path(args.results_root)
    output_dir = _resolve_path(args.output_dir)

    metric_paths = _discover_metric_paths(results_root)
    results = [_load_result(path) for path in metric_paths]
    _validate_compatible(results)
    inclusion_masks = _build_inclusion_masks(results, args.exclude_run)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = results[0]
    results_by_name = {result.name: result for result in results}
    exclusion_caption = ", ".join(
        (
            f"{results_by_name[exclusion.architecture_name].label} "
            f"seed {exclusion.seed}, trajectory {exclusion.trajectory}"
        )
        for exclusion in args.exclude_run
    )
    print(f"Results root: {results_root}")
    print(f"Architectures: {len(results)}")
    print(
        f"Evaluation ensemble: {reference.seeds.size} seeds x "
        f"{reference.trajectory_indices.size} trajectories"
    )
    print(f"Rollout timesteps: {reference.rollout_timesteps.size}")
    print(f"Y-axis mode: {args.yscale}")
    if exclusion_caption:
        print(f"Excluded runs: {exclusion_caption}")

    output_paths = [
        _plot_metric(
            results,
            spec,
            output_dir,
            args.yscale,
            inclusion_masks,
            exclusion_caption,
        )
        for spec in METRIC_SPECS
    ]
    print(f"\nGenerated {len(output_paths)} plots in {output_dir}")


if __name__ == "__main__":
    main()
