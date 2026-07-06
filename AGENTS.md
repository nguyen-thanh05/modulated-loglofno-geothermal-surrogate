# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research codebase for geothermal surrogate model training. Core training code lives in `training/`, model implementations in `models/`, and reusable objective functions in `losses/`. Experiment YAML files are under `configs/homogeneous_dataset/` and `configs/heterogeneous_dataset/`, with ablation configs in `configs/heterogeneous_dataset/loss_ablation/`. Slurm submission helpers are in `slurm/`, and exploratory analysis notebooks are in `notebooks/`. Keep generated artifacts, checkpoints, logs, and large datasets outside the repository unless explicitly needed.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -e . -r requirements.txt`: install the package and training dependencies.
- `python training/train.py --config configs/homogeneous_dataset/modulated_loglo_homo.yml --seed 42`: run a local training job using one YAML config. Config data paths must point to available `.npy` datasets.
- `python slurm/launch.py --models modulated_loglo --variants homo --seeds '{"modulated_loglo":"42"}' --dry-run`: preview Slurm jobs without submitting.
- `python slurm/launch_loss_ablation.py --dry-run`: preview the loss ablation job chain.
- `python -m compileall training models losses slurm`: syntax-check Python modules.

## Coding Style & Naming Conventions

Use Python 3.9+ and 4-space indentation. Follow the existing style: `snake_case` for functions and variables, `PascalCase` for model classes, and lowercase package names. Prefer explicit config keys over hard-coded constants when behavior varies by experiment. Name configs as `<model>_<variant>.yml`, such as `fno_m8x32x16_h64_hetero.yml`.

## Testing Guidelines

No dedicated automated test suite is currently present. Since this is a research codebase, we will rely on manual testing and verification of results. No need to create test files when adding new functionality.
