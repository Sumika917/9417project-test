# COMP9417 Project Codebase

This repository contains the code implementation for the COMP9417 group project:

- Models: `xRFM`, `XGBoost`, `Random Forest`
- Tasks: 5 tabular datasets covering regression and classification
- Outputs: experiment metrics, timings, interpretability assets, and scaling analysis artifacts

## Project Layout

- `src/project9417/`: core package
- `scripts/`: CLI entrypoints
- `data/raw/`: downloaded or manually placed source data
- `data/processed/`: processed tables and metadata
- `data/splits/`: persisted train/validation/test splits
- `data/artifacts/`: metrics, predictions, configs, figures, and interpretability outputs

## Environment Setup

Recommended on Windows PowerShell: use the virtual environment's Python executable directly instead of relying on activation.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The project installs `openpyxl` for the Excel-based `appendicitis` dataset and `cupy-cuda12x` so `XGBoost` can prefer GPU-backed prediction.
On Windows, the project also creates a local CUDA shim under the workspace and redirects CuPy cache/temp files into project-local directories so GPU prediction does not depend on restricted system temp locations.

Optional: activate the environment first.

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation works, you can use `python` and `pytest` directly afterwards. If you do not want to activate the environment, keep using the explicit `.\.venv\Scripts\python.exe` commands below.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If the virtual environment is already activated, the equivalent command is:

```powershell
pytest
```

## Quick Start

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py --dataset all
.\.venv\Scripts\python.exe scripts\prepare_data.py --dataset all
.\.venv\Scripts\python.exe scripts\run_experiments.py --dataset all --models xrfm xgboost random_forest --device auto --seed 42
.\.venv\Scripts\python.exe scripts\run_interpretability.py --dataset appendicitis --model xrfm --seed 42
.\.venv\Scripts\python.exe scripts\run_scaling.py --dataset job_salary --models xrfm xgboost random_forest --device auto --seed 42
.\.venv\Scripts\python.exe scripts\summarize_results.py
```

If Kaggle API credentials are unavailable, place the Kaggle dataset files into the printed `data/raw/<dataset>/` directory and rerun the relevant command.
If you need to refresh a dataset that already exists locally, add `--force-download` to the download command.
`XGBoost` now prefers GPU prediction when CUDA and CuPy are available; on Windows the runtime is configured automatically to use project-local cache/temp directories and a PyTorch-backed CUDA DLL shim. If GPU array conversion still fails, the pipeline automatically falls back to the CPU-input path and records the backend in the run artifacts.

## Common Targeted Commands

Run a single dataset through the whole data-prep pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py --dataset appendicitis
.\.venv\Scripts\python.exe scripts\prepare_data.py --dataset appendicitis
```

Run one dataset with all three models:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --dataset appendicitis --models xrfm xgboost random_forest --device auto --seed 42
```

Run one model across all datasets:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --dataset all --models xgboost --device auto --seed 42
```

Run one model on one dataset only:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --dataset iris --models xgboost --device auto --seed 42
.\.venv\Scripts\python.exe scripts\run_experiments.py --dataset student_exam --models xrfm --device auto --seed 42
```

Refresh only the fixed auxiliary experiments:

```powershell
.\.venv\Scripts\python.exe scripts\run_interpretability.py --dataset appendicitis --model xrfm --device auto --seed 42
.\.venv\Scripts\python.exe scripts\run_scaling.py --dataset job_salary --models xrfm xgboost random_forest --device auto --seed 42
```

Rebuild only the summary tables and figures from existing run records:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_results.py
```

If you need to regenerate the persistent split files, add `--force-rebuild-splits` to `run_experiments.py`.

Run the standalone bonus experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_bonus_residual_agop.py
```

The bonus script reuses the project's prepared `parkinsons` dataset and split logic, and fetches the extra `energy_efficiency` dataset through `ucimlrepo` at runtime.

## Key Outputs

Main experiment outputs:

- `data/artifacts/metrics/leaderboard.csv`
- `data/artifacts/metrics/per_dataset_results.csv`
- `data/artifacts/metrics/timing_summary.csv`
- `data/artifacts/metrics/runs/*.json`
- `data/artifacts/predictions/<dataset>/<model>.csv`
- `data/artifacts/best_configs/<dataset>_<model>.json`

Interpretability outputs:

- `data/artifacts/interpretability/appendicitis_rankings.csv`
- `data/artifacts/interpretability/appendicitis_rank_correlations.csv`
- `data/artifacts/interpretability/appendicitis_leaf_importances.csv`
- `data/artifacts/interpretability/appendicitis_interpretability_summary.json`
- `data/artifacts/interpretability/appendicitis_top_features.png`
- `data/artifacts/interpretability/appendicitis_agop_heatmap.png`

Scaling outputs:

- `data/artifacts/metrics/scaling_results.csv`
- `data/artifacts/figures/job_salary_scaling_metric.png`
- `data/artifacts/figures/job_salary_scaling_fit_time.png`

Summary figures:

- `data/artifacts/figures/primary_metric_comparison.png`
- `data/artifacts/figures/timing_comparison.png`

Bonus outputs:

- `data/artifacts/bonus/bonus_summary.csv`
- `data/artifacts/bonus/bonus_results.json`
- `data/artifacts/bonus/parkinsons_agop_diagonals.png`
- `data/artifacts/bonus/energy_efficiency_agop_diagonals.png`

## Bonus Experiment

The repository includes a standalone bonus experiment for residual-weighted AGOP as an extension of the xRFM split direction.
It compares standard AGOP with a residual-weighted variant that emphasizes samples with larger squared residuals when building the split criterion.

Run it with:

```powershell
.\.venv\Scripts\python.exe scripts\run_bonus_residual_agop.py
```

The script writes its tables and figures into `data/artifacts/bonus/`.
It reuses the existing `parkinsons` dataset pipeline from the project and downloads the additional `energy_efficiency` UCI dataset on demand via `ucimlrepo`.
