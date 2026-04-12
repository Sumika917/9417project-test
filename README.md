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
