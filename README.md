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

## Quick Start

```powershell
python scripts/download_data.py --dataset all
python scripts/prepare_data.py --dataset all
python scripts/run_experiments.py --dataset all --models xrfm xgboost random_forest --device auto --seed 42
python scripts/run_interpretability.py --dataset appendicitis --model xrfm --seed 42
python scripts/run_scaling.py --dataset job_salary --models xrfm xgboost random_forest --device auto --seed 42
python scripts/summarize_results.py
```

If Kaggle API credentials are unavailable, place the Kaggle dataset files into the printed `data/raw/<dataset>/` directory and rerun the relevant command.
