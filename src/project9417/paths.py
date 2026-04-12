from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METRIC_RUNS_DIR = METRICS_DIR / "runs"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
INTERPRETABILITY_DIR = ARTIFACTS_DIR / "interpretability"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
BEST_CONFIGS_DIR = ARTIFACTS_DIR / "best_configs"


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        ARTIFACTS_DIR,
        METRICS_DIR,
        METRIC_RUNS_DIR,
        FIGURES_DIR,
        INTERPRETABILITY_DIR,
        PREDICTIONS_DIR,
        BEST_CONFIGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
