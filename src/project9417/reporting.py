from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .paths import FIGURES_DIR, METRIC_RUNS_DIR, METRICS_DIR


def _load_run_records() -> list[dict]:
    records = []
    for path in sorted(METRIC_RUNS_DIR.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _flatten_metrics(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    timing_rows = []
    for record in records:
        base = {
            "dataset_name": record["dataset_name"],
            "dataset_display_name": record["dataset_display_name"],
            "task_type": record["task_type"],
            "model_name": record["model_name"],
            "train_size": record["train_size"],
            "val_size": record["val_size"],
            "test_size": record["test_size"],
        }
        result_rows.append({**base, **record["metrics"], **{f"val_{k}": v for k, v in record["validation_metrics"].items()}})
        timing_rows.append({**base, **record["timings"]})
    return pd.DataFrame(result_rows), pd.DataFrame(timing_rows)


def _write_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [col for col in ["rmse", "mae", "r2", "accuracy", "roc_auc", "macro_f1"] if col in results_df.columns]
    leaderboard = (
        results_df.melt(
            id_vars=["dataset_display_name", "model_name"],
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )
        .pivot_table(
            index="dataset_display_name",
            columns=["model_name", "metric"],
            values="value",
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    leaderboard.to_csv(METRICS_DIR / "leaderboard.csv")
    return leaderboard


def _plot_primary_metrics(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["primary_metric_name"] = plot_df["task_type"].map({"regression": "rmse", "classification": "accuracy"})
    plot_df["primary_metric_value"] = plot_df.apply(lambda row: row[row["primary_metric_name"]], axis=1)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="dataset_display_name", y="primary_metric_value", hue="model_name")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "primary_metric_comparison.png", dpi=200)
    plt.close()


def _plot_timing_metrics(timing_df: pd.DataFrame) -> None:
    long_df = timing_df.melt(
        id_vars=["dataset_display_name", "model_name"],
        value_vars=["preprocess_time_sec", "fit_time_sec", "predict_time_total_sec"],
        var_name="timing_type",
        value_name="seconds",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_df, x="dataset_display_name", y="seconds", hue="model_name")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "timing_comparison.png", dpi=200)
    plt.close()


def summarize_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _load_run_records()
    if not records:
        raise FileNotFoundError(f"No run records found in {METRIC_RUNS_DIR}")

    results_df, timing_df = _flatten_metrics(records)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_DIR / "per_dataset_results.csv", index=False)
    timing_df.to_csv(METRICS_DIR / "timing_summary.csv", index=False)
    _write_leaderboard(results_df)
    _plot_primary_metrics(results_df)
    _plot_timing_metrics(timing_df)
    return results_df, timing_df
