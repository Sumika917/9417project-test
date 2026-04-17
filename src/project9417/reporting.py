from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .paths import FIGURES_DIR, METRIC_RUNS_DIR, METRICS_DIR


MODEL_ORDER = ["random_forest", "xgboost", "xrfm"]
MODEL_PALETTE = {
    "random_forest": "#4C78A8",
    "xgboost": "#F58518",
    "xrfm": "#54A24B",
}


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
            "prediction_backend": record.get("prediction_backend", "native"),
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    plot_specs = [
        ("classification", "accuracy", "Classification (Accuracy, higher is better)"),
        ("regression", "rmse", "Regression (RMSE, lower is better)"),
    ]

    for axis, (task_type, metric_name, title) in zip(axes, plot_specs):
        subset = (
            results_df.loc[results_df["task_type"] == task_type, ["dataset_display_name", "model_name", metric_name]]
            .dropna()
            .copy()
        )
        subset = subset.rename(columns={metric_name: "metric_value"})
        sns.barplot(
            data=subset,
            x="dataset_display_name",
            y="metric_value",
            hue="model_name",
            hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        if task_type == "regression":
            axis.set_yscale("log")
            axis.set_ylabel("RMSE (log scale)")
        else:
            axis.set_ylabel(metric_name.upper())
        axis.tick_params(axis="x", rotation=25)
        for tick in axis.get_xticklabels():
            tick.set_ha("right")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(FIGURES_DIR / "primary_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_timing_metrics(timing_df: pd.DataFrame) -> None:
    timing_specs = [
        ("preprocess_time_sec", "Preprocess Time (sec)"),
        ("fit_time_sec", "Fit Time (sec)"),
        ("predict_time_per_sample_ms", "Inference Time per Sample (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    for axis, (column_name, title) in zip(axes, timing_specs):
        subset = timing_df[["dataset_display_name", "model_name", column_name]].copy()
        sns.barplot(
            data=subset,
            x="dataset_display_name",
            y=column_name,
            hue="model_name",
            hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE,
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel(title.split(" (")[0])
        axis.tick_params(axis="x", rotation=25)
        for tick in axis.get_xticklabels():
            tick.set_ha("right")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.savefig(FIGURES_DIR / "timing_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


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
