from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .experiment import prepare_experiment
from .models import fit_and_select_model
from .paths import FIGURES_DIR, METRICS_DIR
from .preprocessing import preprocess_splits
from .registry import DATASET_REGISTRY, DEFAULT_MODELS


MODEL_ORDER = ["random_forest", "xgboost", "xrfm"]
MODEL_PALETTE = {
    "random_forest": "#4C78A8",
    "xgboost": "#F58518",
    "xrfm": "#54A24B",
}


@dataclass
class ScalingResult:
    dataframe: pd.DataFrame
    metric_plot_path: str
    time_plot_path: str


def _load_existing_scaling_results(csv_path: str | pd.io.common.FilePath | None) -> pd.DataFrame:
    if csv_path is None:
        return pd.DataFrame()
    path = pd.io.common.stringify_path(csv_path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _write_scaling_plots(output: pd.DataFrame, dataset_name: str, primary_metric: str, display_name: str) -> tuple[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metric_plot_path = FIGURES_DIR / f"{dataset_name}_scaling_metric.png"
    time_plot_path = FIGURES_DIR / f"{dataset_name}_scaling_fit_time.png"
    output = output.sort_values(["model_name", "sample_size"]).reset_index(drop=True)

    fig, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=output,
        x="sample_size",
        y="primary_metric",
        hue="model_name",
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        marker="o",
        ax=axis,
    )
    axis.legend(title="Model")
    axis.set_xscale("log")
    axis.set_xlabel("Training sample size")
    axis.set_ylabel(primary_metric.upper())
    axis.set_title(f"{display_name}: Test {primary_metric.upper()} vs training size")
    axis.grid(True, which="both", alpha=0.25)
    metric_offsets = {
        "random_forest": (8, 12),
        "xgboost": (8, 0),
        "xrfm": (8, -12),
    }
    for row in output.sort_values("sample_size").groupby("model_name", sort=False).tail(1).itertuples():
        dx, dy = metric_offsets.get(row.model_name, (8, 0))
        axis.annotate(
            f"{row.model_name}: {row.primary_metric:.0f}",
            (row.sample_size, row.primary_metric),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(metric_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=output,
        x="sample_size",
        y="fit_time_sec",
        hue="model_name",
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        marker="o",
        ax=axis,
    )
    axis.legend(title="Model")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Training sample size")
    axis.set_ylabel("Fit time (sec)")
    axis.set_title(f"{display_name}: Fit Time vs training size")
    axis.grid(True, which="both", alpha=0.25)
    time_offsets = {
        "random_forest": (8, 12),
        "xgboost": (8, 4),
        "xrfm": (8, -8),
    }
    for row in output.sort_values("sample_size").groupby("model_name", sort=False).tail(1).itertuples():
        dx, dy = time_offsets.get(row.model_name, (8, 0))
        axis.annotate(
            f"{row.model_name}: {row.fit_time_sec:.1f}s",
            (row.sample_size, row.fit_time_sec),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(time_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(metric_plot_path), str(time_plot_path)


def run_scaling_analysis(
    dataset_name: str = "job_salary",
    model_names: list[str] | None = None,
    seed: int = 42,
    device: str = "auto",
    sample_sizes: list[int] | None = None,
) -> ScalingResult:
    if model_names is None:
        model_names = list(DEFAULT_MODELS)
    if sample_sizes is None:
        sample_sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000]

    spec = DATASET_REGISTRY[dataset_name]
    experiment = prepare_experiment(dataset_name, seed=seed, force_rebuild_splits=False)
    full_train = experiment.train_df
    val_df = experiment.val_df
    test_df = experiment.test_df
    filtered_sizes = list(dict.fromkeys([size for size in sample_sizes if size < len(full_train)] + [len(full_train)]))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "scaling_results.csv"
    existing = _load_existing_scaling_results(csv_path)
    if not existing.empty:
        existing = existing[existing["dataset_name"] == dataset_name].copy()
    completed = {
        (str(row.dataset_name), int(row.sample_size), str(row.model_name))
        for row in existing.itertuples()
    }
    records: list[dict[str, float | int | str]] = existing.to_dict(orient="records")
    for sample_size in filtered_sizes:
        sampled_train = full_train.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)
        for model_name in model_names:
            key = (dataset_name, int(sample_size), model_name)
            if key in completed:
                continue
            bundle = preprocess_splits(
                train_df=sampled_train,
                val_df=val_df,
                test_df=test_df,
                feature_columns=experiment.prepared_dataset.feature_columns,
                numeric_columns=experiment.prepared_dataset.numeric_columns,
                categorical_columns=experiment.prepared_dataset.categorical_columns,
                target_column=experiment.prepared_dataset.target_column,
                task_type=spec.task_type,
                model_family=model_name,
            )
            result = fit_and_select_model(model_name, spec, bundle, device=device, seed=seed)
            record = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "sample_size": sample_size,
                "primary_metric": result.test_metrics[spec.primary_metric],
                "fit_time_sec": result.fit_time_sec,
                "preprocess_time_sec": bundle.preprocess_time_sec,
                "total_runtime_sec": bundle.preprocess_time_sec + result.total_runtime_sec,
            }
            records.append(record)
            completed.add(key)
            pd.DataFrame(records).sort_values(["sample_size", "model_name"]).to_csv(csv_path, index=False)

    output = pd.DataFrame(records).sort_values(["sample_size", "model_name"]).reset_index(drop=True)
    metric_plot_path, time_plot_path = _write_scaling_plots(output, dataset_name, spec.primary_metric, spec.display_name)

    return ScalingResult(output, metric_plot_path, time_plot_path)
