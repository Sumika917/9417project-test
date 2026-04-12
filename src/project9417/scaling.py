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


@dataclass
class ScalingResult:
    dataframe: pd.DataFrame
    metric_plot_path: str
    time_plot_path: str


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

    records: list[dict[str, float | int | str]] = []
    for sample_size in filtered_sizes:
        sampled_train = full_train.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)
        for model_name in model_names:
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
            records.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "sample_size": sample_size,
                    "primary_metric": result.test_metrics[spec.primary_metric],
                    "fit_time_sec": result.fit_time_sec,
                    "preprocess_time_sec": bundle.preprocess_time_sec,
                    "total_runtime_sec": bundle.preprocess_time_sec + result.total_runtime_sec,
                }
            )

    output = pd.DataFrame(records)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "scaling_results.csv"
    output.to_csv(csv_path, index=False)

    metric_plot_path = FIGURES_DIR / f"{dataset_name}_scaling_metric.png"
    time_plot_path = FIGURES_DIR / f"{dataset_name}_scaling_fit_time.png"

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=output, x="sample_size", y="primary_metric", hue="model_name", marker="o")
    plt.xscale("log")
    plt.ylabel(spec.primary_metric)
    plt.title(f"{spec.display_name}: Test {spec.primary_metric} vs n")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=output, x="sample_size", y="fit_time_sec", hue="model_name", marker="o")
    plt.xscale("log")
    plt.ylabel("fit_time_sec")
    plt.title(f"{spec.display_name}: Fit time vs n")
    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=200)
    plt.close()

    return ScalingResult(output, str(metric_plot_path), str(time_plot_path))
