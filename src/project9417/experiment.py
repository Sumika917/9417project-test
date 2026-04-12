from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .datasets import PreparedDataset, prepare_dataset
from .models import ModelRunResult, fit_and_select_model
from .paths import BEST_CONFIGS_DIR, METRIC_RUNS_DIR, PREDICTIONS_DIR, ensure_project_dirs
from .preprocessing import PreprocessedData, preprocess_splits
from .registry import DEFAULT_MODELS
from .splits import DatasetSplit, create_or_load_split
from .utils import as_serializable, write_json


@dataclass
class PreparedExperiment:
    prepared_dataset: PreparedDataset
    split: DatasetSplit
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def prepare_experiment(dataset_name: str, seed: int = 42, force_rebuild_splits: bool = False) -> PreparedExperiment:
    prepared = prepare_dataset(dataset_name)
    split = create_or_load_split(
        frame=prepared.dataframe,
        spec=prepared.spec,
        target_column=prepared.target_column,
        group_column=prepared.group_column,
        seed=seed,
        force_rebuild=force_rebuild_splits,
    )
    df = prepared.dataframe
    return PreparedExperiment(
        prepared_dataset=prepared,
        split=split,
        train_df=df.iloc[split.train_idx].reset_index(drop=True),
        val_df=df.iloc[split.val_idx].reset_index(drop=True),
        test_df=df.iloc[split.test_idx].reset_index(drop=True),
    )


def preprocess_for_model(experiment: PreparedExperiment, model_name: str) -> PreprocessedData:
    prepared = experiment.prepared_dataset
    return preprocess_splits(
        train_df=experiment.train_df,
        val_df=experiment.val_df,
        test_df=experiment.test_df,
        feature_columns=prepared.feature_columns,
        numeric_columns=prepared.numeric_columns,
        categorical_columns=prepared.categorical_columns,
        target_column=prepared.target_column,
        task_type=prepared.spec.task_type,
        model_family=model_name,
    )


def _prediction_frame(
    dataset_name: str,
    model_result: ModelRunResult,
    bundle: PreprocessedData,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset_name": dataset_name,
            "row_id": range(len(bundle.y_test)),
            "y_true": bundle.y_test,
            "y_pred": model_result.y_pred_test,
        }
    )
    if model_result.y_proba_test is not None:
        for idx in range(model_result.y_proba_test.shape[1]):
            frame[f"proba_{idx}"] = model_result.y_proba_test[:, idx]
    return frame


def persist_model_artifacts(
    experiment: PreparedExperiment,
    bundle: PreprocessedData,
    model_result: ModelRunResult,
    seed: int,
) -> dict[str, Any]:
    ensure_project_dirs()
    dataset_name = experiment.prepared_dataset.spec.name
    model_name = model_result.model_name

    prediction_dir = PREDICTIONS_DIR / dataset_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{model_name}.csv"
    _prediction_frame(dataset_name, model_result, bundle).to_csv(prediction_path, index=False)

    config_path = BEST_CONFIGS_DIR / f"{dataset_name}_{model_name}.json"
    write_json(config_path, as_serializable(model_result.best_params))

    record = {
        "dataset_name": dataset_name,
        "dataset_display_name": experiment.prepared_dataset.spec.display_name,
        "task_type": experiment.prepared_dataset.spec.task_type,
        "model_name": model_name,
        "seed": seed,
        "feature_count": len(bundle.feature_names),
        "train_size": len(bundle.y_train),
        "val_size": len(bundle.y_val),
        "test_size": len(bundle.y_test),
        "metrics": as_serializable(model_result.test_metrics),
        "validation_metrics": as_serializable(model_result.validation_metrics),
        "timings": {
            "preprocess_time_sec": bundle.preprocess_time_sec,
            "fit_time_sec": model_result.fit_time_sec,
            "predict_time_total_sec": model_result.predict_time_total_sec,
            "predict_time_per_sample_ms": model_result.predict_time_per_sample_ms,
            "total_runtime_sec": bundle.preprocess_time_sec + model_result.total_runtime_sec,
        },
        "best_config_path": str(config_path),
        "prediction_path": str(prediction_path),
        "best_params": as_serializable(model_result.best_params),
        "feature_names": bundle.feature_names,
    }
    run_record_path = METRIC_RUNS_DIR / f"{dataset_name}__{model_name}.json"
    write_json(run_record_path, record)
    return record


def run_experiments(
    dataset_names: list[str],
    model_names: list[str] | None = None,
    seed: int = 42,
    device: str = "auto",
    force_rebuild_splits: bool = False,
) -> list[dict[str, Any]]:
    if model_names is None:
        model_names = list(DEFAULT_MODELS)

    records: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        experiment = prepare_experiment(dataset_name, seed=seed, force_rebuild_splits=force_rebuild_splits)
        for model_name in model_names:
            bundle = preprocess_for_model(experiment, model_name)
            model_result = fit_and_select_model(
                model_name=model_name,
                spec=experiment.prepared_dataset.spec,
                bundle=bundle,
                device=device,
                seed=seed,
            )
            records.append(persist_model_artifacts(experiment, bundle, model_result, seed=seed))
    return records
