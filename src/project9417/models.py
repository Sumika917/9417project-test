from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np

from .metrics import classification_metrics, is_higher_better, regression_metrics
from .preprocessing import PreprocessedData
from .registry import DatasetSpec


@dataclass
class ModelRunResult:
    model_name: str
    estimator: Any
    best_params: dict[str, Any]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    fit_time_sec: float
    predict_time_total_sec: float
    predict_time_per_sample_ms: float
    total_runtime_sec: float
    y_pred_test: np.ndarray
    y_proba_test: np.ndarray | None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _flatten_predictions(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array


def _ensure_proba_shape(proba: np.ndarray | None) -> np.ndarray | None:
    if proba is None:
        return None
    arr = np.asarray(proba)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    return arr


def _evaluate_predictions(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, y_pred.astype(int), y_proba)
    return regression_metrics(y_true.astype(float), y_pred.astype(float))


def _xrfm_param_grid(task_type: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for iters, reg in product([3, 5], [1e-3, 5e-3]):
        configs.append(
            {
                "max_leaf_size": 60_000,
                "split_method": "top_vector_agop_on_subset",
                "rfm_params": {
                    "model": {
                        "kernel": "l2_high_dim",
                        "bandwidth": 10.0,
                        "exponent": 1.0,
                        "diag": False,
                        "bandwidth_mode": "constant",
                    },
                    "fit": {
                        "reg": reg,
                        "iters": iters,
                        "verbose": False,
                        "early_stop_rfm": True,
                        "return_best_params": True,
                    },
                },
                "tuning_metric": "accuracy" if task_type == "classification" else "mse",
            }
        )
    return configs


def _xgboost_param_grid(task_type: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for max_depth, learning_rate in product([4, 6], [0.05, 0.1]):
        configs.append(
            {
                "n_estimators": 300,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": -1,
            }
        )
    return configs


def _rf_param_grid(task_type: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for max_depth, min_samples_leaf in product([None, 20], [1, 4]):
        payload = {
            "n_estimators": 300,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": 42,
            "n_jobs": -1,
        }
        if task_type == "classification":
            payload["class_weight"] = None
        configs.append(payload)
    return configs


def _get_param_grid(model_name: str, task_type: str) -> list[dict[str, Any]]:
    if model_name == "xrfm":
        return _xrfm_param_grid(task_type)
    if model_name == "xgboost":
        return _xgboost_param_grid(task_type)
    if model_name == "random_forest":
        return _rf_param_grid(task_type)
    raise ValueError(f"Unsupported model: {model_name}")


def _build_estimator(
    model_name: str,
    params: dict[str, Any],
    task_type: str,
    bundle: PreprocessedData,
    device: str,
    seed: int,
) -> Any:
    if model_name == "xrfm":
        try:
            from xrfm import xRFM
        except ImportError as exc:
            raise RuntimeError("xrfm is not installed.") from exc
        return xRFM(
            rfm_params=params["rfm_params"],
            max_leaf_size=params["max_leaf_size"],
            split_method=params["split_method"],
            tuning_metric=params["tuning_metric"],
            categorical_info=bundle.categorical_info,
            device=device,
            random_state=seed,
            verbose=False,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed.") from exc
        common = dict(params)
        common["random_state"] = seed
        if task_type == "classification":
            num_class = int(len(np.unique(bundle.y_train)))
            if num_class > 2:
                common["num_class"] = num_class
                common["objective"] = "multi:softprob"
                common["eval_metric"] = "mlogloss"
            else:
                common["objective"] = "binary:logistic"
                common["eval_metric"] = "logloss"
            if device == "cuda":
                common["device"] = "cuda"
            return XGBClassifier(**common)
        common["objective"] = "reg:squarederror"
        common["eval_metric"] = "rmse"
        if device == "cuda":
            common["device"] = "cuda"
        return XGBRegressor(**common)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params = dict(params)
        params["random_state"] = seed
        if task_type == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def _fit_estimator(estimator: Any, model_name: str, bundle: PreprocessedData) -> None:
    if model_name == "xrfm":
        estimator.fit(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)
        return
    estimator.fit(bundle.X_train, bundle.y_train)


def _predict_with_optional_proba(estimator: Any, X: np.ndarray, task_type: str) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = _flatten_predictions(estimator.predict(X))
    y_proba = None
    if task_type == "classification" and hasattr(estimator, "predict_proba"):
        try:
            y_proba = _ensure_proba_shape(np.asarray(estimator.predict_proba(X)))
        except Exception:
            y_proba = None
    return y_pred, y_proba


def _select_better(metric_name: str, current: float, incumbent: float | None) -> bool:
    if incumbent is None:
        return True
    if is_higher_better(metric_name):
        return current > incumbent
    return current < incumbent


def fit_and_select_model(
    model_name: str,
    spec: DatasetSpec,
    bundle: PreprocessedData,
    device: str = "auto",
    seed: int = 42,
) -> ModelRunResult:
    resolved_device = resolve_device(device)
    param_grid = _get_param_grid(model_name, spec.task_type)
    best_candidate: dict[str, Any] | None = None
    best_score: float | None = None
    search_start = perf_counter()

    for params in param_grid:
        estimator = _build_estimator(model_name, params, spec.task_type, bundle, resolved_device, seed)
        fit_start = perf_counter()
        _fit_estimator(estimator, model_name, bundle)
        fit_time = perf_counter() - fit_start
        val_pred, val_proba = _predict_with_optional_proba(estimator, bundle.X_val, spec.task_type)
        val_metrics = _evaluate_predictions(spec.task_type, bundle.y_val, val_pred, val_proba)
        score = val_metrics[spec.primary_metric]
        if _select_better(spec.primary_metric, score, best_score):
            best_score = score
            best_candidate = {
                "estimator": estimator,
                "params": params,
                "validation_metrics": val_metrics,
                "fit_time_sec": fit_time,
            }

    if best_candidate is None:
        raise RuntimeError(f"Model search produced no candidate for {model_name}")

    predict_start = perf_counter()
    y_pred_test, y_proba_test = _predict_with_optional_proba(
        best_candidate["estimator"],
        bundle.X_test,
        spec.task_type,
    )
    predict_time_total_sec = perf_counter() - predict_start
    test_metrics = _evaluate_predictions(spec.task_type, bundle.y_test, y_pred_test, y_proba_test)

    return ModelRunResult(
        model_name=model_name,
        estimator=best_candidate["estimator"],
        best_params=best_candidate["params"],
        validation_metrics=best_candidate["validation_metrics"],
        test_metrics=test_metrics,
        fit_time_sec=float(best_candidate["fit_time_sec"]),
        predict_time_total_sec=float(predict_time_total_sec),
        predict_time_per_sample_ms=float(1000.0 * predict_time_total_sec / max(len(bundle.y_test), 1)),
        total_runtime_sec=float(perf_counter() - search_start),
        y_pred_test=y_pred_test,
        y_proba_test=y_proba_test,
    )
