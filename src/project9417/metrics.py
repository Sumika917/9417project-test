from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float("nan"),
    }
    if y_proba is None:
        return metrics

    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            positive_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def is_higher_better(metric_name: str) -> bool:
    return metric_name in {"accuracy", "roc_auc", "r2", "macro_f1"}


def metric_sort_key(metric_name: str, value: float) -> float:
    if np.isnan(value):
        return -np.inf if is_higher_better(metric_name) else np.inf
    return value if is_higher_better(metric_name) else -value


def to_native_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}
