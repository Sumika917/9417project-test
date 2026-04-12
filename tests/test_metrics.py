import numpy as np

from project9417.metrics import classification_metrics, regression_metrics


def test_regression_metrics_keys():
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert set(metrics) == {"rmse", "mae", "r2"}


def test_classification_metrics_binary_auc():
    metrics = classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        ),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
