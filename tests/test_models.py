import sys

import numpy as np

from project9417.models import _get_param_grid, _predict_with_optional_proba


class _FakeGPUArray:
    def __init__(self, value):
        self.value = value


class _FakeCuPyModule:
    @staticmethod
    def asarray(x):
        return _FakeGPUArray(x)

    @staticmethod
    def asnumpy(x):
        return x


class _FakeXGBClassifier:
    def __init__(self, fail_on_gpu_predict: bool = False):
        self.fail_on_gpu_predict = fail_on_gpu_predict

    def predict(self, X):
        if self.fail_on_gpu_predict and isinstance(X, _FakeGPUArray):
            raise RuntimeError("gpu predict failed")
        return np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_xrfm_param_grid_enables_best_agop_collection():
    grid = _get_param_grid("xrfm", "classification", collect_leaf_agops=True)

    assert grid[0]["rfm_params"]["fit"]["get_agop_best_model"] is True


def test_xgboost_prediction_prefers_gpu_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPyModule())

    y_pred, y_proba, backend = _predict_with_optional_proba(
        _FakeXGBClassifier(),
        np.array([[1.0], [2.0]], dtype=np.float32),
        "classification",
        "xgboost",
        "cuda",
    )

    assert backend == "gpu_inplace"
    assert y_pred.tolist() == [0, 1]
    assert y_proba.shape == (2, 2)


def test_xgboost_prediction_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPyModule())

    y_pred, y_proba, backend = _predict_with_optional_proba(
        _FakeXGBClassifier(fail_on_gpu_predict=True),
        np.array([[1.0], [2.0]], dtype=np.float32),
        "classification",
        "xgboost",
        "cuda",
    )

    assert backend == "cpu_fallback"
    assert y_pred.tolist() == [0, 1]
    assert y_proba.shape == (2, 2)
