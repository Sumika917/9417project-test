from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from xrfm import RFM


@dataclass
class XRFMPredictor:
    bandwidth: float = 2.0
    iters: int = 3
    reg: float = 1e-3
    kernel: str = "l2"
    device: str = "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XRFMPredictor":
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        self.model_ = RFM(
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            iters=self.iters,
            device=self.device,
            verbose=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.model_.fit((Xt, yt), (Xt, yt), reg=self.reg, verbose=False)
        return self

    def _mat(self):
        model = self.model_
        return model.sqrtM if model.use_sqrtM else model.M

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            predictions = self.model_.predict(Xt)
        return predictions.detach().cpu().numpy().reshape(-1)

    def gradients(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            grads = self.model_.kernel_obj.get_function_grads(
                x=self.model_.centers,
                z=Xt,
                coefs=self.model_.weights.t(),
                mat=self._mat(),
            )
        return grads.detach().cpu().numpy()[0]


def build_predictor(bandwidth: float, reg: float = 1e-3) -> XRFMPredictor:
    return XRFMPredictor(bandwidth=bandwidth, reg=reg)


def standard_agop(grads: np.ndarray) -> np.ndarray:
    return grads.T @ grads / max(grads.shape[0], 1)


def residual_weighted_agop(
    grads: np.ndarray,
    residuals: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    weights = np.asarray(residuals, dtype=float).reshape(-1) ** 2
    total_weight = weights.sum()
    if total_weight < eps:
        return standard_agop(grads)
    return (grads * weights[:, None]).T @ grads / total_weight


def top_eigvec(matrix: np.ndarray) -> np.ndarray:
    _, eigenvectors = np.linalg.eigh(matrix)
    vector = eigenvectors[:, -1]
    if vector[np.argmax(np.abs(vector))] < 0:
        vector = -vector
    return vector


def split_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    direction: np.ndarray,
    sigma: float,
    reg: float,
) -> dict:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    train_projection = X_train @ direction
    test_projection = X_test @ direction
    threshold = float(np.median(train_projection))

    left_train_mask = train_projection <= threshold
    right_train_mask = ~left_train_mask
    left_test_mask = test_projection <= threshold
    right_test_mask = ~left_test_mask

    predictions = np.empty_like(y_test, dtype=float)
    for train_mask, test_mask in ((left_train_mask, left_test_mask), (right_train_mask, right_test_mask)):
        if train_mask.sum() < 4 or test_mask.sum() == 0:
            if test_mask.sum() > 0:
                fallback = np.mean(y_train[train_mask]) if train_mask.any() else np.mean(y_train)
                predictions[test_mask] = fallback
            continue
        leaf = build_predictor(bandwidth=sigma, reg=reg).fit(X_train[train_mask], y_train[train_mask])
        predictions[test_mask] = leaf.predict(X_test[test_mask])

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return {
        "direction": direction,
        "threshold": threshold,
        "left_size_train": int(left_train_mask.sum()),
        "right_size_train": int(right_train_mask.sum()),
        "test_rmse": rmse,
    }
