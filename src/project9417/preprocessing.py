from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    categorical_info: dict[str, Any] | None
    label_encoder: LabelEncoder | None
    preprocess_time_sec: float
    numerical_columns: list[str]
    categorical_columns: list[str]


def _build_categorical_info(
    n_num: int,
    categories: list[np.ndarray],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to build xRFM categorical_info.") from exc

    categorical_indices = []
    categorical_vectors = []
    start = n_num
    for cats in categories:
        cat_len = len(cats)
        idxs = torch.arange(start, start + cat_len, dtype=torch.long)
        categorical_indices.append(idxs)
        categorical_vectors.append(torch.eye(cat_len, dtype=torch.float32))
        start += cat_len
    numerical_indices = torch.arange(0, n_num, dtype=torch.long)
    return {
        "numerical_indices": numerical_indices,
        "categorical_indices": categorical_indices,
        "categorical_vectors": categorical_vectors,
    }


def _ensure_2d(y: pd.Series | pd.DataFrame) -> np.ndarray:
    values = y.to_numpy()
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    return values


def _prepare_categorical_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    categorical = frame[columns].copy().astype("object")
    return categorical.where(pd.notna(categorical), np.nan)


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    task_type: str,
    model_family: str,
) -> PreprocessedData:
    start = perf_counter()
    X_train_df = train_df[feature_columns].copy()
    X_val_df = val_df[feature_columns].copy()
    X_test_df = test_df[feature_columns].copy()

    num_cols = [col for col in numeric_columns if col in feature_columns]
    cat_cols = [col for col in categorical_columns if col in feature_columns]

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_num_train = num_imputer.fit_transform(X_train_df[num_cols])
        X_num_val = num_imputer.transform(X_val_df[num_cols])
        X_num_test = num_imputer.transform(X_test_df[num_cols])
        if model_family == "xrfm":
            scaler = StandardScaler()
            X_num_train = scaler.fit_transform(X_num_train)
            X_num_val = scaler.transform(X_num_val)
            X_num_test = scaler.transform(X_num_test)
    else:
        X_num_train = np.empty((len(train_df), 0), dtype=np.float32)
        X_num_val = np.empty((len(val_df), 0), dtype=np.float32)
        X_num_test = np.empty((len(test_df), 0), dtype=np.float32)

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_cat = cat_imputer.fit_transform(_prepare_categorical_frame(X_train_df, cat_cols))
        val_cat = cat_imputer.transform(_prepare_categorical_frame(X_val_df, cat_cols))
        test_cat = cat_imputer.transform(_prepare_categorical_frame(X_test_df, cat_cols))
        X_cat_train = cat_encoder.fit_transform(train_cat)
        X_cat_val = cat_encoder.transform(val_cat)
        X_cat_test = cat_encoder.transform(test_cat)
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    else:
        cat_encoder = None
        X_cat_train = np.empty((len(train_df), 0), dtype=np.float32)
        X_cat_val = np.empty((len(val_df), 0), dtype=np.float32)
        X_cat_test = np.empty((len(test_df), 0), dtype=np.float32)
        cat_feature_names = []

    X_train = np.hstack([X_num_train, X_cat_train]).astype(np.float32)
    X_val = np.hstack([X_num_val, X_cat_val]).astype(np.float32)
    X_test = np.hstack([X_num_test, X_cat_test]).astype(np.float32)
    feature_names = num_cols + cat_feature_names

    y_train_raw = _ensure_2d(train_df[target_column])
    y_val_raw = _ensure_2d(val_df[target_column])
    y_test_raw = _ensure_2d(test_df[target_column])

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(pd.Series(y_train_raw).astype(str))
        y_val = label_encoder.transform(pd.Series(y_val_raw).astype(str))
        y_test = label_encoder.transform(pd.Series(y_test_raw).astype(str))
    else:
        label_encoder = None
        y_train = pd.to_numeric(pd.Series(y_train_raw), errors="coerce").to_numpy(dtype=np.float32)
        y_val = pd.to_numeric(pd.Series(y_val_raw), errors="coerce").to_numpy(dtype=np.float32)
        y_test = pd.to_numeric(pd.Series(y_test_raw), errors="coerce").to_numpy(dtype=np.float32)

    categorical_info = None
    if model_family == "xrfm":
        categories = [] if cat_encoder is None else list(cat_encoder.categories_)
        categorical_info = _build_categorical_info(len(num_cols), categories)

    return PreprocessedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        categorical_info=categorical_info,
        label_encoder=label_encoder,
        preprocess_time_sec=perf_counter() - start,
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )
