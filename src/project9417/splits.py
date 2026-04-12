from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .paths import SPLITS_DIR
from .registry import DatasetSpec
from .utils import read_json, write_json


@dataclass
class DatasetSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    split_path: Path


def get_split_path(dataset_name: str, seed: int) -> Path:
    return SPLITS_DIR / f"{dataset_name}_seed{seed}.json"


def _stratify_values(frame: pd.DataFrame, target_column: str, task_type: str) -> pd.Series | None:
    if task_type != "classification":
        return None
    values = frame[target_column]
    if values.nunique(dropna=False) < 2:
        return None
    return values


def create_or_load_split(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    target_column: str,
    group_column: str | None,
    seed: int = 42,
    force_rebuild: bool = False,
) -> DatasetSplit:
    split_path = get_split_path(spec.name, seed)
    if split_path.exists() and not force_rebuild:
        payload = read_json(split_path)
        return DatasetSplit(
            train_idx=np.array(payload["train_idx"], dtype=int),
            val_idx=np.array(payload["val_idx"], dtype=int),
            test_idx=np.array(payload["test_idx"], dtype=int),
            split_path=split_path,
        )

    indices = np.arange(len(frame))
    train_val_size = 0.85
    val_share_of_train_val = 0.15 / 0.85

    if group_column:
        groups = frame[group_column].to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_rel, test_rel = next(outer.split(indices, groups=groups))
        train_val_idx = indices[train_val_rel]
        test_idx = indices[test_rel]

        inner = GroupShuffleSplit(n_splits=1, test_size=val_share_of_train_val, random_state=seed)
        inner_groups = groups[train_val_idx]
        train_rel, val_rel = next(inner.split(train_val_idx, groups=inner_groups))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    else:
        stratify = _stratify_values(frame, target_column, spec.task_type)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.15,
            random_state=seed,
            stratify=stratify,
        )
        inner_stratify = frame.iloc[train_val_idx][target_column] if stratify is not None else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share_of_train_val,
            random_state=seed,
            stratify=inner_stratify,
        )

    payload = {
        "dataset_name": spec.name,
        "seed": seed,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    write_json(split_path, payload)
    return DatasetSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, split_path=split_path)
