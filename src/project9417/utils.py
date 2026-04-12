from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_serializable(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def resolve_column_name(columns: list[str], preferred: str, aliases: tuple[str, ...]) -> str:
    normalized_to_original = {normalize_name(col): col for col in columns}
    for candidate in (preferred, *aliases):
        exact = [col for col in columns if col == candidate]
        if exact:
            return exact[0]
        normalized = normalize_name(candidate)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]
    raise KeyError(f"Unable to resolve column {preferred!r} from available columns: {columns}")


def as_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
