from __future__ import annotations

import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs
from .registry import DATASET_REGISTRY, DatasetSpec
from .utils import as_serializable, normalize_name, read_json, read_table, resolve_column_name, write_json

SUPPORTED_TABLE_SUFFIXES = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".data"}


@dataclass
class PreparedDataset:
    spec: DatasetSpec
    dataframe: pd.DataFrame
    target_column: str
    group_column: str | None
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    processed_path: Path
    metadata_path: Path


def get_raw_dataset_dir(dataset_name: str) -> Path:
    return RAW_DATA_DIR / dataset_name


def get_processed_dataset_paths(dataset_name: str) -> tuple[Path, Path]:
    base_dir = PROCESSED_DATA_DIR / dataset_name
    return base_dir / "data.csv", base_dir / "metadata.json"


def has_downloaded_raw_data(spec: DatasetSpec) -> bool:
    dataset_dir = get_raw_dataset_dir(spec.name)
    if not dataset_dir.exists():
        return False
    if spec.source_type == "uci":
        if (dataset_dir / "uci_snapshot.csv").exists():
            return True
    return any(path.is_file() and path.suffix.lower() in SUPPORTED_TABLE_SUFFIXES for path in dataset_dir.rglob("*"))


def _ensure_kaggle_download(spec: DatasetSpec) -> None:
    dataset_dir = get_raw_dataset_dir(spec.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        spec.source_id,
        "-p",
        str(dataset_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install it or place the raw files manually into "
            f"{dataset_dir}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Ensure credentials are configured or place the raw files manually into "
            f"{dataset_dir}. stderr={exc.stderr.strip()}"
        ) from exc


def _download_uci_dataset(spec: DatasetSpec) -> pd.DataFrame:
    raw_dir = get_raw_dataset_dir(spec.name)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        fetch_ucirepo = None

    if fetch_ucirepo is not None:
        try:
            dataset = fetch_ucirepo(id=int(spec.source_id))
            features = dataset.data.features.copy()
            targets = dataset.data.targets.copy()
            if not isinstance(targets, pd.DataFrame):
                targets = pd.DataFrame(targets)
            frame = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
            frame.to_csv(raw_dir / "uci_snapshot.csv", index=False)
            write_json(
                raw_dir / "uci_metadata.json",
                {
                    "metadata": as_serializable(dict(dataset.metadata)),
                    "variables": as_serializable(dataset.variables.to_dict()),
                    "source_url": spec.source_url,
                    "source_download_url": spec.source_download_url,
                    "download_method": "ucimlrepo",
                },
            )
            return frame
        except Exception as exc:
            if not spec.source_download_url:
                raise RuntimeError(
                    f"Failed to download UCI dataset {spec.name} via ucimlrepo and no fallback URL is configured."
                ) from exc

    if not spec.source_download_url:
        raise RuntimeError(f"No direct download URL configured for UCI dataset {spec.name}.")

    parsed = urllib.parse.urlparse(spec.source_download_url)
    filename = Path(parsed.path).name or f"{spec.name}_download"
    download_path = raw_dir / filename
    urllib.request.urlretrieve(spec.source_download_url, download_path)

    if download_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(download_path) as archive:
            archive.extractall(raw_dir)
    elif download_path.suffix.lower() not in SUPPORTED_TABLE_SUFFIXES:
        raise RuntimeError(
            f"Unsupported fallback file type for UCI dataset {spec.name}: {download_path.suffix.lower()}"
        )

    table_path = _find_candidate_raw_table(raw_dir)
    frame = _read_uci_fallback_table(spec, table_path)
    write_json(
        raw_dir / "uci_metadata.json",
        {
            "source_url": spec.source_url,
            "source_download_url": spec.source_download_url,
            "download_method": "direct",
            "table_path": str(table_path),
        },
    )
    return frame


def _read_uci_fallback_table(spec: DatasetSpec, table_path: Path) -> pd.DataFrame:
    if table_path.suffix.lower() != ".data":
        return read_table(table_path)
    if spec.name == "iris":
        return pd.read_csv(
            table_path,
            header=None,
            names=[
                "sepal length",
                "sepal width",
                "petal length",
                "petal width",
                "class",
            ],
        )
    return pd.read_csv(table_path)


def download_dataset(
    dataset_name: str,
    allow_manual_fallback: bool = True,
    force_download: bool = False,
) -> Path:
    ensure_project_dirs()
    spec = DATASET_REGISTRY[dataset_name]
    dataset_dir = get_raw_dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not force_download and has_downloaded_raw_data(spec):
        return dataset_dir
    if spec.source_type == "uci":
        _download_uci_dataset(spec)
        return dataset_dir

    try:
        _ensure_kaggle_download(spec)
    except RuntimeError:
        if not allow_manual_fallback:
            raise
        manual_hint = dataset_dir / "README_MANUAL_PLACEMENT.txt"
        manual_hint.write_text(
            "Automatic Kaggle download was unavailable.\n"
            f"Source URL: {spec.source_url}\n"
            f"Place the raw dataset files for {spec.source_id} into this directory and rerun the command.\n",
            encoding="utf-8",
        )
    return dataset_dir


def _find_candidate_raw_table(dataset_dir: Path) -> Path:
    candidates = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_TABLE_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(f"No supported table files found in {dataset_dir}")
    return sorted(candidates, key=lambda path: (path.suffix.lower() != ".csv", len(path.parts), path.name))[0]


def load_raw_dataset(spec: DatasetSpec) -> pd.DataFrame:
    dataset_dir = get_raw_dataset_dir(spec.name)
    if spec.source_type == "uci":
        csv_path = dataset_dir / "uci_snapshot.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        if has_downloaded_raw_data(spec):
            return _read_uci_fallback_table(spec, _find_candidate_raw_table(dataset_dir))
        return _download_uci_dataset(spec)

    table_path = _find_candidate_raw_table(dataset_dir)
    return read_table(table_path)


def _drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    removable = [col for col in frame.columns if frame[col].isna().all()]
    if removable:
        frame = frame.drop(columns=removable)
    return frame


def _coerce_object_categories(frame: pd.DataFrame, target_column: str, group_column: str | None) -> pd.DataFrame:
    protected = {target_column}
    if group_column:
        protected.add(group_column)
    for column in frame.columns:
        if column in protected:
            continue
        if frame[column].dtype == "object":
            sample = frame[column].dropna().astype(str)
            if not sample.empty and sample.nunique() / max(len(sample), 1) < 0.5:
                frame[column] = frame[column].astype("category")
    return frame


def prepare_dataset(dataset_name: str, force: bool = False) -> PreparedDataset:
    spec = DATASET_REGISTRY[dataset_name]
    processed_path, metadata_path = get_processed_dataset_paths(dataset_name)
    if processed_path.exists() and metadata_path.exists() and not force:
        frame = pd.read_csv(processed_path)
        metadata = read_json(metadata_path)
        return PreparedDataset(
            spec=spec,
            dataframe=frame,
            target_column=metadata["target_column"],
            group_column=metadata.get("group_column"),
            feature_columns=metadata["feature_columns"],
            numeric_columns=metadata["numeric_columns"],
            categorical_columns=metadata["categorical_columns"],
            processed_path=processed_path,
            metadata_path=metadata_path,
        )

    raw_frame = load_raw_dataset(spec).copy()
    raw_frame = _drop_empty_columns(raw_frame)
    raw_frame.columns = [str(col).strip() for col in raw_frame.columns]

    target_column = resolve_column_name(raw_frame.columns.tolist(), spec.target_column, spec.target_aliases)
    group_column = None
    if spec.group_column:
        group_column = resolve_column_name(raw_frame.columns.tolist(), spec.group_column, spec.group_aliases or (spec.group_column,))

    raw_frame = raw_frame.dropna(subset=[target_column]).reset_index(drop=True)

    drop_columns: list[str] = []
    for candidate in (*spec.drop_columns, *spec.drop_aliases):
        normalized_candidate = normalize_name(candidate)
        for column in raw_frame.columns:
            if normalize_name(column) == normalized_candidate and column != target_column:
                drop_columns.append(column)
    if drop_columns:
        raw_frame = raw_frame.drop(columns=sorted(set(drop_columns)))

    for candidate in spec.preferred_id_columns:
        for column in list(raw_frame.columns):
            if normalize_name(column) == normalize_name(candidate) and column not in {target_column, group_column}:
                raw_frame = raw_frame.drop(columns=[column])

    raw_frame = _coerce_object_categories(raw_frame, target_column, group_column)

    feature_columns = [col for col in raw_frame.columns if col not in {target_column, group_column}]
    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(raw_frame[col]) and not pd.api.types.is_bool_dtype(raw_frame[col])
    ]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(processed_path, index=False)
    metadata = {
        "dataset_name": spec.name,
        "display_name": spec.display_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "source_id": spec.source_id,
        "source_url": spec.source_url,
        "target_column": target_column,
        "group_column": group_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "n_rows": int(len(raw_frame)),
        "n_features": int(len(feature_columns)),
    }
    write_json(metadata_path, metadata)
    return PreparedDataset(
        spec=spec,
        dataframe=raw_frame,
        target_column=target_column,
        group_column=group_column,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        processed_path=processed_path,
        metadata_path=metadata_path,
    )


def prepare_all_datasets(force: bool = False) -> dict[str, PreparedDataset]:
    return {name: prepare_dataset(name, force=force) for name in DATASET_REGISTRY}


def reset_dataset_artifacts(dataset_name: str) -> None:
    processed_path, metadata_path = get_processed_dataset_paths(dataset_name)
    for path in [processed_path.parent, get_raw_dataset_dir(dataset_name)]:
        if path.exists():
            shutil.rmtree(path)
