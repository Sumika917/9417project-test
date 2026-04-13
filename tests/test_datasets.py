from pathlib import Path

from project9417.datasets import _read_uci_fallback_table, has_downloaded_raw_data
from project9417.registry import DATASET_REGISTRY


def test_has_downloaded_raw_data_for_uci(tmp_path, monkeypatch):
    monkeypatch.setattr("project9417.datasets.RAW_DATA_DIR", tmp_path)
    dataset_dir = tmp_path / "iris"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "uci_snapshot.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    assert has_downloaded_raw_data(DATASET_REGISTRY["iris"])


def test_has_downloaded_raw_data_for_uci_direct_table(tmp_path, monkeypatch):
    monkeypatch.setattr("project9417.datasets.RAW_DATA_DIR", tmp_path)
    dataset_dir = tmp_path / "appendicitis"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "app_data.xlsx").write_text("placeholder", encoding="utf-8")

    assert has_downloaded_raw_data(DATASET_REGISTRY["appendicitis"])


def test_has_downloaded_raw_data_for_kaggle(tmp_path, monkeypatch):
    monkeypatch.setattr("project9417.datasets.RAW_DATA_DIR", tmp_path)
    dataset_dir = tmp_path / "job_salary"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text("salary\n1\n", encoding="utf-8")

    assert has_downloaded_raw_data(DATASET_REGISTRY["job_salary"])


def test_has_downloaded_raw_data_ignores_non_table_files(tmp_path, monkeypatch):
    monkeypatch.setattr("project9417.datasets.RAW_DATA_DIR", tmp_path)
    dataset_dir = tmp_path / "student_exam"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "README_MANUAL_PLACEMENT.txt").write_text("manual", encoding="utf-8")

    assert not has_downloaded_raw_data(DATASET_REGISTRY["student_exam"])


def test_read_uci_fallback_table_for_iris_data(tmp_path):
    data_path = tmp_path / "iris.data"
    data_path.write_text("5.1,3.5,1.4,0.2,Iris-setosa\n", encoding="utf-8")

    frame = _read_uci_fallback_table(DATASET_REGISTRY["iris"], data_path)

    assert list(frame.columns) == ["sepal length", "sepal width", "petal length", "petal width", "class"]
    assert frame.iloc[0]["class"] == "Iris-setosa"
