import pandas as pd

from project9417.registry import DATASET_REGISTRY
from project9417 import splits as split_module
from project9417.splits import create_or_load_split


def test_group_split_has_no_subject_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(split_module, "SPLITS_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "feature": range(20),
            "subject#": [f"s{i // 2}" for i in range(20)],
            "total_UPDRS": range(20),
        }
    )
    split = create_or_load_split(
        frame=df,
        spec=DATASET_REGISTRY["parkinsons"],
        target_column="total_UPDRS",
        group_column="subject#",
        seed=7,
        force_rebuild=True,
    )
    train_subjects = set(df.iloc[split.train_idx]["subject#"])
    val_subjects = set(df.iloc[split.val_idx]["subject#"])
    test_subjects = set(df.iloc[split.test_idx]["subject#"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
