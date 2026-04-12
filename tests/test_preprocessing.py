import pandas as pd

from project9417.preprocessing import preprocess_splits


def test_xrfm_preprocessing_builds_categorical_info():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "target": ["yes", "no", "yes", "no"],
        }
    )
    bundle = preprocess_splits(
        train_df=df.iloc[:2].reset_index(drop=True),
        val_df=df.iloc[2:3].reset_index(drop=True),
        test_df=df.iloc[3:].reset_index(drop=True),
        feature_columns=["num", "cat"],
        numeric_columns=["num"],
        categorical_columns=["cat"],
        target_column="target",
        task_type="classification",
        model_family="xrfm",
    )
    assert bundle.categorical_info is not None
    assert len(bundle.feature_names) == 3
