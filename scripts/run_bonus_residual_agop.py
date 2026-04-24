from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from project9417.datasets import prepare_dataset
from project9417.paths import ARTIFACTS_DIR
from project9417.residual_agop import (
    build_predictor,
    residual_weighted_agop,
    split_and_score,
    standard_agop,
    top_eigvec,
)
from project9417.splits import create_or_load_split


OUT_DIR = ARTIFACTS_DIR / "bonus"
MAX_TRAIN_ROWS = 2000
SEED = 42
MAIN_DATASETS = [
    ("parkinsons", 2.0, 1e-3),
]
BONUS_UCI_DATASETS = [
    (242, "energy_efficiency", 0, 1.5, 1e-3),
]


def load_main_dataset(
    name: str,
    max_train_rows: int = MAX_TRAIN_ROWS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    prepared = prepare_dataset(name)
    frame = prepared.dataframe
    feature_columns = [col for col in prepared.numeric_columns if col in prepared.feature_columns]
    target_column = prepared.target_column

    split = create_or_load_split(
        frame=frame,
        spec=prepared.spec,
        target_column=target_column,
        group_column=prepared.group_column,
        seed=seed,
    )
    train_idx = np.concatenate([split.train_idx, split.val_idx])
    test_idx = split.test_idx

    selected_columns = feature_columns + [target_column]
    train_frame = frame.iloc[train_idx][selected_columns].dropna().reset_index(drop=True)
    test_frame = frame.iloc[test_idx][selected_columns].dropna().reset_index(drop=True)

    if len(train_frame) > max_train_rows:
        train_frame = train_frame.sample(max_train_rows, random_state=seed).reset_index(drop=True)

    return (
        train_frame[feature_columns].to_numpy(dtype=float),
        train_frame[target_column].to_numpy(dtype=float),
        test_frame[feature_columns].to_numpy(dtype=float),
        test_frame[target_column].to_numpy(dtype=float),
        feature_columns,
    )


def load_uci_bonus_dataset(
    uci_id: int,
    target_col_idx: int = 0,
    max_train_rows: int = MAX_TRAIN_ROWS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=uci_id)
    feature_frame = dataset.data.features.select_dtypes(include="number").copy()
    target_series = dataset.data.targets.iloc[:, target_col_idx].copy()

    valid_mask = feature_frame.notna().all(axis=1) & target_series.notna()
    feature_frame = feature_frame[valid_mask].reset_index(drop=True)
    target_series = target_series[valid_mask].reset_index(drop=True)

    feature_columns = list(feature_frame.columns)
    X = feature_frame.to_numpy(dtype=float)
    y = target_series.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)

    if len(X_train) > max_train_rows:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X_train), size=max_train_rows, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    return X_train, y_train, X_test, y_test, feature_columns


def run_split_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    tag: str,
    sigma: float,
    reg: float = 1e-3,
    feature_names: list[str] | None = None,
) -> dict:
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictor = build_predictor(bandwidth=sigma, reg=reg).fit(X_train_scaled, y_train)
    gradients = predictor.gradients(X_train_scaled)
    residuals = y_train - predictor.predict(X_train_scaled)

    agop = standard_agop(gradients)
    weighted_agop = residual_weighted_agop(gradients, residuals)

    standard_vector = top_eigvec(agop)
    weighted_vector = top_eigvec(weighted_agop)
    cosine_abs = float(abs(np.dot(standard_vector, weighted_vector)))

    root_rmse = float(np.sqrt(np.mean((y_test - predictor.predict(X_test_scaled)) ** 2)))
    standard_split = split_and_score(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        standard_vector,
        sigma=sigma,
        reg=reg,
    )
    weighted_split = split_and_score(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        weighted_vector,
        sigma=sigma,
        reg=reg,
    )

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    figure, axis = plt.subplots(figsize=(max(8, len(feature_names) * 0.55), 4))
    positions = np.arange(len(feature_names))
    width = 0.38
    axis.bar(positions - width / 2, np.diag(agop), width, label="standard AGOP")
    axis.bar(positions + width / 2, np.diag(weighted_agop), width, label="residual-weighted AGOP")
    axis.set_xticks(positions)
    axis.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("diagonal value")
    axis.set_title("AGOP diagonals - " + tag)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{tag}_agop_diagonals.png", dpi=180)
    plt.close(figure)

    return {
        "dataset": tag,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "sigma": sigma,
        "reg": reg,
        "root_predictor_test_rmse": root_rmse,
        "direction_cosine_abs": cosine_abs,
        "top_std_feature": feature_names[int(np.argmax(np.abs(standard_vector)))],
        "top_res_feature": feature_names[int(np.argmax(np.abs(weighted_vector)))],
        "split_std_test_rmse": standard_split["test_rmse"],
        "split_res_test_rmse": weighted_split["test_rmse"],
        "delta_rmse": standard_split["test_rmse"] - weighted_split["test_rmse"],
        "v_std": standard_vector.tolist(),
        "v_res": weighted_vector.tolist(),
        "feature_names": feature_names,
        "agop_diag": np.diag(agop).tolist(),
        "agop_res_diag": np.diag(weighted_agop).tolist(),
    }


def _print_result(result: dict) -> None:
    print(
        "  std top feature : "
        + result["top_std_feature"]
        + "\n"
        + "  res top feature : "
        + result["top_res_feature"]
        + "\n"
        + "  |cos|           : "
        + f"{result['direction_cosine_abs']:.4f}"
        + "\n"
        + "  RMSE std / res  : "
        + f"{result['split_std_test_rmse']:.4f}"
        + " / "
        + f"{result['split_res_test_rmse']:.4f}"
        + "\n"
        + "  delta(std-res)  : "
        + f"{result['delta_rmse']:+.4f}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for name, sigma, reg in MAIN_DATASETS:
        print("Dataset: " + name)
        try:
            X_train, y_train, X_test, y_test, feature_names = load_main_dataset(name, seed=SEED)
            print(f"  train={X_train.shape[0]} x {X_train.shape[1]}  test={X_test.shape[0]}")
            result = run_split_comparison(
                X_train,
                y_train,
                X_test,
                y_test,
                tag=name,
                sigma=sigma,
                reg=reg,
                feature_names=feature_names,
            )
            all_results.append(result)
            _print_result(result)
        except Exception as exc:
            warnings.warn("Skipping " + name + ": " + str(exc))
            print("  SKIPPED -- " + str(exc))

    for uci_id, tag, target_idx, sigma, reg in BONUS_UCI_DATASETS:
        print(f"Dataset (bonus UCI {uci_id}): " + tag)
        try:
            X_train, y_train, X_test, y_test, feature_names = load_uci_bonus_dataset(
                uci_id,
                target_col_idx=target_idx,
                seed=SEED,
            )
            print(f"  train={X_train.shape[0]} x {X_train.shape[1]}  test={X_test.shape[0]}")
            result = run_split_comparison(
                X_train,
                y_train,
                X_test,
                y_test,
                tag=tag,
                sigma=sigma,
                reg=reg,
                feature_names=feature_names,
            )
            all_results.append(result)
            _print_result(result)
        except Exception as exc:
            warnings.warn("Skipping " + tag + ": " + str(exc))
            print("  SKIPPED -- " + str(exc))

    if not all_results:
        print("\nNo datasets loaded. Exiting.")
        return

    with (OUT_DIR / "bonus_results.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    summary = pd.DataFrame(
        [
            {
                "dataset": result["dataset"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "root_predictor_test_rmse": round(result["root_predictor_test_rmse"], 4),
                "top_std_feature": result["top_std_feature"],
                "top_res_feature": result["top_res_feature"],
                "direction_cosine_abs": round(result["direction_cosine_abs"], 4),
                "split_std_test_rmse": round(result["split_std_test_rmse"], 4),
                "split_res_test_rmse": round(result["split_res_test_rmse"], 4),
                "delta_rmse_std_minus_res": round(result["delta_rmse"], 4),
            }
            for result in all_results
        ]
    )
    summary.to_csv(OUT_DIR / "bonus_summary.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    disagreement = summary[summary["direction_cosine_abs"] < 0.95]
    improvement = summary[summary["delta_rmse_std_minus_res"] > 0]
    print("\nDisagreement (|cos|<0.95): " + str(disagreement["dataset"].tolist()))
    print("Improvement  (delta>0):     " + str(improvement["dataset"].tolist()))
    print("\nArtifacts written to: " + str(OUT_DIR))


if __name__ == "__main__":
    main()
