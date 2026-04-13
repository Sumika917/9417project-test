from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from .experiment import prepare_experiment, preprocess_for_model
from .models import fit_and_select_model
from .paths import INTERPRETABILITY_DIR
from .utils import as_serializable, write_json


@dataclass
class InterpretabilityOutputs:
    ranking_table: pd.DataFrame
    correlation_table: pd.DataFrame
    global_agop: pd.Series
    leaf_importances: pd.DataFrame
    agop_source: str


def _agop_to_diagonal(agop: Any) -> np.ndarray:
    if hasattr(agop, "detach"):
        agop = agop.detach().cpu().numpy()
    arr = np.asarray(agop)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        return np.diag(arr).astype(float)
    raise ValueError(f"Unsupported AGOP shape: {arr.shape}")


def _route_leaf_counts(tree: dict[str, Any], X: np.ndarray) -> list[int]:
    counts: list[int] = []

    def recurse(node: dict[str, Any], subset: np.ndarray) -> None:
        if node["type"] == "leaf":
            counts.append(int(len(subset)))
            return
        direction = node["split_direction"]
        threshold = node["split_point"]
        if hasattr(direction, "detach"):
            direction = direction.detach().cpu().numpy()
        if hasattr(threshold, "detach"):
            threshold = float(threshold.detach().cpu().item())
        direction = np.asarray(direction, dtype=float).reshape(-1)
        projections = subset @ direction
        left_mask = projections <= threshold
        recurse(node["left"], subset[left_mask])
        recurse(node["right"], subset[~left_mask])

    recurse(tree, np.asarray(X, dtype=float))
    return counts


def _collect_leaf_importance_matrices(estimator: Any) -> tuple[list[Any], str]:
    try:
        return estimator.collect_best_agops(), "agop_best_model"
    except AttributeError:
        if hasattr(estimator, "collect_Ms"):
            return estimator.collect_Ms(), "M_fallback"
        raise


def _weighted_global_agop(
    estimator: Any,
    X_train: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.Series, pd.DataFrame, str]:
    agops, agop_source = _collect_leaf_importance_matrices(estimator)
    leaf_rows: list[dict[str, Any]] = []
    weighted_sum = np.zeros(len(feature_names), dtype=float)
    total_weight = 0.0
    offset = 0
    for tree_idx, tree in enumerate(estimator.trees):
        tree_counts = _route_leaf_counts(tree, X_train)
        tree_agops = agops[offset : offset + len(tree_counts)]
        offset += len(tree_counts)
        for leaf_idx, (agop, count) in enumerate(zip(tree_agops, tree_counts)):
            diag = _agop_to_diagonal(agop)[: len(feature_names)]
            weighted_sum += diag * count
            total_weight += count
            row = {"tree_idx": tree_idx, "leaf_idx": leaf_idx, "leaf_count": count}
            for feature_idx in range(len(feature_names)):
                row[feature_names[feature_idx]] = float(diag[feature_idx])
            leaf_rows.append(row)
    global_diag = weighted_sum / max(total_weight, 1.0)
    return pd.Series(global_diag, index=feature_names, name="agop_weighted"), pd.DataFrame(leaf_rows), agop_source


def _permutation_importance_scores(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(random_state)
    baseline = accuracy_score(y, estimator.predict(X))
    scores = np.zeros(X.shape[1], dtype=float)
    for col_idx in range(X.shape[1]):
        shuffled = X.copy()
        shuffled[:, col_idx] = rng.permutation(shuffled[:, col_idx])
        score = accuracy_score(y, estimator.predict(shuffled))
        scores[col_idx] = baseline - score
    return scores


def run_appendicitis_interpretability(seed: int = 42, device: str = "auto", top_k: int = 15) -> InterpretabilityOutputs:
    experiment = prepare_experiment("appendicitis", seed=seed, force_rebuild_splits=False)
    bundle = preprocess_for_model(experiment, "xrfm")
    result = fit_and_select_model(
        "xrfm",
        experiment.prepared_dataset.spec,
        bundle,
        device=device,
        seed=seed,
        collect_xrfm_agops=True,
    )

    X_train = bundle.X_train
    y_train = bundle.y_train
    feature_names = bundle.feature_names

    global_agop, leaf_importances, agop_source = _weighted_global_agop(result.estimator, X_train, feature_names)
    pca = PCA(n_components=1, random_state=seed)
    pca.fit(X_train)
    pca_scores = pd.Series(np.abs(pca.components_[0]), index=feature_names, name="pca_loading")
    mi_scores = pd.Series(
        mutual_info_classif(X_train, y_train, discrete_features=False, random_state=seed),
        index=feature_names,
        name="mutual_info",
    )
    permutation_scores = pd.Series(
        _permutation_importance_scores(result.estimator, X_train, y_train, random_state=seed),
        index=feature_names,
        name="permutation_importance",
    )

    ranking_table = pd.concat([global_agop, pca_scores, mi_scores, permutation_scores], axis=1)
    correlation_table = ranking_table.corr(method="spearman")

    INTERPRETABILITY_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = INTERPRETABILITY_DIR / "appendicitis_rankings.csv"
    correlation_path = INTERPRETABILITY_DIR / "appendicitis_rank_correlations.csv"
    leaf_path = INTERPRETABILITY_DIR / "appendicitis_leaf_importances.csv"
    ranking_table.sort_values("agop_weighted", ascending=False).to_csv(ranking_path, index_label="feature")
    correlation_table.to_csv(correlation_path, index=True)
    leaf_importances.to_csv(leaf_path, index=False)

    top_features = ranking_table.sort_values("agop_weighted", ascending=False).head(top_k).index.tolist()
    melted = (
        ranking_table.loc[top_features, ["agop_weighted", "pca_loading", "mutual_info", "permutation_importance"]]
        .reset_index(names="feature")
        .melt(id_vars="feature", var_name="method", value_name="score")
    )
    plt.figure(figsize=(14, 7))
    sns.barplot(data=melted, x="feature", y="score", hue="method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(INTERPRETABILITY_DIR / "appendicitis_top_features.png", dpi=200)
    plt.close()

    heatmap = leaf_importances[[col for col in top_features if col in leaf_importances.columns]].copy()
    if not heatmap.empty:
        heatmap.index = [f"tree{row.tree_idx}_leaf{row.leaf_idx}" for row in leaf_importances.itertuples()]
        plt.figure(figsize=(10, max(5, len(heatmap) * 0.35)))
        sns.heatmap(heatmap, annot=True, cmap="mako")
        plt.tight_layout()
        plt.savefig(INTERPRETABILITY_DIR / "appendicitis_agop_heatmap.png", dpi=200)
        plt.close()

    write_json(
        INTERPRETABILITY_DIR / "appendicitis_interpretability_summary.json",
        {
            "seed": seed,
            "top_features": top_features,
            "agop_source": agop_source,
            "validation_metrics": as_serializable(result.validation_metrics),
            "test_metrics": as_serializable(result.test_metrics),
            "artifacts": {
                "ranking_path": str(ranking_path),
                "correlation_path": str(correlation_path),
                "leaf_path": str(leaf_path),
            },
        },
    )
    return InterpretabilityOutputs(
        ranking_table=ranking_table,
        correlation_table=correlation_table,
        global_agop=global_agop,
        leaf_importances=leaf_importances,
        agop_source=agop_source,
    )
