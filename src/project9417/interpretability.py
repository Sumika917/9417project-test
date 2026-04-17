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

METHOD_COLUMNS = ["agop_weighted", "pca_loading", "mutual_info", "permutation_importance"]
METHOD_LABELS = {
    "agop_weighted": "Weighted AGOP",
    "pca_loading": "PCA loading",
    "mutual_info": "Mutual information",
    "permutation_importance": "Permutation importance",
}


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

    agop_top_features = ranking_table.sort_values("agop_weighted", ascending=False).head(top_k).index.tolist()
    method_ranks = ranking_table[METHOD_COLUMNS].rank(ascending=False, method="average")
    rank_denominator = max(len(ranking_table) - 1, 1)
    rank_percentiles = 1.0 - (method_ranks - 1.0) / rank_denominator
    average_rank = method_ranks.mean(axis=1).sort_values()
    consensus_top_features = average_rank.head(top_k).index.tolist()
    selected_features = list(dict.fromkeys(agop_top_features + consensus_top_features))
    ordered_features = average_rank.loc[selected_features].sort_values().index.tolist()

    rank_heatmap = rank_percentiles.loc[ordered_features, METHOD_COLUMNS].rename(columns=METHOD_LABELS)
    fig, axis = plt.subplots(figsize=(10, max(6, len(rank_heatmap) * 0.42)))
    sns.heatmap(rank_heatmap, cmap="crest", vmin=0.0, vmax=1.0, linewidths=0.4, cbar_kws={"label": "Rank percentile"}, ax=axis)
    axis.set_title("Appendicitis: Relative importance ranking agreement")
    axis.set_xlabel("Method")
    axis.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(INTERPRETABILITY_DIR / "appendicitis_top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    leaf_feature_columns = [col for col in agop_top_features if col in leaf_importances.columns]
    leaf_plot_df = leaf_importances.sort_values(["leaf_count", "tree_idx", "leaf_idx"], ascending=[False, True, True]).reset_index(drop=True)
    if len(leaf_plot_df) > 1 and leaf_feature_columns:
        heatmap = leaf_plot_df[leaf_feature_columns].copy()
        heatmap.index = [
            f"tree{row.tree_idx}_leaf{row.leaf_idx} (n={int(row.leaf_count)})"
            for row in leaf_plot_df.itertuples()
        ]
        fig, axis = plt.subplots(figsize=(11, max(5, len(heatmap) * 0.55)))
        sns.heatmap(
            heatmap,
            cmap="mako",
            linewidths=0.3,
            cbar_kws={"label": "AGOP diagonal importance"},
            ax=axis,
        )
        axis.set_title("Appendicitis: xRFM leaf-level AGOP heatmap")
        axis.set_xlabel("Feature")
        axis.set_ylabel("Leaf")
        fig.tight_layout()
        fig.savefig(INTERPRETABILITY_DIR / "appendicitis_agop_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    elif leaf_feature_columns:
        row = leaf_plot_df.iloc[0]
        profile = (
            pd.Series({feature: float(row[feature]) for feature in leaf_feature_columns})
            .sort_values(ascending=True)
        )
        fig, axis = plt.subplots(figsize=(10, max(5, len(profile) * 0.4)))
        axis.barh(profile.index, profile.values, color="#54A24B")
        axis.set_title(f"Appendicitis: Single-leaf xRFM AGOP profile (n={int(row['leaf_count'])})")
        axis.set_xlabel("AGOP diagonal importance")
        axis.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(INTERPRETABILITY_DIR / "appendicitis_agop_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    write_json(
        INTERPRETABILITY_DIR / "appendicitis_interpretability_summary.json",
        {
            "seed": seed,
            "top_features": ordered_features,
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
