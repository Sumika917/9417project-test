from __future__ import annotations

from typing import Any


def ensure_xrfm_runtime_patch() -> None:
    try:
        import xrfm.tree_utils as tree_utils
        import xrfm.xrfm as xrfm_module
    except ImportError:
        return

    if getattr(xrfm_module.xRFM, "_project9417_agop_patch", False):
        return

    original_get_param_tree = tree_utils.get_param_tree

    def patched_get_param_tree(tree: dict[str, Any], is_root: bool = False) -> dict[str, Any]:
        param_tree = original_get_param_tree(tree, is_root=is_root)
        if tree["type"] == "leaf":
            agop = getattr(tree["model"], "agop_best_model", None)
            if agop is not None:
                param_tree["agop_best_model"] = agop
        return param_tree

    def patched_build_leaf_models_from_param_trees(self: Any, param_trees: list[dict[str, Any]]) -> None:
        self.trees = []

        def set_leaf_model_single_tree(tree: dict[str, Any]) -> dict[str, Any]:
            if tree["type"] == "leaf":
                leaf_model = xrfm_module.RFM(
                    **self.rfm_params["model"],
                    categorical_info=self.categorical_info,
                    device=self.device,
                    **self.extra_rfm_params_,
                )
                leaf_model.kernel_obj.bandwidth = tree["bandwidth"]
                leaf_model.weights = tree["weights"]
                leaf_model.M = tree["M"]
                leaf_model.sqrtM = tree["sqrtM"]
                if "agop_best_model" in tree:
                    leaf_model.agop_best_model = tree["agop_best_model"]
                tree["model"] = leaf_model
                return tree
            tree["left"] = set_leaf_model_single_tree(tree["left"])
            tree["right"] = set_leaf_model_single_tree(tree["right"])
            tree.setdefault("adaptive_temp_scaling", 1.0)
            return tree

        for param_tree in param_trees:
            tree = set_leaf_model_single_tree(param_tree)
            self.trees.append(tree)
            self._ensure_tree_cache(tree)

    tree_utils.get_param_tree = patched_get_param_tree
    xrfm_module.get_param_tree = patched_get_param_tree
    xrfm_module.xRFM._build_leaf_models_from_param_trees = patched_build_leaf_models_from_param_trees
    xrfm_module.xRFM._project9417_agop_patch = True
