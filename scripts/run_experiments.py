from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from project9417.experiment import run_experiments
from project9417.registry import ALL_DATASETS, DEFAULT_MODELS
from project9417.reporting import summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main model experiments.")
    parser.add_argument("--dataset", default="all", choices=["all", *ALL_DATASETS])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(ALL_DATASETS) if args.dataset == "all" else [args.dataset]
    records = run_experiments(
        dataset_names=datasets,
        model_names=args.models,
        seed=args.seed,
        device=args.device,
        force_rebuild_splits=args.force_rebuild_splits,
    )
    summarize_results()
    print(f"[completed] wrote {len(records)} experiment run records")


if __name__ == "__main__":
    main()
