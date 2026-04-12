from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from project9417.datasets import prepare_dataset
from project9417.registry import ALL_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets into processed tables and metadata.")
    parser.add_argument("--dataset", default="all", choices=["all", *ALL_DATASETS])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(ALL_DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepared = prepare_dataset(dataset, force=args.force)
        print(
            f"[prepared] {dataset}: rows={len(prepared.dataframe)} "
            f"features={len(prepared.feature_columns)} target={prepared.target_column}"
        )


if __name__ == "__main__":
    main()
