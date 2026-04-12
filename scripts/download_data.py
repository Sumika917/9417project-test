from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from project9417.datasets import download_dataset
from project9417.registry import ALL_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument("--dataset", default="all", choices=["all", *ALL_DATASETS])
    parser.add_argument("--no-manual-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(ALL_DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        path = download_dataset(dataset, allow_manual_fallback=not args.no_manual_fallback)
        print(f"[downloaded] {dataset}: {path}")


if __name__ == "__main__":
    main()
