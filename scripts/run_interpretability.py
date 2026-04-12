from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from project9417.interpretability import run_appendicitis_interpretability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run appendicitis interpretability analysis.")
    parser.add_argument("--dataset", default="appendicitis", choices=["appendicitis"])
    parser.add_argument("--model", default="xrfm", choices=["xrfm"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_appendicitis_interpretability(seed=args.seed, device=args.device, top_k=args.top_k)
    print(
        f"[interpretability] features={len(outputs.ranking_table)} "
        f"leaf_rows={len(outputs.leaf_importances)}"
    )


if __name__ == "__main__":
    main()
