from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from project9417.registry import DEFAULT_MODELS
from project9417.scaling import run_scaling_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-sample scaling analysis.")
    parser.add_argument("--dataset", default="job_salary", choices=["job_salary"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_scaling_analysis(
        dataset_name=args.dataset,
        model_names=args.models,
        device=args.device,
        seed=args.seed,
    )
    print(f"[scaling] rows={len(result.dataframe)} metric_plot={result.metric_plot_path}")


if __name__ == "__main__":
    main()
