from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

from project9417.reporting import summarize_results


def main() -> None:
    results_df, timing_df = summarize_results()
    print(f"[summary] results_rows={len(results_df)} timing_rows={len(timing_df)}")


if __name__ == "__main__":
    main()
