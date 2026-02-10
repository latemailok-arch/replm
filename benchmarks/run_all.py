"""Run all benchmarks and generate plots.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --benchmark s_niah --mode rlm
    python -m benchmarks.run_all --plot-only
"""

from __future__ import annotations

import argparse
import sys

from .config import BenchmarkConfig
from .plotting import plot_accuracy_vs_size, plot_iterations_vs_size, plot_tokens_vs_size
from .results import BenchmarkResults


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all RLM benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=["s_niah", "aggregation", "all"],
        default="all",
    )
    parser.add_argument("--mode", choices=["base", "rlm", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip running, just regenerate plots from saved results",
    )
    args = parser.parse_args()

    config = BenchmarkConfig()

    if not args.plot_only:
        run_args = ["--mode", args.mode, "--seed", str(args.seed)]
        if args.sizes:
            run_args += ["--sizes"] + args.sizes

        if args.benchmark in ("s_niah", "all"):
            print("\n" + "=" * 60)
            print("  S-NIAH Benchmark")
            print("=" * 60)
            sys.argv = ["benchmarks.s_niah.run"] + run_args
            from .s_niah.run import main as run_sniah

            run_sniah()

        if args.benchmark in ("aggregation", "all"):
            print("\n" + "=" * 60)
            print("  Synthetic Aggregation Benchmark")
            print("=" * 60)
            sys.argv = ["benchmarks.aggregation.run"] + run_args
            from .aggregation.run import main as run_agg

            run_agg()

    _generate_plots(config)


def _generate_plots(config: BenchmarkConfig) -> None:
    plots_dir = config.output_dir / "plots"

    for bench, title_prefix in [
        ("s_niah", "S-NIAH"),
        ("aggregation", "Synthetic Aggregation"),
    ]:
        base_path = config.output_dir / f"{bench}_base.json"
        rlm_path = config.output_dir / f"{bench}_rlm.json"

        if base_path.exists() and rlm_path.exists():
            base = BenchmarkResults.load(base_path)
            rlm = BenchmarkResults.load(rlm_path)

            plot_accuracy_vs_size(
                base,
                rlm,
                title=f"{title_prefix}: Accuracy vs Context Size",
                output_path=plots_dir / f"{bench}_accuracy.png",
            )
            plot_tokens_vs_size(
                base,
                rlm,
                title=f"{title_prefix}: Token Usage vs Context Size",
                output_path=plots_dir / f"{bench}_tokens.png",
            )
            plot_iterations_vs_size(
                rlm,
                title=f"{title_prefix}: RLM Iterations & Sub-calls",
                output_path=plots_dir / f"{bench}_iterations.png",
            )
        elif rlm_path.exists():
            rlm = BenchmarkResults.load(rlm_path)
            plot_iterations_vs_size(
                rlm,
                title=f"{title_prefix}: RLM Iterations & Sub-calls",
                output_path=plots_dir / f"{bench}_iterations.png",
            )

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
