"""Run the S-NIAH benchmark.

Usage:
    python -m benchmarks.s_niah.run --mode base
    python -m benchmarks.s_niah.run --mode rlm
    python -m benchmarks.s_niah.run --mode both
    python -m benchmarks.s_niah.run --mode rlm --sizes 32K 65K
"""

from __future__ import annotations

import argparse

from benchmarks.config import BenchmarkConfig
from benchmarks.runner import BenchmarkRunner, Task
from benchmarks.scoring import score_contains_match

from .generate import generate_tasks


def _to_runner_tasks(sniah_tasks: list) -> list[Task]:
    return [
        Task(
            task_id=t.task_id,
            benchmark="s_niah",
            context_size_label=t.context_size_label,
            context=t.context,
            query=t.query,
            expected=t.value,
            score_fn=score_contains_match,
        )
        for t in sniah_tasks
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run S-NIAH benchmark")
    parser.add_argument("--mode", choices=["base", "rlm", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", nargs="*", default=None)
    args = parser.parse_args(argv)

    config = BenchmarkConfig()
    tasks = generate_tasks(seed=args.seed)

    if args.sizes:
        tasks = [t for t in tasks if t.context_size_label in args.sizes]

    runner_tasks = _to_runner_tasks(tasks)
    modes = ["base", "rlm"] if args.mode == "both" else [args.mode]

    for mode in modes:
        path = config.output_dir / f"s_niah_{mode}.json"
        runner = BenchmarkRunner(config, path, mode)
        results = runner.run(runner_tasks)
        correct = sum(1 for t in results.tasks if t.correct)
        print(f"\n[{mode}] Done. {correct}/{len(results.tasks)} correct")
        print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
