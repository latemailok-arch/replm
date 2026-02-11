"""Quick GPT-5 benchmark — replicates paper setup (GPT-5 root + GPT-5-mini sub).

S-NIAH:       4 sizes (32K, 130K, 500K, 1M) x 5 tasks = 20 tasks per mode
Aggregation:  2 sizes (100, 500 entries) x 5 tasks = 10 tasks per mode
Total:        60 task runs (30 base + 30 RLM)

Usage:
    uv run python -m benchmarks.run_gpt5
"""

from __future__ import annotations

import time

from benchmarks.config import gpt5_config
from benchmarks.plotting import (
    plot_accuracy_vs_size,
    plot_iterations_vs_size,
    plot_tokens_vs_size,
)
from benchmarks.results import BenchmarkResults
from benchmarks.runner import BenchmarkRunner, Task
from benchmarks.scoring import (
    score_comparison,
    score_contains_match,
    score_numeric_tolerance,
)

TASKS_PER_SIZE = 5

SNIAH_SIZES = ["32K", "130K", "500K", "1M"]
AGG_SIZES = ["100_entries", "500_entries"]


def _sniah_tasks() -> list[Task]:
    from benchmarks.s_niah.generate import generate_tasks

    all_tasks = generate_tasks()
    counts: dict[str, int] = {}
    filtered = []
    for t in all_tasks:
        if t.context_size_label not in SNIAH_SIZES:
            continue
        n = counts.get(t.context_size_label, 0)
        if n >= TASKS_PER_SIZE:
            continue
        counts[t.context_size_label] = n + 1
        filtered.append(t)

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
        for t in filtered
    ]


def _agg_tasks() -> list[Task]:
    from benchmarks.aggregation.generate import generate_tasks

    all_tasks = generate_tasks()
    counts: dict[str, int] = {}
    filtered = []
    for t in all_tasks:
        if t.context_size_label not in AGG_SIZES:
            continue
        n = counts.get(t.context_size_label, 0)
        if n >= TASKS_PER_SIZE:
            continue
        counts[t.context_size_label] = n + 1
        filtered.append(t)

    return [
        Task(
            task_id=t.task_id,
            benchmark="aggregation",
            context_size_label=t.context_size_label,
            context=t.context,
            query=t.query,
            expected=t.expected,
            score_fn=(
                score_comparison if t.query_type == "comparison" else score_numeric_tolerance
            ),
        )
        for t in filtered
    ]


def _run_benchmark(
    config, label: str, tasks: list[Task], modes: list[str],
) -> dict[str, BenchmarkResults]:
    results = {}
    for mode in modes:
        path = config.output_dir / f"gpt5_{label}_{mode}.json"
        runner = BenchmarkRunner(config, path, mode)
        t0 = time.monotonic()
        res = runner.run(tasks)
        elapsed = time.monotonic() - t0
        correct = sum(1 for t in res.tasks if t.correct)
        total = len(res.tasks)
        print(f"\n[{mode}] {label}: {correct}/{total} correct  ({elapsed:.0f}s)")
        results[mode] = res
    return results


def main() -> None:
    config = gpt5_config(max_sub_calls=1000)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root:  {config.root_model}")
    print(f"Sub:   {config.sub_model}")
    print(f"Base reasoning: {config.base_reasoning_effort}")
    print(f"Sizes: S-NIAH {SNIAH_SIZES}, Agg {AGG_SIZES}")
    print(f"Tasks: {TASKS_PER_SIZE} per size")
    print()

    sniah = _sniah_tasks()
    agg = _agg_tasks()

    print(f"Generated {len(sniah)} S-NIAH tasks, {len(agg)} aggregation tasks")

    t_start = time.monotonic()

    sniah_res = _run_benchmark(config, "s_niah", sniah, ["base", "rlm"])
    agg_res = _run_benchmark(config, "agg", agg, ["base", "rlm"])

    total_elapsed = time.monotonic() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total wall-clock: {total_elapsed / 60:.1f} min")

    # -- Summary table --
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for bench, res in [("S-NIAH", sniah_res), ("Aggregation", agg_res)]:
        for mode in ["base", "rlm"]:
            if mode not in res:
                continue
            tasks = res[mode].tasks
            correct = sum(1 for t in tasks if t.correct)
            total = len(tasks)
            errors = sum(1 for t in tasks if t.error)
            pct = correct / total * 100 if total else 0
            print(f"  {bench:15s} {mode:4s}: {correct}/{total} ({pct:.0f}%)  errors={errors}")

    # -- Plots --
    plots = config.output_dir / "plots"

    plot_accuracy_vs_size(
        sniah_res["base"], sniah_res["rlm"],
        "S-NIAH Accuracy: Base vs RLM (GPT-5)",
        plots / "gpt5_sniah_accuracy.png",
    )
    plot_tokens_vs_size(
        sniah_res["base"], sniah_res["rlm"],
        "S-NIAH Token Usage: Base vs RLM (GPT-5)",
        plots / "gpt5_sniah_tokens.png",
    )
    plot_iterations_vs_size(
        sniah_res["rlm"],
        "S-NIAH: RLM Iterations & Sub-calls (GPT-5)",
        plots / "gpt5_sniah_iterations.png",
    )

    plot_accuracy_vs_size(
        agg_res["base"], agg_res["rlm"],
        "Aggregation Accuracy: Base vs RLM (GPT-5)",
        plots / "gpt5_agg_accuracy.png",
    )
    plot_tokens_vs_size(
        agg_res["base"], agg_res["rlm"],
        "Aggregation Token Usage: Base vs RLM (GPT-5)",
        plots / "gpt5_agg_tokens.png",
    )
    plot_iterations_vs_size(
        agg_res["rlm"],
        "Aggregation: RLM Iterations & Sub-calls (GPT-5)",
        plots / "gpt5_agg_iterations.png",
    )

    print("\nPlots saved to", plots)
    print("Done.")


if __name__ == "__main__":
    main()
