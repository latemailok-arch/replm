"""Smoke test: 2 tasks per benchmark at the smallest size, both modes.

Usage:
    uv run python -m benchmarks.smoke_test
"""

from __future__ import annotations

import json
import time

from benchmarks.config import BenchmarkConfig
from benchmarks.runner import BenchmarkRunner, Task
from benchmarks.scoring import (
    score_comparison,
    score_contains_match,
    score_numeric_tolerance,
)

LIMIT = 2  # tasks per benchmark per mode


def _sniah_tasks() -> list[Task]:
    from benchmarks.s_niah.generate import generate_tasks

    tasks = generate_tasks()
    smallest = [t for t in tasks if t.context_size_label == "32K"][:LIMIT]
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
        for t in smallest
    ]


def _agg_tasks() -> list[Task]:
    from benchmarks.aggregation.generate import generate_tasks

    tasks = generate_tasks()
    smallest = [t for t in tasks if t.context_size_label == "100_entries"][:LIMIT]
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
        for t in smallest
    ]


def main() -> None:
    config = BenchmarkConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root model:  {config.root_model}")
    print(f"Sub model:   {config.sub_model}")
    print(f"Base URL:    {config.base_url}")
    print()

    sniah = _sniah_tasks()
    agg = _agg_tasks()

    all_results: list[dict] = []

    for label, tasks in [("S-NIAH 32K", sniah), ("Aggregation 100ent", agg)]:
        for mode in ["base", "rlm"]:
            print(f"\n{'=' * 60}")
            print(f"  {label} — {mode.upper()} mode ({len(tasks)} tasks)")
            print(f"{'=' * 60}")

            path = config.output_dir / f"smoke_{label.replace(' ', '_')}_{mode}.json"
            runner = BenchmarkRunner(config, path, mode)
            t0 = time.monotonic()
            results = runner.run(tasks)
            elapsed = time.monotonic() - t0

            for r in results.tasks:
                summary = {
                    "task_id": r.task_id,
                    "mode": r.mode,
                    "correct": r.correct,
                    "score": r.score,
                    "predicted": r.predicted[:80],
                    "expected": r.expected[:80],
                    "input_tokens": r.total_input_tokens,
                    "output_tokens": r.total_output_tokens,
                    "iterations": r.iterations,
                    "sub_calls": r.sub_calls,
                    "elapsed_s": round(r.elapsed_seconds, 1),
                    "error": r.error,
                }
                all_results.append(summary)
                print(f"\n  {r.task_id}:")
                print(f"    correct={r.correct}  score={r.score}")
                print(f"    predicted={r.predicted[:60]!r}")
                print(f"    expected={r.expected!r}")
                print(f"    tokens: {r.total_input_tokens} in / {r.total_output_tokens} out")
                print(f"    iterations={r.iterations}  sub_calls={r.sub_calls}")
                print(f"    elapsed={r.elapsed_seconds:.1f}s")
                if r.error:
                    print(f"    ERROR: {r.error}")

            print(f"\n  Batch elapsed: {elapsed:.1f}s")

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    header = (
        f"{'Task':<30} {'Mode':<6} {'OK?':<5} "
        f"{'InTok':>8} {'OutTok':>8} {'Iters':>6} {'SubC':>6} {'Time':>7}"
    )
    print(header)
    print("-" * 85)
    for r in all_results:
        iters = r["iterations"] if r["iterations"] is not None else "-"
        subc = r["sub_calls"] if r["sub_calls"] is not None else "-"
        print(
            f"{r['task_id']:<30} {r['mode']:<6} "
            f"{'Y' if r['correct'] else 'N':<5} "
            f"{r['input_tokens']:>8} {r['output_tokens']:>8} "
            f"{str(iters):>6} {str(subc):>6} "
            f"{r['elapsed_s']:>6.1f}s"
        )

    total_in = sum(r["input_tokens"] for r in all_results)
    total_out = sum(r["output_tokens"] for r in all_results)
    total_time = sum(r["elapsed_s"] for r in all_results)
    print("-" * 85)
    print(
        f"{'TOTAL':<37} {'':>4} {total_in:>8} {total_out:>8} {'':>6} {'':>6} {total_time:>6.1f}s"
    )

    # Save raw results
    smoke_path = config.output_dir / "smoke_summary.json"
    smoke_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nDetailed results saved to: {smoke_path}")


if __name__ == "__main__":
    main()
