"""Benchmark runner — executes tasks in base or RLM mode with resumption."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from rlm import RLMWrapper

from .config import BenchmarkConfig
from .results import BenchmarkResults, TaskResult


@dataclass
class Task:
    """A single benchmark task."""

    task_id: str
    benchmark: str
    context_size_label: str
    context: str | list[str]
    query: str
    expected: str
    score_fn: Callable[[str, str], tuple[bool, float]]


class BenchmarkRunner:
    """Runs tasks in base or RLM mode with crash-safe incremental saving."""

    def __init__(
        self,
        config: BenchmarkConfig,
        results_path: Path,
        mode: str,
    ) -> None:
        self._config = config
        self._results_path = results_path
        self._mode = mode

        self._client: OpenAI = config.make_openai_client()
        self._rlm: RLMWrapper | None = None
        if mode == "rlm":
            self._rlm = config.make_rlm_wrapper()

        self._results = BenchmarkResults.load_or_create(
            results_path, benchmark="", model=config.root_model
        )

    def run(self, tasks: list[Task]) -> BenchmarkResults:
        completed = self._results.completed_task_ids()
        remaining = [t for t in tasks if t.task_id not in completed]
        total = len(tasks)
        done = len(completed)

        print(f"[{self._mode}] {done}/{total} done, {len(remaining)} remaining")

        for i, task in enumerate(remaining):
            label = f"[{self._mode}] ({done + i + 1}/{total}) {task.task_id}"
            print(f"{label} ...", end=" ", flush=True)

            result = self._execute_with_retry(task)
            self._results.add(result)
            self._results.save(self._results_path)

            if result.error:
                print(f"ERR: {result.error[:60]}")
            elif result.correct:
                print(f"OK  ({result.elapsed_seconds:.1f}s)")
            else:
                print(f"WRONG  pred={result.predicted[:40]!r}  ({result.elapsed_seconds:.1f}s)")

        return self._results

    def _execute_with_retry(self, task: Task) -> TaskResult:
        last_error: str | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                return self._execute_task(task)
            except Exception as exc:
                last_error = f"attempt {attempt}: {exc}"
                if attempt < self._config.max_retries:
                    print(f"\n  [retry {attempt}/{self._config.max_retries}] {exc}")
                    time.sleep(self._config.retry_delay)

        return TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            mode=self._mode,
            context_size_label=task.context_size_label,
            context_chars=_context_len(task.context),
            query=task.query,
            expected=task.expected,
            predicted="",
            correct=False,
            score=0.0,
            error=last_error,
        )

    def _execute_task(self, task: Task) -> TaskResult:
        start = time.monotonic()
        ctx_chars = _context_len(task.context)

        if self._mode == "base":
            return self._execute_base(task, start, ctx_chars)
        return self._execute_rlm(task, start, ctx_chars)

    def _execute_base(self, task: Task, start: float, ctx_chars: int) -> TaskResult:
        if ctx_chars > self._config.base_context_limit_chars:
            return TaskResult(
                task_id=task.task_id,
                benchmark=task.benchmark,
                mode="base",
                context_size_label=task.context_size_label,
                context_chars=ctx_chars,
                query=task.query,
                expected=task.expected,
                predicted="",
                correct=False,
                score=0.0,
                error="context_exceeded",
                elapsed_seconds=time.monotonic() - start,
            )

        context_str = task.context if isinstance(task.context, str) else "\n\n".join(task.context)
        user_msg = f"Context:\n{context_str}\n\nQuestion: {task.query}"

        kwargs: dict[str, object] = dict(
            model=self._config.root_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided context."
                    " Be concise and precise.",
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=self._config.temperature,
            max_tokens=16384,
        )
        if self._config.reasoning_effort is not None:
            kwargs["extra_body"] = {
                "reasoning_effort": self._config.reasoning_effort,
            }
        response = self._client.chat.completions.create(**kwargs)

        predicted = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        correct, score = task.score_fn(predicted, task.expected)

        return TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            mode="base",
            context_size_label=task.context_size_label,
            context_chars=ctx_chars,
            query=task.query,
            expected=task.expected,
            predicted=predicted,
            correct=correct,
            score=score,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            elapsed_seconds=time.monotonic() - start,
        )

    def _execute_rlm(self, task: Task, start: float, ctx_chars: int) -> TaskResult:
        assert self._rlm is not None
        resp = self._rlm.generate(query=task.query, context=task.context)
        predicted = resp.answer.strip()
        correct, score = task.score_fn(predicted, task.expected)

        return TaskResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            mode="rlm",
            context_size_label=task.context_size_label,
            context_chars=ctx_chars,
            query=task.query,
            expected=task.expected,
            predicted=predicted,
            correct=correct,
            score=score,
            iterations=resp.iterations,
            sub_calls=resp.sub_calls,
            total_input_tokens=resp.total_input_tokens,
            total_output_tokens=resp.total_output_tokens,
            cost=resp.cost,
            elapsed_seconds=time.monotonic() - start,
        )


def _context_len(ctx: str | list[str]) -> int:
    if isinstance(ctx, str):
        return len(ctx)
    return sum(len(c) for c in ctx)
