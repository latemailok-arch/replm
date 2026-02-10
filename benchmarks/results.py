"""JSON-based result storage with resumption support."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    """Result of a single benchmark task."""

    task_id: str
    benchmark: str
    mode: str  # "base" or "rlm"
    context_size_label: str
    context_chars: int
    query: str
    expected: str
    predicted: str
    correct: bool
    score: float
    error: str | None = None

    iterations: int | None = None
    sub_calls: int | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost: float = 0.0
    elapsed_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


@dataclass
class BenchmarkResults:
    """Container for all results of a benchmark run."""

    benchmark: str
    model: str
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    tasks: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def completed_task_ids(self) -> set[str]:
        return {t.task_id for t in self.tasks}

    def add(self, result: TaskResult) -> None:
        self.tasks.append(result)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> BenchmarkResults:
        with open(path) as f:
            data = json.load(f)
        tasks = [TaskResult(**t) for t in data.pop("tasks", [])]
        return cls(**data, tasks=tasks)

    @classmethod
    def load_or_create(cls, path: Path, benchmark: str, model: str) -> BenchmarkResults:
        if path.exists():
            return cls.load(path)
        return cls(benchmark=benchmark, model=model)
