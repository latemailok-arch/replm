"""Generate Synthetic Aggregation (OOLONG-lite) benchmark tasks.

Each task asks about the frequency distribution of semantic categories
across a dataset of trivia questions. The model must classify every
entry to answer correctly — making this O(n) in processing complexity.
Based on OOLONG trec_coarse (Bertsch et al 2025).
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass

from .questions import CATEGORIES, TRIVIA_BANK

ENTRY_COUNTS: dict[str, int] = {
    "100_entries": 100,
    "500_entries": 500,
    "1000_entries": 1_000,
    "2000_entries": 2_000,
    "5000_entries": 5_000,
}

TASKS_PER_SIZE: int = 20


@dataclass
class AggregationTask:
    task_id: str
    context_size_label: str
    num_entries: int
    query: str
    expected: str
    query_type: str  # "comparison" or "count"
    category_counts: dict[str, int]
    context: str
    seed: int


def _random_date(rng: random.Random) -> str:
    start = datetime.date(2022, 1, 1)
    delta = (datetime.date(2024, 12, 31) - start).days
    d = start + datetime.timedelta(days=rng.randint(0, delta))
    return d.isoformat()


def _generate_entries(num_entries: int, rng: random.Random) -> tuple[str, dict[str, int]]:
    questions = rng.choices(TRIVIA_BANK, k=num_entries)
    counts: dict[str, int] = {c: 0 for c in CATEGORIES}

    lines: list[str] = []
    for q in questions:
        date = _random_date(rng)
        user_id = rng.randint(1000, 9999)
        lines.append(f"Date: {date} || User: {user_id} || Question: {q.question}")
        counts[q.category] += 1

    return "\n".join(lines), counts


def _make_comparison_query(counts: dict[str, int], rng: random.Random) -> tuple[str, str]:
    a, b = rng.sample(list(counts.keys()), 2)
    if counts[a] > counts[b]:
        expected = "more common than"
    elif counts[a] < counts[b]:
        expected = "less common than"
    else:
        expected = "same frequency as"

    query = (
        "Each question in the data above can be classified into exactly one "
        f"of these categories: {', '.join(CATEGORIES)}. "
        "The categories are NOT provided in the data — you must infer them "
        "from the semantics of each question. "
        f"Is the label '{a}' more common, less common, or the same frequency "
        f"as the label '{b}'? "
        "Answer with exactly one of: 'more common than', 'less common than', "
        "or 'same frequency as'."
    )
    return query, expected


def _make_count_query(counts: dict[str, int], rng: random.Random) -> tuple[str, str]:
    cat = rng.choice(list(counts.keys()))
    expected = str(counts[cat])
    query = (
        "Each question in the data above can be classified into exactly one "
        f"of these categories: {', '.join(CATEGORIES)}. "
        "The categories are NOT provided in the data — you must infer them "
        "from the semantics of each question. "
        f"How many entries have the label '{cat}'? "
        "Answer with only the number."
    )
    return query, expected


def generate_tasks(seed: int = 42) -> list[AggregationTask]:
    """Generate all aggregation tasks deterministically.

    Returns 20 tasks x 5 sizes = 100 tasks.
    """
    rng = random.Random(seed)
    tasks: list[AggregationTask] = []
    task_num = 0

    for size_label, num_entries in ENTRY_COUNTS.items():
        for i in range(TASKS_PER_SIZE):
            task_num += 1
            context, counts = _generate_entries(num_entries, rng)

            if i % 2 == 0:
                query, expected = _make_comparison_query(counts, rng)
                query_type = "comparison"
            else:
                query, expected = _make_count_query(counts, rng)
                query_type = "count"

            tasks.append(
                AggregationTask(
                    task_id=f"agg_{size_label}_{i:03d}",
                    context_size_label=size_label,
                    num_entries=num_entries,
                    query=query,
                    expected=expected,
                    query_type=query_type,
                    category_counts=counts,
                    context=context,
                    seed=seed,
                )
            )

    return tasks
