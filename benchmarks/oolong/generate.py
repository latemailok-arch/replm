"""Generate OOLONG benchmark tasks using real TREC coarse data.

Replicates the OOLONG trec_coarse benchmark (Bertsch et al 2025) with
context sizes matching the paper (2^13 to 2^18 tokens). Each task asks
the model to classify every entry and answer a count or comparison query
— making this O(n) in processing complexity.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from itertools import cycle

from .trec_data import CATEGORIES, TREC_QUESTIONS

# Context sizes matching the paper (approximate token counts).
# Each entry is ~80 chars ≈ 20 tokens.
ENTRY_COUNTS: dict[str, int] = {
    "8k": 400,
    "16k": 800,
    "32k": 1_600,
    "64k": 3_200,
    "128k": 6_400,
    "256k": 12_800,
}

TASKS_PER_SIZE: int = 10


@dataclass
class OolongTask:
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
    """Build context string and count category occurrences.

    Cycles through TREC_QUESTIONS if num_entries exceeds the dataset size.
    """
    pool = list(TREC_QUESTIONS)
    rng.shuffle(pool)
    source = cycle(pool)

    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    lines: list[str] = []

    for _ in range(num_entries):
        q = next(source)
        date = _random_date(rng)
        user_id = rng.randint(1000, 9999)
        lines.append(f"Date: {date} || User: {user_id} || Question: {q.text}")
        counts[q.label] += 1

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


def generate_tasks(seed: int = 42) -> list[OolongTask]:
    """Generate all OOLONG tasks deterministically.

    Returns 10 tasks x 6 sizes = 60 tasks.
    """
    rng = random.Random(seed)
    tasks: list[OolongTask] = []

    for size_label, num_entries in ENTRY_COUNTS.items():
        for i in range(TASKS_PER_SIZE):
            context, counts = _generate_entries(num_entries, rng)

            if i % 2 == 0:
                query, expected = _make_comparison_query(counts, rng)
                query_type = "comparison"
            else:
                query, expected = _make_count_query(counts, rng)
                query_type = "count"

            tasks.append(
                OolongTask(
                    task_id=f"oolong_{size_label}_{i:03d}",
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
