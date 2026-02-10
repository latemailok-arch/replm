"""Generate S-NIAH benchmark tasks.

Each task embeds a single needle into a haystack of diverse filler sentences
at a random position. Based on the RULER S-NIAH format (Hsieh et al 2024).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .filler import FILLER_SENTENCES

NEEDLE_KEYS: list[str] = [
    "telescope",
    "algorithm",
    "cathedral",
    "molecule",
    "symphony",
    "glacier",
    "pentagon",
    "carousel",
    "blueprint",
    "labyrinth",
    "tornado",
    "compass",
    "volcano",
    "pendulum",
    "mercury",
    "spectrum",
    "catalyst",
    "meridian",
    "parabola",
    "chrysalis",
    "umbrella",
    "fortress",
    "satellite",
    "corridor",
    "avalanche",
    "calendar",
    "elephant",
    "harmonica",
    "trapezoid",
    "reservoir",
    "cinnamon",
    "silhouette",
    "barricade",
    "telescope",
    "hemisphere",
    "mandolin",
    "sandstone",
    "perimeter",
    "turbine",
    "monument",
]

CONTEXT_SIZES: dict[str, int] = {
    "32K": 32_000,
    "65K": 65_000,
    "130K": 130_000,
    "260K": 260_000,
    "500K": 500_000,
    "1M": 1_000_000,
}

TASKS_PER_SIZE: int = 20


@dataclass
class SNIAHTask:
    task_id: str
    context_size_label: str
    target_chars: int
    key: str
    value: str
    needle: str
    query: str
    needle_position_frac: float
    context: str
    seed: int


def _build_haystack(target_chars: int, rng: random.Random) -> str:
    """Build a haystack by shuffling and repeating filler sentences."""
    sentences = list(FILLER_SENTENCES)
    parts: list[str] = []
    current_len = 0
    while current_len < target_chars:
        rng.shuffle(sentences)
        for s in sentences:
            parts.append(s)
            current_len += len(s) + 1
            if current_len >= target_chars:
                break
    return " ".join(parts)[:target_chars]


def generate_tasks(seed: int = 42) -> list[SNIAHTask]:
    """Generate all S-NIAH tasks deterministically.

    Returns 20 tasks x 6 sizes = 120 tasks.
    """
    rng = random.Random(seed)
    tasks: list[SNIAHTask] = []
    task_num = 0

    for size_label, target_chars in CONTEXT_SIZES.items():
        for i in range(TASKS_PER_SIZE):
            task_num += 1

            key = f"{NEEDLE_KEYS[task_num % len(NEEDLE_KEYS)]}_{task_num}"
            value = str(rng.randint(1_000_000, 9_999_999))

            needle = f"The special magic number for '{key}' is: {value}."
            query = (
                f"What is the special magic number for '{key}'? "
                "Return only the number, nothing else."
            )

            haystack = _build_haystack(target_chars - len(needle) - 2, rng)

            # Insert at random position, snapping to word boundary.
            frac = rng.random()
            pos = int(len(haystack) * frac)
            while pos < len(haystack) and haystack[pos] != " ":
                pos += 1

            context = haystack[:pos] + " " + needle + " " + haystack[pos:]

            tasks.append(
                SNIAHTask(
                    task_id=f"s_niah_{size_label}_{i:03d}",
                    context_size_label=size_label,
                    target_chars=target_chars,
                    key=key,
                    value=value,
                    needle=needle,
                    query=query,
                    needle_position_frac=frac,
                    context=context,
                    seed=seed,
                )
            )

    return tasks
