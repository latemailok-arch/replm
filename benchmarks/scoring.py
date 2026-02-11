"""Scoring functions for benchmark evaluation."""

from __future__ import annotations

import re


def score_exact_match(predicted: str, expected: str) -> tuple[bool, float]:
    """Exact string match after normalization."""
    p = _normalize(predicted)
    e = _normalize(expected)
    correct = p == e
    return correct, 1.0 if correct else 0.0


def score_contains_match(predicted: str, expected: str) -> tuple[bool, float]:
    """Check if expected value appears anywhere in predicted text."""
    p = _normalize(predicted)
    e = _normalize(expected)
    correct = e in p
    return correct, 1.0 if correct else 0.0


def score_numeric_tolerance(
    predicted: str, expected: str, tolerance: float = 0.05
) -> tuple[bool, float]:
    """Numeric comparison with relative tolerance (default ±5%)."""
    try:
        p_val = _extract_number(predicted)
        e_val = _extract_number(expected)
    except ValueError:
        return score_exact_match(predicted, expected)

    correct = p_val == 0 if e_val == 0 else abs(p_val - e_val) / abs(e_val) <= tolerance
    return correct, 1.0 if correct else 0.0


def score_comparison(predicted: str, expected: str) -> tuple[bool, float]:
    """Score comparison answers: 'more common than' / 'less common than' / 'same frequency as'."""
    p = _normalize_comparison(predicted)
    e = _normalize_comparison(expected)
    correct = p == e
    return correct, 1.0 if correct else 0.0


def score_oolong_numeric(predicted: str, expected: str) -> tuple[bool, float]:
    """OOLONG scoring: score = 0.75^|y - y_hat| for numerical answers.

    Returns (exact_match, graded_score) where graded_score decays
    exponentially with the absolute difference between predicted and expected.
    """
    try:
        p_val = _extract_number(predicted)
        e_val = _extract_number(expected)
    except ValueError:
        return score_exact_match(predicted, expected)

    diff = abs(e_val - p_val)
    correct = diff == 0
    score = 0.75 ** diff
    return correct, score


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower().strip())


def _extract_number(s: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if match is None:
        raise ValueError(f"No number found in: {s!r}")
    return float(match.group())


def _normalize_comparison(s: str) -> str:
    low = s.lower().strip()
    if "more common" in low or "more frequent" in low:
        return "more common than"
    if "less common" in low or "less frequent" in low:
        return "less common than"
    if "same" in low or "equal" in low:
        return "same frequency as"
    return low
