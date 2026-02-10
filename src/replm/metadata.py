"""Truncation and metadata helpers.

The root model must never see the full stdout or the full user context inside
its context window.  These helpers produce short metadata summaries that tell
the model *about* the content (length, a short prefix, etc.) without copying
the content itself into the LLM's token budget.
"""

from __future__ import annotations


def make_metadata(content: str, prefix_chars: int = 1000) -> str:
    """Create a truncated metadata string for the root model.

    If *content* fits within *prefix_chars* it is returned in full.
    Otherwise the first *prefix_chars* characters are shown with a
    ``[truncated]`` marker.
    """
    if len(content) <= prefix_chars:
        return content

    return (
        f"[Output: {len(content):,} chars total] "
        f"First {prefix_chars:,} chars:\n"
        f"{content[:prefix_chars]}\n"
        f"[truncated]"
    )


def context_type_label(context: str | list[str]) -> str:
    """Return a human-friendly label for the context type."""
    if isinstance(context, str):
        return "string"
    return f"list of {len(context)} strings"


def context_total_length(context: str | list[str]) -> int:
    """Return the total character count of the context."""
    if isinstance(context, str):
        return len(context)
    return sum(len(c) for c in context)


def context_chunk_lengths(context: str | list[str]) -> list[int]:
    """Return per-chunk character lengths."""
    if isinstance(context, str):
        return [len(context)]
    return [len(c) for c in context]
