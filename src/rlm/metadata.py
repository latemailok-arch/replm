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


def make_context_metadata(context: str | list[str], prefix_chars: int = 500) -> str:
    """Create initial metadata about the user's context.

    This is what the root model sees on the very first turn instead of the
    actual context.
    """
    if isinstance(context, str):
        total = len(context)
        prefix = context[:prefix_chars]
        lines = [
            f"Your context is a string with {total:,} total characters.",
            f"It is stored in the variable `context` in your REPL environment.",
        ]
        if total > prefix_chars:
            lines.append(f"First {prefix_chars:,} characters:\n{prefix}\n[truncated]")
        else:
            lines.append(f"Full content:\n{prefix}")
        return "\n".join(lines)

    # list[str]
    chunk_lengths = [len(c) for c in context]
    total = sum(chunk_lengths)
    lines = [
        f"Your context is a list of {len(context):,} strings "
        f"with {total:,} total characters.",
        f"It is stored in the variable `context` in your REPL environment.",
        f"Chunk lengths: {chunk_lengths}",
    ]
    # Show a prefix of the first chunk.
    if context:
        first_prefix = context[0][:prefix_chars]
        lines.append(f"First {prefix_chars:,} chars of chunk 0:\n{first_prefix}")
        if len(context[0]) > prefix_chars:
            lines.append("[truncated]")
    return "\n".join(lines)


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
