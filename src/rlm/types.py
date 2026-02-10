"""Public response and event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HistoryEntry:
    """A single entry in the RLM execution trace."""

    role: str
    """One of 'root', 'repl_output', 'sub_call'."""

    content: str
    """Full content (not truncated)."""

    truncated_content: str
    """What the root model actually saw."""

    iteration: int
    """Which root-loop iteration produced this entry."""


@dataclass
class RLMResponse:
    """The result of a single ``RLMWrapper.generate()`` call."""

    answer: str
    """The final answer string."""

    iterations: int
    """Number of root loop iterations used."""

    sub_calls: int
    """Total sub-LLM invocations."""

    total_input_tokens: int
    """Aggregated input tokens across all LLM calls (root + sub)."""

    total_output_tokens: int
    """Aggregated output tokens across all LLM calls (root + sub)."""

    cost_per_input_token: float = 0.0
    """Cost per input token in USD (set via ``RLMConfig.cost_per_input_token``)."""

    cost_per_output_token: float = 0.0
    """Cost per output token in USD (set via ``RLMConfig.cost_per_output_token``)."""

    history: list[HistoryEntry] = field(default_factory=list)
    """Full execution trace for debugging / observability."""

    repl_variables: dict[str, str] = field(default_factory=dict)
    """Final REPL state: variable names mapped to their ``repr``."""

    @property
    def cost(self) -> float:
        """Estimated total cost in USD based on token usage and configured pricing.

        Returns 0.0 if pricing was not configured.
        """
        return (
            self.total_input_tokens * self.cost_per_input_token
            + self.total_output_tokens * self.cost_per_output_token
        )


@dataclass
class RLMEvent:
    """Passed to the ``on_event`` callback for streaming observability."""

    type: str
    """Event type.

    One of: ``iteration_start``, ``code_generated``, ``code_executed``,
    ``sub_call_start``, ``sub_call_end``, ``final_answer``.
    """

    iteration: int
    """Current root-loop iteration."""

    preview: str
    """Short human-readable preview of what happened."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Full details (type-specific)."""
