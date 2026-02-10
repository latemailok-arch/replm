"""Shared budget tracking across recursion depths.

When ``max_recursion_depth > 1``, multiple ``SubCallManager`` instances at
different recursion depths share a single :class:`SharedBudget` so that
sub-call limits and token totals apply globally.
"""

from __future__ import annotations

import threading


class SharedBudget:
    """Thread-safe shared counters for sub-call and token budgets.

    Parameters
    ----------
    max_sub_calls:
        Global limit on total sub-LLM calls across all depths.
    """

    def __init__(self, max_sub_calls: int = 500) -> None:
        self.max_sub_calls = max_sub_calls
        self._call_count: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_input_tokens(self) -> int:
        return self._input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._output_tokens

    def increment_call(self) -> int:
        """Increment call count atomically and return the new count.

        Raises :class:`~rlm.exceptions.MaxSubCallsExceeded` when the
        budget is exhausted.
        """
        with self._lock:
            if self._call_count >= self.max_sub_calls:
                from .exceptions import MaxSubCallsExceeded

                raise MaxSubCallsExceeded(self._call_count, self.max_sub_calls)
            self._call_count += 1
            return self._call_count

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token counts atomically."""
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
