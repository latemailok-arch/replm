"""Sub-call manager — creates the ``llm_query`` function injected into the REPL."""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from .config import RLMConfig
from .exceptions import MaxSubCallsExceeded
from .types import RLMEvent

logger = logging.getLogger(__name__)

_SUB_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the query based on the provided "
    "context. Be precise and thorough."
)


class _ChatClient(Protocol):
    """Minimal structural type for an OpenAI-compatible client."""

    class chat:  # noqa: N801
        class completions:  # noqa: N801
            @staticmethod
            def create(**kwargs: Any) -> Any: ...


class SubCallManager:
    """Tracks and executes sub-LLM calls.

    An instance of this class creates the ``llm_query`` function that is
    injected into the :class:`~rlm.repl.REPLEnvironment`.
    """

    def __init__(self, client: Any, config: RLMConfig, model: str) -> None:
        self._client = client
        self._config = config
        self._model = model
        self.call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._event_callback: Callable[[RLMEvent], None] | None = None
        self._current_iteration: int = 0

    def set_event_callback(
        self, callback: Callable[[RLMEvent], None] | None
    ) -> None:
        self._event_callback = callback

    def set_current_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    def make_query_fn(self) -> Callable[[str], str]:
        """Return a ``llm_query(prompt) -> str`` suitable for REPL injection."""

        def llm_query(prompt: str) -> str:
            if self.call_count >= self._config.max_sub_calls:
                raise MaxSubCallsExceeded(self.call_count, self._config.max_sub_calls)

            # Truncate input if necessary.
            if len(prompt) > self._config.sub_call_max_input_chars:
                prompt = prompt[: self._config.sub_call_max_input_chars]

            self.call_count += 1

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_start",
                        iteration=self._current_iteration,
                        preview=f"Sub-call #{self.call_count}: {prompt[:80]}...",
                        detail={"call_count": self.call_count, "prompt_len": len(prompt)},
                    )
                )

            if self._config.verbose:
                logger.info(
                    "sub-call #%d  prompt_len=%d  model=%s",
                    self.call_count,
                    len(prompt),
                    self._model,
                )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SUB_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._config.sub_temperature,
                max_tokens=self._config.sub_max_tokens,
            )

            text: str = response.choices[0].message.content or ""

            # Track token usage when available.
            usage = getattr(response, "usage", None)
            if usage:
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_end",
                        iteration=self._current_iteration,
                        preview=f"Sub-call #{self.call_count} done: {text[:80]}...",
                        detail={
                            "call_count": self.call_count,
                            "response_len": len(text),
                        },
                    )
                )

            return text

        return llm_query

    def reset(self) -> None:
        """Reset counters for a new generation."""
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
