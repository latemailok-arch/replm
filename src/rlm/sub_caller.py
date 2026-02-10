"""Sub-call manager — creates the ``llm_query`` function injected into the REPL."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from .budget import SharedBudget
from .config import RLMConfig
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

    Parameters
    ----------
    client:
        An OpenAI-compatible client.
    config:
        RLM configuration.
    model:
        Model identifier for sub-calls.
    budget:
        Shared budget for tracking calls/tokens across recursion depths.
        If ``None``, a new budget is created from config defaults.
    depth:
        Current recursion depth (0 = root level).
    """

    def __init__(
        self,
        client: Any,
        config: RLMConfig,
        model: str,
        budget: SharedBudget | None = None,
        depth: int = 0,
    ) -> None:
        self._client = client
        self._config = config
        self._model = model
        self._budget = budget or SharedBudget(max_sub_calls=config.max_sub_calls)
        self._depth = depth
        self._event_callback: Callable[[RLMEvent], None] | None = None
        self._current_iteration: int = 0

    @property
    def call_count(self) -> int:
        return self._budget.call_count

    @property
    def total_input_tokens(self) -> int:
        return self._budget.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._budget.total_output_tokens

    @property
    def budget(self) -> SharedBudget:
        """The shared budget instance (for passing to inner orchestrators)."""
        return self._budget

    def set_event_callback(self, callback: Callable[[RLMEvent], None] | None) -> None:
        self._event_callback = callback

    def set_current_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    def make_query_fn(self) -> Callable[[str], str]:
        """Return a ``llm_query(prompt) -> str`` suitable for REPL injection.

        When ``depth + 1 < max_recursion_depth``, sub-calls run through a full
        inner :class:`~rlm.orchestrator.Orchestrator` (recursive RLM).
        Otherwise they are plain LLM calls.

        With the default ``max_recursion_depth=1``, sub-calls are always
        plain (no recursion).  Set ``max_recursion_depth=2`` to get one level
        of nested orchestrators, etc.
        """
        if self._depth + 1 < self._config.max_recursion_depth:
            return self._make_recursive_query_fn()
        return self._make_plain_query_fn()

    # -- Plain sub-call (leaf level) -----------------------------------------

    def _make_plain_query_fn(self) -> Callable[[str], str]:
        def llm_query(prompt: str) -> str:
            self._budget.increment_call()

            if len(prompt) > self._config.sub_call_max_input_chars:
                prompt = prompt[: self._config.sub_call_max_input_chars]

            count = self._budget.call_count

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_start",
                        iteration=self._current_iteration,
                        preview=f"Sub-call #{count}: {prompt[:80]}...",
                        detail={
                            "call_count": count,
                            "prompt_len": len(prompt),
                            "depth": self._depth,
                        },
                        depth=self._depth,
                    )
                )

            if self._config.verbose:
                logger.info(
                    "sub-call #%d  depth=%d  prompt_len=%d  model=%s",
                    count,
                    self._depth,
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

            usage = getattr(response, "usage", None)
            if usage:
                self._budget.add_tokens(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                )

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_end",
                        iteration=self._current_iteration,
                        preview=f"Sub-call #{count} done: {text[:80]}...",
                        detail={
                            "call_count": count,
                            "response_len": len(text),
                            "depth": self._depth,
                        },
                        depth=self._depth,
                    )
                )

            return text

        return llm_query

    # -- Recursive sub-call (inner RLM) -------------------------------------

    def _make_recursive_query_fn(self) -> Callable[[str], str]:
        def llm_query(prompt: str) -> str:
            self._budget.increment_call()

            if len(prompt) > self._config.sub_call_max_input_chars:
                prompt = prompt[: self._config.sub_call_max_input_chars]

            count = self._budget.call_count

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_start",
                        iteration=self._current_iteration,
                        preview=f"Recursive sub-call #{count} (depth {self._depth + 1})",
                        detail={
                            "call_count": count,
                            "prompt_len": len(prompt),
                            "depth": self._depth,
                            "recursive": True,
                        },
                        depth=self._depth,
                    )
                )

            if self._config.verbose:
                logger.info(
                    "recursive sub-call #%d  depth=%d->%d  prompt_len=%d",
                    count,
                    self._depth,
                    self._depth + 1,
                    len(prompt),
                )

            # Local import to avoid circular dependency.
            from .orchestrator import Orchestrator

            inner_orch = Orchestrator(
                client=self._client,
                config=self._config,
                root_model=self._model,
                sub_model=self._model,
                budget=self._budget,
                depth=self._depth + 1,
            )

            resp = inner_orch.run(
                query=prompt,
                context=prompt,
                on_event=self._event_callback,
            )

            if self._event_callback:
                self._event_callback(
                    RLMEvent(
                        type="sub_call_end",
                        iteration=self._current_iteration,
                        preview=f"Recursive sub-call #{count} done (depth {self._depth + 1})",
                        detail={
                            "call_count": count,
                            "response_len": len(resp.answer),
                            "depth": self._depth,
                            "recursive": True,
                            "inner_iterations": resp.iterations,
                        },
                        depth=self._depth,
                    )
                )

            return resp.answer

        return llm_query
