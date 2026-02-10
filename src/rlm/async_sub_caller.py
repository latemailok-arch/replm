"""Async sub-call manager — creates ``llm_query`` and ``llm_query_batch`` for the REPL.

The REPL executes code in a worker thread, so the sync ``llm_query`` /
``llm_query_batch`` callables bridge back to the event loop via
:func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .budget import SharedBudget
from .cache import SubCallCache
from .config import RLMConfig
from .types import RLMEvent

logger = logging.getLogger(__name__)

_SUB_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the query based on the provided "
    "context. Be precise and thorough."
)


class AsyncSubCallManager:
    """Async variant of :class:`~rlm.sub_caller.SubCallManager`.

    Produces sync ``llm_query`` / ``llm_query_batch`` functions that
    internally schedule async work on the running event loop so they can be
    called from the REPL worker thread.

    Parameters
    ----------
    client:
        An async OpenAI-compatible client (e.g. ``openai.AsyncOpenAI``).
    config:
        RLM configuration.
    model:
        Model identifier for sub-calls.
    budget:
        Shared budget for tracking calls/tokens across recursion depths.
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
        cache: SubCallCache | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._model = model
        self._budget = budget or SharedBudget(max_sub_calls=config.max_sub_calls)
        self._depth = depth
        self._cache = cache
        self._event_callback: Callable[[RLMEvent], None] | None = None
        self._current_iteration: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Properties delegating to shared budget --------------------------------

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
        return self._budget

    # -- Configuration ---------------------------------------------------------

    def set_event_callback(self, callback: Callable[[RLMEvent], None] | None) -> None:
        self._event_callback = callback

    def set_current_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the event loop reference for thread→loop bridging."""
        self._loop = loop

    # -- Public API: create sync callables for REPL injection ------------------

    def make_query_fn(self) -> Callable[[str], str]:
        """Return a sync ``llm_query(prompt) -> str`` for REPL injection.

        Dispatches between recursive (inner AsyncOrchestrator) and plain
        (single async LLM call) based on depth vs max_recursion_depth.
        """
        if self._depth + 1 < self._config.max_recursion_depth:
            return self._make_recursive_query_fn()
        return self._make_plain_query_fn()

    def make_batch_fn(self) -> Callable[[list[str]], list[str]]:
        """Return ``llm_query_batch(prompts) -> list[str]`` for parallel sub-calls."""

        def llm_query_batch(prompts: list[str]) -> list[str]:
            loop = self._loop
            if loop is None:
                raise RuntimeError("Event loop not set — call set_loop() first")
            future = asyncio.run_coroutine_threadsafe(self._async_llm_query_batch(prompts), loop)
            return future.result()

        return llm_query_batch

    # -- Async implementations ------------------------------------------------

    async def _async_llm_query(self, prompt: str) -> str:
        """Single async sub-call (plain / leaf level)."""
        if len(prompt) > self._config.sub_call_max_input_chars:
            prompt = prompt[: self._config.sub_call_max_input_chars]

        # -- Check cache before consuming budget --------------------------------
        cache_key: str | None = None
        if self._cache is not None:
            cache_key = SubCallCache.key(
                self._model, prompt, self._config.sub_temperature
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                if self._event_callback:
                    self._event_callback(
                        RLMEvent(
                            type="sub_call_cache_hit",
                            iteration=self._current_iteration,
                            preview=f"Cache hit: {prompt[:80]}...",
                            detail={
                                "prompt_len": len(prompt),
                                "depth": self._depth,
                            },
                            depth=self._depth,
                        )
                    )
                return cached

        # -- Budget + API call --------------------------------------------------
        self._budget.increment_call()
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
                "async sub-call #%d  depth=%d  prompt_len=%d  model=%s",
                count,
                self._depth,
                len(prompt),
                self._model,
            )

        result = await self._client.acomplete(
            model=self._model,
            messages=[
                {"role": "system", "content": _SUB_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._config.sub_temperature,
            max_tokens=self._config.sub_max_tokens,
        )

        text: str = result.content
        self._budget.add_tokens(result.input_tokens, result.output_tokens)

        # -- Store in cache -----------------------------------------------------
        if cache_key is not None:
            self._cache.put(cache_key, text)

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

    async def _async_llm_query_batch(self, prompts: list[str]) -> list[str]:
        """Run multiple sub-calls in parallel via ``asyncio.gather``."""
        return list(await asyncio.gather(*(self._async_llm_query(p) for p in prompts)))

    async def _async_recursive_query(self, prompt: str) -> str:
        """Recursive sub-call via an inner AsyncOrchestrator."""
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
                "async recursive sub-call #%d  depth=%d->%d  prompt_len=%d",
                count,
                self._depth,
                self._depth + 1,
                len(prompt),
            )

        # Snapshot token state before the inner run so we can compute
        # how many tokens the inner orchestrator's root calls consumed
        # (as opposed to inner sub-call tokens already tracked by budget).
        tokens_before_in = self._budget.total_input_tokens
        tokens_before_out = self._budget.total_output_tokens

        # Local import to avoid circular dependency.
        from .async_orchestrator import AsyncOrchestrator

        inner_orch = AsyncOrchestrator(
            client=self._client,
            config=self._config,
            root_model=self._model,
            sub_model=self._model,
            budget=self._budget,
            depth=self._depth + 1,
        )

        resp = await inner_orch.run(
            query=prompt,
            context=prompt,
            on_event=self._event_callback,
        )

        # The inner orchestrator tracked its own root-model tokens locally
        # (not via the shared budget).  Add them so the caller's totals are
        # accurate.  Sub-call tokens were already accumulated through the
        # shared budget, so subtract them to avoid double-counting.
        inner_root_in = (
            resp.total_input_tokens - self._budget.total_input_tokens + tokens_before_in
        )
        inner_root_out = (
            resp.total_output_tokens - self._budget.total_output_tokens + tokens_before_out
        )
        if inner_root_in > 0 or inner_root_out > 0:
            self._budget.add_tokens(max(inner_root_in, 0), max(inner_root_out, 0))

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

    # -- Sync wrappers (called from REPL thread) -------------------------------

    def _make_plain_query_fn(self) -> Callable[[str], str]:
        def llm_query(prompt: str) -> str:
            loop = self._loop
            if loop is None:
                raise RuntimeError("Event loop not set — call set_loop() first")
            future = asyncio.run_coroutine_threadsafe(self._async_llm_query(prompt), loop)
            return future.result()

        return llm_query

    def _make_recursive_query_fn(self) -> Callable[[str], str]:
        def llm_query(prompt: str) -> str:
            loop = self._loop
            if loop is None:
                raise RuntimeError("Event loop not set — call set_loop() first")
            future = asyncio.run_coroutine_threadsafe(self._async_recursive_query(prompt), loop)
            return future.result()

        return llm_query
