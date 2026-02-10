"""Streaming RLM loop — yields ``StreamChunk`` objects as the root model produces tokens.

The root loop still needs the full response before it can parse code blocks and
execute them, so streaming applies to the root model's text output only.  Code
execution results and other events are yielded as non-token chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .async_sub_caller import AsyncSubCallManager
from .budget import SharedBudget
from .cache import SubCallCache
from .client import CompletionResult, ContentStream, wrap_if_needed
from .config import RLMConfig
from .metadata import (
    context_chunk_lengths,
    context_total_length,
    context_type_label,
    make_metadata,
)
from .parser import parse_response
from .prompt import build_nudge_prompt, build_root_system_prompt
from .repl import REPLEnvironment
from .types import HistoryEntry, RLMResponse

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A single chunk yielded by :meth:`~rlm.wrapper.RLMWrapper.astream_generate`.

    Attributes
    ----------
    type:
        One of ``"token"``, ``"iteration_start"``, ``"code_executed"``,
        ``"final_answer"``.
    content:
        The text content of this chunk.  For ``"token"`` chunks, this is
        a partial content delta.  For ``"final_answer"``, it is the full
        answer string.
    iteration:
        The current root-loop iteration number.
    detail:
        Extra metadata (type-specific).
    """

    type: str
    content: str
    iteration: int
    detail: dict[str, Any] = field(default_factory=dict)


async def _make_content_stream_from_result(
    result: CompletionResult,
) -> AsyncIterator[str]:
    """Wrap a non-streaming result as a single-item async iterator."""
    yield result.content


class StreamOrchestrator:
    """Async generator variant of :class:`~rlm.async_orchestrator.AsyncOrchestrator`.

    Instead of returning a single :class:`RLMResponse`, this orchestrator
    *yields* :class:`StreamChunk` objects.  The final chunk (type
    ``"final_answer"``) includes the complete :class:`RLMResponse` in its
    ``detail["response"]``.

    Parameters
    ----------
    client:
        An async LLM client.  If it has an ``astream()`` method, tokens
        are streamed in real-time.  Otherwise falls back to ``acomplete()``.
    config:
        RLM configuration.
    root_model / sub_model:
        Model identifiers.
    budget / depth:
        For recursive orchestrators (usually left at defaults).
    """

    def __init__(
        self,
        client: Any,
        config: RLMConfig,
        root_model: str,
        sub_model: str,
        budget: SharedBudget | None = None,
        depth: int = 0,
    ) -> None:
        self._client = wrap_if_needed(client)
        self._config = config
        self._root_model = root_model
        self._sub_model = sub_model
        self._budget = budget
        self._depth = depth

    def _get_root_call_kwargs(self) -> dict[str, Any]:
        """Common kwargs for root model calls."""
        kwargs: dict[str, Any] = {
            "model": self._root_model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.root_max_tokens,
        }
        if self._config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._config.reasoning_effort
        return kwargs

    async def _root_call_stream(
        self, messages: list[dict[str, str]]
    ) -> tuple[ContentStream, bool]:
        """Return a ContentStream for the root call.

        Returns ``(stream, is_real_stream)`` — *is_real_stream* is ``True``
        when the client supports native streaming, ``False`` when we fall
        back to wrapping ``acomplete()`` as a single-chunk stream.
        """
        kwargs = self._get_root_call_kwargs()
        kwargs["messages"] = messages

        if hasattr(self._client, "astream"):
            return self._client.astream(**kwargs), True

        # Fallback: non-streaming → wrap as single-chunk ContentStream
        result = await self._client.acomplete(**kwargs)
        stream = ContentStream(_make_content_stream_from_result(result))
        stream._set_usage(result.input_tokens, result.output_tokens)
        return stream, False

    async def run(
        self,
        query: str,
        context: str | list[str],
    ) -> AsyncIterator[StreamChunk]:
        """Execute the RLM loop, yielding stream chunks.

        The final chunk has ``type="final_answer"`` and includes the full
        :class:`RLMResponse` in ``detail["response"]``.
        """
        start_time = time.monotonic()
        config = self._config

        # -- 1. Sub-call manager + REPL ----------------------------------------
        cache = SubCallCache() if config.cache_sub_calls else None
        sub_mgr = AsyncSubCallManager(
            self._client,
            config,
            self._sub_model,
            budget=self._budget,
            depth=self._depth,
            cache=cache,
        )
        sub_mgr.set_loop(asyncio.get_running_loop())

        llm_query_fn = sub_mgr.make_query_fn() if config.enable_sub_calls else None
        llm_query_batch_fn = sub_mgr.make_batch_fn() if config.enable_sub_calls else None

        repl = REPLEnvironment(
            context=context,
            llm_query_fn=llm_query_fn,
            llm_query_batch_fn=llm_query_batch_fn,
            timeout=config.sandbox_timeout,
            sandbox_mode=config.sandbox_mode,
        )

        # -- 2. System prompt --------------------------------------------------
        system_prompt = build_root_system_prompt(
            context_type=context_type_label(context),
            context_total_length=context_total_length(context),
            context_lengths=context_chunk_lengths(context),
            config=config,
            include_batch_fn=config.enable_sub_calls,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        history: list[HistoryEntry] = []
        root_input_tokens = 0
        root_output_tokens = 0

        # -- 3. Root loop ------------------------------------------------------
        for iteration in range(1, config.max_iterations + 1):
            sub_mgr.set_current_iteration(iteration)

            yield StreamChunk(
                type="iteration_start",
                content=f"Starting iteration {iteration}/{config.max_iterations}",
                iteration=iteration,
            )

            # -- 3a. Stream root model tokens -----------------------------------
            stream, _is_real = await self._root_call_stream(messages)
            async for delta in stream:
                yield StreamChunk(type="token", content=delta, iteration=iteration)

            result = stream.result
            assistant_text = result.content
            root_input_tokens += result.input_tokens
            root_output_tokens += result.output_tokens

            # -- 3b. Parse response --------------------------------------------
            parsed = parse_response(assistant_text)

            # -- 3c. Check for FINAL directive ---------------------------------
            if parsed.final_answer is not None:
                resp = self._build_response(
                    answer=parsed.final_answer,
                    iterations=iteration,
                    sub_mgr=sub_mgr,
                    root_input_tokens=root_input_tokens,
                    root_output_tokens=root_output_tokens,
                    history=history,
                    repl=repl,
                    start_time=start_time,
                    cache=cache,
                )
                yield StreamChunk(
                    type="final_answer",
                    content=resp.answer,
                    iteration=iteration,
                    detail={"response": resp},
                )
                return

            # -- 3d. Execute code blocks in REPL --------------------------------
            loop = asyncio.get_running_loop()
            combined_stdout = ""
            for block in parsed.code_blocks:
                stdout, _had_error = await loop.run_in_executor(None, repl.execute, block)
                combined_stdout += stdout

            if parsed.code_blocks:
                yield StreamChunk(
                    type="code_executed",
                    content=combined_stdout[:500],
                    iteration=iteration,
                    detail={"stdout_len": len(combined_stdout)},
                )

            # -- 3e. Check for FINAL_VAR or Final variable ---------------------
            if parsed.final_var is not None:
                if repl.has_variable(parsed.final_var):
                    resp = self._build_response(
                        answer=str(repl.get_variable(parsed.final_var)),
                        iterations=iteration,
                        sub_mgr=sub_mgr,
                        root_input_tokens=root_input_tokens,
                        root_output_tokens=root_output_tokens,
                        history=history,
                        repl=repl,
                        start_time=start_time,
                        cache=cache,
                    )
                    yield StreamChunk(
                        type="final_answer",
                        content=resp.answer,
                        iteration=iteration,
                        detail={"response": resp},
                    )
                    return
                else:
                    available = ", ".join(repl.variable_names) or "(none)"
                    combined_stdout += (
                        f"\n[Error] FINAL_VAR referenced '{parsed.final_var}' "
                        f"but no such variable exists. "
                        f"Available variables: [{available}]"
                    )

            if repl.has_variable("Final"):
                answer = str(repl.get_variable("Final"))
                resp = self._build_response(
                    answer=answer,
                    iterations=iteration,
                    sub_mgr=sub_mgr,
                    root_input_tokens=root_input_tokens,
                    root_output_tokens=root_output_tokens,
                    history=history,
                    repl=repl,
                    start_time=start_time,
                    cache=cache,
                )
                yield StreamChunk(
                    type="final_answer",
                    content=resp.answer,
                    iteration=iteration,
                    detail={"response": resp},
                )
                return

            # -- 3f. Truncated metadata ----------------------------------------
            truncated_stdout = make_metadata(combined_stdout, config.metadata_prefix_chars)

            history.append(
                HistoryEntry(
                    role="root",
                    content=assistant_text,
                    truncated_content=assistant_text,
                    iteration=iteration,
                )
            )
            history.append(
                HistoryEntry(
                    role="repl_output",
                    content=combined_stdout,
                    truncated_content=truncated_stdout,
                    iteration=iteration,
                )
            )

            # -- 3g. Append to message history ---------------------------------
            messages.append({"role": "assistant", "content": assistant_text})
            repl_label = "[REPL Output] " if combined_stdout else "[REPL] No output."
            messages.append({"role": "user", "content": repl_label + truncated_stdout})

        # -- 4. Max iterations: nudge ------------------------------------------
        messages.append({"role": "user", "content": build_nudge_prompt()})

        stream, _is_real = await self._root_call_stream(messages)
        async for delta in stream:
            yield StreamChunk(
                type="token",
                content=delta,
                iteration=config.max_iterations,
            )

        result = stream.result
        text = result.content
        root_input_tokens += result.input_tokens
        root_output_tokens += result.output_tokens

        parsed = parse_response(text)

        answer: str | None = None
        if parsed.final_answer is not None:
            answer = parsed.final_answer
        elif parsed.final_var is not None and repl.has_variable(parsed.final_var):
            answer = str(repl.get_variable(parsed.final_var))

        if answer is None:
            for name in ("final_answer", "answer", "result", "output"):
                if repl.has_variable(name):
                    answer = str(repl.get_variable(name))
                    break

        if answer is None:
            answer = text

        resp = self._build_response(
            answer=answer,
            iterations=config.max_iterations,
            sub_mgr=sub_mgr,
            root_input_tokens=root_input_tokens,
            root_output_tokens=root_output_tokens,
            history=history,
            repl=repl,
            start_time=start_time,
            cache=cache,
        )
        yield StreamChunk(
            type="final_answer",
            content=resp.answer,
            iteration=config.max_iterations,
            detail={"response": resp},
        )

    # -- Internal helpers ------------------------------------------------------

    def _build_response(
        self,
        answer: str,
        iterations: int,
        sub_mgr: AsyncSubCallManager,
        root_input_tokens: int,
        root_output_tokens: int,
        history: list[HistoryEntry],
        repl: REPLEnvironment,
        start_time: float,
        cache: SubCallCache | None = None,
    ) -> RLMResponse:
        elapsed = time.monotonic() - start_time if start_time else 0.0
        return RLMResponse(
            answer=answer,
            iterations=iterations,
            sub_calls=sub_mgr.call_count,
            total_input_tokens=root_input_tokens + sub_mgr.total_input_tokens,
            total_output_tokens=root_output_tokens + sub_mgr.total_output_tokens,
            cache_hits=cache.stats.hits if cache else 0,
            cost_per_input_token=self._config.cost_per_input_token,
            cost_per_output_token=self._config.cost_per_output_token,
            history=history,
            repl_variables=repl.variable_summaries,
            elapsed_seconds=elapsed,
        )
