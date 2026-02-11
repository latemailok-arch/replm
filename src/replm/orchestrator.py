"""Core RLM loop — Algorithm 1 from the paper."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .budget import SharedBudget
from .cache import SubCallCache
from .client import wrap_if_needed
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
from .sub_caller import SubCallManager
from .tracing import span
from .types import HistoryEntry, RLMEvent, RLMResponse

logger = logging.getLogger(__name__)


class Orchestrator:
    """Implements the root REPL loop (Algorithm 1).

    Parameters
    ----------
    client:
        An OpenAI-compatible client instance.
    config:
        :class:`~rlm.config.RLMConfig` with all tuning knobs.
    root_model:
        Model identifier for the root LLM calls.
    sub_model:
        Model identifier for sub-LLM calls (falls back to *root_model*).
    budget:
        Shared budget for tracking calls/tokens across recursion depths.
        If ``None``, a fresh budget is created from *config* defaults.
    depth:
        Current recursion depth (0 = root level).
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

    def run(
        self,
        query: str,
        context: str | list[str],
        on_event: Callable[[RLMEvent], None] | None = None,
    ) -> RLMResponse:
        """Execute the full RLM loop and return an :class:`RLMResponse`."""
        with span(
            "rlm.generate",
            {
                "rlm.query_len": len(query),
                "rlm.model": self._root_model,
            },
        ) as root_span:
            return self._run_inner(query, context, on_event, root_span)

    def _run_inner(
        self,
        query: str,
        context: str | list[str],
        on_event: Callable[[RLMEvent], None] | None,
        root_span: Any,
    ) -> RLMResponse:
        start_time = time.monotonic()

        config = self._config

        if config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(name)s %(levelname)s: %(message)s",
            )
            logger.setLevel(logging.INFO)

        # -- 1. Sub-call manager + REPL initialization -----------------------
        cache = SubCallCache() if config.cache_sub_calls else None
        sub_mgr = SubCallManager(
            self._client,
            config,
            self._sub_model,
            budget=self._budget,
            depth=self._depth,
            cache=cache,
        )
        sub_mgr.set_event_callback(on_event)

        llm_query_fn = sub_mgr.make_query_fn() if config.enable_sub_calls else None

        repl = REPLEnvironment(
            context=context,
            llm_query_fn=llm_query_fn,
            timeout=config.sandbox_timeout,
            sandbox_mode=config.sandbox_mode,
        )

        # -- 2. System prompt with metadata (NOT the context!) ---------------
        system_prompt = build_root_system_prompt(
            context_type=context_type_label(context),
            context_total_length=context_total_length(context),
            context_lengths=context_chunk_lengths(context),
            config=config,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        history: list[HistoryEntry] = []
        root_input_tokens = 0
        root_output_tokens = 0
        has_executed_code = False  # Track whether any REPL code has run.

        # -- 3. Root loop ----------------------------------------------------
        for iteration in range(1, config.max_iterations + 1):
            sub_mgr.set_current_iteration(iteration)

            if on_event:
                on_event(
                    RLMEvent(
                        type="iteration_start",
                        iteration=iteration,
                        preview=f"Starting iteration {iteration}/{config.max_iterations}",
                    )
                )

            # -- 3a. Call root model -----------------------------------------
            result = self._client.complete(
                model=self._root_model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.root_max_tokens,
                reasoning_effort=config.reasoning_effort,
            )

            assistant_text: str = result.content
            root_input_tokens += result.input_tokens
            root_output_tokens += result.output_tokens

            if config.verbose:
                logger.info("iter %d  root output (%d chars)", iteration, len(assistant_text))

            if on_event:
                on_event(
                    RLMEvent(
                        type="code_generated",
                        iteration=iteration,
                        preview=assistant_text[:120],
                        detail={"full_text": assistant_text},
                    )
                )

            # -- 3b. Parse response ------------------------------------------
            parsed = parse_response(assistant_text)

            # -- 3c. Check for FINAL directive (direct answer, no code needed)
            if parsed.final_answer is not None:
                if has_executed_code or self._depth > 0:
                    return self._build_response(
                        answer=parsed.final_answer,
                        iterations=iteration,
                        sub_mgr=sub_mgr,
                        root_input_tokens=root_input_tokens,
                        root_output_tokens=root_output_tokens,
                        history=history,
                        repl=repl,
                        on_event=on_event,
                        start_time=start_time,
                        cache=cache,
                        root_span=root_span,
                    )
                # Safeguard: reject FINAL before any code was executed.
                # The model must examine the context via the REPL first.
                parsed.final_answer = None
                if config.verbose:
                    logger.info(
                        "iter %d  rejected premature FINAL (no code executed yet)",
                        iteration,
                    )

            # -- 3d. Execute code blocks in REPL -----------------------------
            combined_stdout = ""
            for block in parsed.code_blocks:
                stdout, _had_error = repl.execute(block)
                combined_stdout += stdout

            if parsed.code_blocks:
                has_executed_code = True

            if on_event and parsed.code_blocks:
                on_event(
                    RLMEvent(
                        type="code_executed",
                        iteration=iteration,
                        preview=combined_stdout[:120],
                        detail={"stdout_len": len(combined_stdout)},
                    )
                )

            # -- 3e. Check for FINAL_VAR or ``Final`` variable after code ran
            if parsed.final_var is not None:
                if repl.has_variable(parsed.final_var):
                    return self._build_response(
                        answer=str(repl.get_variable(parsed.final_var)),
                        iterations=iteration,
                        sub_mgr=sub_mgr,
                        root_input_tokens=root_input_tokens,
                        root_output_tokens=root_output_tokens,
                        history=history,
                        repl=repl,
                        on_event=on_event,
                        start_time=start_time,
                        cache=cache,
                    )
                else:
                    # Tell the model the variable doesn't exist so it can fix it.
                    available = ", ".join(repl.variable_names) or "(none)"
                    combined_stdout += (
                        f"\n[Error] FINAL_VAR referenced '{parsed.final_var}' "
                        f"but no such variable exists. "
                        f"Available variables: [{available}]"
                    )

            if repl.has_variable("Final"):
                answer = str(repl.get_variable("Final"))
                return self._build_response(
                    answer=answer,
                    iterations=iteration,
                    sub_mgr=sub_mgr,
                    root_input_tokens=root_input_tokens,
                    root_output_tokens=root_output_tokens,
                    history=history,
                    repl=repl,
                    on_event=on_event,
                    start_time=start_time,
                    cache=cache,
                    root_span=root_span,
                )

            # -- 3f. Truncated metadata of stdout ----------------------------
            truncated_stdout = make_metadata(combined_stdout, config.metadata_prefix_chars)

            # Record history.
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

            # -- 3g. Append to message history for next turn -----------------
            messages.append({"role": "assistant", "content": assistant_text})

            if combined_stdout:
                messages.append(
                    {"role": "user", "content": "[REPL Output] " + truncated_stdout}
                )
            elif not parsed.code_blocks:
                # Model didn't write any code — nudge it to use the REPL.
                nudge = (
                    "[System] You must examine the context before providing a final "
                    "answer. The `context` variable is loaded in the REPL. Write "
                    "Python code in ```repl blocks to access it. Example:\n\n"
                    "```repl\nprint(context[:500])\n```"
                )
                messages.append({"role": "user", "content": nudge})
            else:
                messages.append(
                    {"role": "user", "content": "[REPL] No output."}
                )

        # -- 4. Max iterations: nudge for a final answer ---------------------
        return self._force_final(
            messages=messages,
            repl=repl,
            sub_mgr=sub_mgr,
            root_input_tokens=root_input_tokens,
            root_output_tokens=root_output_tokens,
            history=history,
            on_event=on_event,
            start_time=start_time,
            cache=cache,
            root_span=root_span,
        )

    # -- Helpers -------------------------------------------------------------

    def _force_final(
        self,
        messages: list[dict[str, str]],
        repl: REPLEnvironment,
        sub_mgr: SubCallManager,
        root_input_tokens: int,
        root_output_tokens: int,
        history: list[HistoryEntry],
        on_event: Callable[[RLMEvent], None] | None,
        start_time: float = 0.0,
        cache: SubCallCache | None = None,
        root_span: Any = None,
    ) -> RLMResponse:
        """Send a nudge message and attempt to extract a final answer."""
        config = self._config

        messages.append({"role": "user", "content": build_nudge_prompt()})

        result = self._client.complete(
            model=self._root_model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.root_max_tokens,
            reasoning_effort=config.reasoning_effort,
        )

        text: str = result.content
        root_input_tokens += result.input_tokens
        root_output_tokens += result.output_tokens

        parsed = parse_response(text)

        answer: str | None = None
        if parsed.final_answer is not None:
            answer = parsed.final_answer
        elif parsed.final_var is not None and repl.has_variable(parsed.final_var):
            answer = str(repl.get_variable(parsed.final_var))

        if answer is None:
            # Last resort: check for plausible answer variables.
            for name in ("final_answer", "answer", "result", "output"):
                if repl.has_variable(name):
                    answer = str(repl.get_variable(name))
                    break

        if answer is None:
            # Return the raw nudge response as the answer.
            answer = text

        return self._build_response(
            answer=answer,
            iterations=config.max_iterations,
            sub_mgr=sub_mgr,
            root_input_tokens=root_input_tokens,
            root_output_tokens=root_output_tokens,
            history=history,
            repl=repl,
            on_event=on_event,
            start_time=start_time,
            cache=cache,
            root_span=root_span,
        )

    def _build_response(
        self,
        answer: str,
        iterations: int,
        sub_mgr: SubCallManager,
        root_input_tokens: int,
        root_output_tokens: int,
        history: list[HistoryEntry],
        repl: REPLEnvironment,
        on_event: Callable[[RLMEvent], None] | None,
        start_time: float = 0.0,
        cache: SubCallCache | None = None,
        root_span: Any = None,
    ) -> RLMResponse:
        if on_event:
            on_event(
                RLMEvent(
                    type="final_answer",
                    iteration=iterations,
                    preview=answer[:120],
                )
            )

        elapsed = time.monotonic() - start_time if start_time else 0.0

        total_in = root_input_tokens + sub_mgr.total_input_tokens
        total_out = root_output_tokens + sub_mgr.total_output_tokens

        if root_span is not None:
            root_span.set_attribute("rlm.iterations", iterations)
            root_span.set_attribute("rlm.sub_calls", sub_mgr.call_count)
            root_span.set_attribute("rlm.total_input_tokens", total_in)
            root_span.set_attribute("rlm.total_output_tokens", total_out)
            root_span.set_attribute("rlm.elapsed_seconds", elapsed)

        return RLMResponse(
            answer=answer,
            iterations=iterations,
            sub_calls=sub_mgr.call_count,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            cache_hits=cache.stats.hits if cache else 0,
            cost_per_input_token=self._config.cost_per_input_token,
            cost_per_output_token=self._config.cost_per_output_token,
            history=history,
            repl_variables=repl.variable_summaries,
            elapsed_seconds=elapsed,
        )
