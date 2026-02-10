"""Core RLM loop — Algorithm 1 from the paper."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .config import RLMConfig
from .exceptions import MaxIterationsExceeded
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
    """

    def __init__(
        self,
        client: Any,
        config: RLMConfig,
        root_model: str,
        sub_model: str,
    ) -> None:
        self._client = client
        self._config = config
        self._root_model = root_model
        self._sub_model = sub_model

    def run(
        self,
        query: str,
        context: str | list[str],
        on_event: Callable[[RLMEvent], None] | None = None,
    ) -> RLMResponse:
        """Execute the full RLM loop and return an :class:`RLMResponse`."""

        config = self._config

        # -- 1. Sub-call manager + REPL initialization -----------------------
        sub_mgr = SubCallManager(self._client, config, self._sub_model)
        sub_mgr.set_event_callback(on_event)
        llm_query_fn = sub_mgr.make_query_fn()

        repl = REPLEnvironment(
            context=context,
            llm_query_fn=llm_query_fn,
            timeout=config.sandbox_timeout,
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
            response = self._client.chat.completions.create(
                model=self._root_model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.root_max_tokens,
            )

            assistant_text: str = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            if usage:
                root_input_tokens += getattr(usage, "prompt_tokens", 0)
                root_output_tokens += getattr(usage, "completion_tokens", 0)

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

            # -- 3c. Check for FINAL directive *before* executing code -------
            if parsed.is_done:
                answer = self._resolve_final(parsed, repl, iteration, history)
                if answer is not None:
                    return self._build_response(
                        answer=answer,
                        iterations=iteration,
                        sub_mgr=sub_mgr,
                        root_input_tokens=root_input_tokens,
                        root_output_tokens=root_output_tokens,
                        history=history,
                        repl=repl,
                        on_event=on_event,
                    )

            # -- 3d. Execute code blocks in REPL -----------------------------
            combined_stdout = ""
            for block in parsed.code_blocks:
                stdout, _had_error = repl.execute(block)
                combined_stdout += stdout

            if on_event and parsed.code_blocks:
                on_event(
                    RLMEvent(
                        type="code_executed",
                        iteration=iteration,
                        preview=combined_stdout[:120],
                        detail={"stdout_len": len(combined_stdout)},
                    )
                )

            # Check if code set ``Final`` in the REPL namespace.
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
                )

            # -- 3e. Truncated metadata of stdout ----------------------------
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

            # -- 3f. Append to message history for next turn -----------------
            messages.append({"role": "assistant", "content": assistant_text})

            repl_label = "[REPL Output] " if combined_stdout else "[REPL] No output."
            messages.append(
                {"role": "user", "content": repl_label + truncated_stdout}
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
        )

    # -- Helpers -------------------------------------------------------------

    def _resolve_final(
        self,
        parsed: Any,
        repl: REPLEnvironment,
        iteration: int,
        history: list[HistoryEntry],
    ) -> str | None:
        """Try to resolve a FINAL or FINAL_VAR directive to a string answer."""
        if parsed.final_answer is not None:
            return parsed.final_answer

        if parsed.final_var is not None:
            if repl.has_variable(parsed.final_var):
                return str(repl.get_variable(parsed.final_var))
            # Variable doesn't exist yet — let the model keep going so it can
            # fix the reference.
            return None

        return None

    def _force_final(
        self,
        messages: list[dict[str, str]],
        repl: REPLEnvironment,
        sub_mgr: SubCallManager,
        root_input_tokens: int,
        root_output_tokens: int,
        history: list[HistoryEntry],
        on_event: Callable[[RLMEvent], None] | None,
    ) -> RLMResponse:
        """Send a nudge message and attempt to extract a final answer."""
        config = self._config

        messages.append({"role": "user", "content": build_nudge_prompt()})

        response = self._client.chat.completions.create(
            model=self._root_model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.root_max_tokens,
        )

        text: str = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage:
            root_input_tokens += getattr(usage, "prompt_tokens", 0)
            root_output_tokens += getattr(usage, "completion_tokens", 0)

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
    ) -> RLMResponse:
        if on_event:
            on_event(
                RLMEvent(
                    type="final_answer",
                    iteration=iterations,
                    preview=answer[:120],
                )
            )

        return RLMResponse(
            answer=answer,
            iterations=iterations,
            sub_calls=sub_mgr.call_count,
            total_input_tokens=root_input_tokens + sub_mgr.total_input_tokens,
            total_output_tokens=root_output_tokens + sub_mgr.total_output_tokens,
            history=history,
            repl_variables=repl.variable_summaries,
        )
