"""Public entry point — ``RLMWrapper``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import RLMConfig
from .orchestrator import Orchestrator
from .types import RLMEvent, RLMResponse


class RLMWrapper:
    """Wrap an OpenAI-compatible client as a Recursive Language Model.

    Parameters
    ----------
    client:
        An OpenAI-compatible client (e.g. ``openai.OpenAI(...)``).
    root_model:
        Model identifier for the root orchestrator.
    sub_model:
        Model identifier for sub-calls.  Defaults to *root_model*.
    config:
        Optional :class:`RLMConfig` with tuning knobs.

    Example
    -------
    >>> from openai import OpenAI
    >>> from rlm import RLMWrapper
    >>> wrapper = RLMWrapper(OpenAI(api_key="sk-..."), root_model="gpt-4.1")
    >>> resp = wrapper.generate("Summarize this.", long_text)
    >>> print(resp.answer)
    """

    def __init__(
        self,
        client: Any,
        root_model: str = "gpt-4.1",
        sub_model: str | None = None,
        config: RLMConfig | None = None,
    ) -> None:
        self._client = client
        self._root_model = root_model
        self._sub_model = sub_model or root_model
        self._config = config or RLMConfig()

    def generate(
        self,
        query: str,
        context: str | list[str],
        on_event: Callable[[RLMEvent], None] | None = None,
    ) -> RLMResponse:
        """Process *query* over a (potentially very long) *context* using the RLM loop.

        Parameters
        ----------
        query:
            The question or task.
        context:
            The long input.  Either a single string or a list of strings
            (e.g. a list of documents).  Stored as a REPL variable — never
            fed into the LLM context window.
        on_event:
            Optional callback for streaming observability events.

        Returns
        -------
        RLMResponse
            Contains the answer, iteration/sub-call counts, token usage, and
            the full execution trace.
        """
        orchestrator = Orchestrator(
            client=self._client,
            config=self._config,
            root_model=self._root_model,
            sub_model=self._sub_model,
        )
        return orchestrator.run(query=query, context=context, on_event=on_event)
