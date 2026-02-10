"""Provider abstraction — ``LLMClient`` protocol and ``OpenAIAdapter``.

Any object that implements :class:`LLMClient` can be used as the client for
:class:`~rlm.wrapper.RLMWrapper`.  The :class:`OpenAIAdapter` wraps a standard
``openai.OpenAI`` or ``openai.AsyncOpenAI`` client to satisfy this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class CompletionResult:
    """The result of a single LLM completion call."""

    content: str
    """The text response."""

    input_tokens: int
    """Number of input (prompt) tokens consumed."""

    output_tokens: int
    """Number of output (completion) tokens produced."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol that any LLM provider must satisfy.

    Implement :meth:`complete` for synchronous usage and :meth:`acomplete`
    for async usage.  You only need to implement the one(s) you actually use.
    """

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Run a chat completion synchronously."""
        ...

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Run a chat completion asynchronously."""
        ...


def _extract_result(response: Any) -> CompletionResult:
    """Extract content and token counts from an OpenAI-style response."""
    content: str = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    return CompletionResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OpenAIAdapter:
    """Wraps an ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client.

    Parameters
    ----------
    client:
        Any object with a ``client.chat.completions.create(...)`` method
        (sync or async).
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_result(response)

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_result(response)


def wrap_if_needed(client: Any) -> Any:
    """Auto-wrap an OpenAI-style client in :class:`OpenAIAdapter` if needed.

    If *client* already has a ``complete`` method, it is returned as-is.
    If it has a ``chat.completions`` attribute (OpenAI SDK pattern), it is
    wrapped in :class:`OpenAIAdapter`.
    """
    if hasattr(client, "complete"):
        return client
    if hasattr(client, "chat"):
        return OpenAIAdapter(client)
    return client
