"""Provider abstraction — ``LLMClient`` protocol and ``OpenAIAdapter``.

Any object that implements :class:`LLMClient` can be used as the client for
:class:`~rlm.wrapper.RLMWrapper`.  The :class:`OpenAIAdapter` wraps a standard
``openai.OpenAI`` or ``openai.AsyncOpenAI`` client to satisfy this interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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


class ContentStream:
    """Async stream of content tokens with post-consumption metadata.

    Iterate to receive content deltas.  After the stream is exhausted,
    access :attr:`result` for the full text and token counts.

    Usage::

        stream = adapter.astream(model=..., messages=..., ...)
        async for delta in stream:
            print(delta, end="", flush=True)
        full_result = stream.result  # CompletionResult
    """

    def __init__(self, aiter: AsyncIterator[str]) -> None:
        self._aiter = aiter
        self._content_parts: list[str] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._done: bool = False

    def __aiter__(self) -> ContentStream:
        return self

    async def __anext__(self) -> str:
        try:
            delta = await self._aiter.__anext__()
            self._content_parts.append(delta)
            return delta
        except StopAsyncIteration:
            self._done = True
            raise

    def _set_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts (called internally after stream ends)."""
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    @property
    def result(self) -> CompletionResult:
        """Full completion result — available only after iteration completes."""
        if not self._done:
            raise RuntimeError("Stream not yet consumed — iterate first")
        return CompletionResult(
            content="".join(self._content_parts),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


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
        reasoning_effort: str | None = None,
    ) -> CompletionResult:
        """Run a chat completion synchronously."""
        ...

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
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
        reasoning_effort: str | None = None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = self._client.chat.completions.create(**kwargs)
        return _extract_result(response)

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = await self._client.chat.completions.create(**kwargs)
        return _extract_result(response)

    def astream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> ContentStream:
        """Return a :class:`ContentStream` that yields content tokens.

        After iterating, access ``stream.result`` for the full text and
        token counts.
        """
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        async def _gen() -> AsyncIterator[str]:
            response = await self._client.chat.completions.create(**kwargs)
            input_tokens = 0
            output_tokens = 0
            async for chunk in response:
                # Extract content delta
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        yield delta.content
                # Extract usage from final chunk
                usage = getattr(chunk, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_tokens", 0)
                    output_tokens = getattr(usage, "completion_tokens", 0)
            stream._set_usage(input_tokens, output_tokens)

        stream = ContentStream(_gen())
        return stream


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
