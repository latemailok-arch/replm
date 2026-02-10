"""Tests for the provider abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.client import CompletionResult, OpenAIAdapter, wrap_if_needed

# ---------------------------------------------------------------------------
# Mock OpenAI SDK objects
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    prompt_tokens: int = 42
    completion_tokens: int = 17


@dataclass
class _Message:
    content: str = "hello from mock"


@dataclass
class _Choice:
    message: _Message


@dataclass
class _OpenAIResponse:
    choices: list[_Choice]
    usage: _Usage | None = None


class _MockCompletions:
    def __init__(self, response: _OpenAIResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _OpenAIResponse:
        self.calls.append(kwargs)
        return self._response

    async def create_async(self, **kwargs: Any) -> _OpenAIResponse:
        self.calls.append(kwargs)
        return self._response


class _MockChat:
    def __init__(self, response: _OpenAIResponse) -> None:
        self.completions = _MockCompletions(response)


class _MockOpenAIClient:
    """Mimics the OpenAI SDK's `client.chat.completions.create()` pattern."""

    def __init__(self, response: _OpenAIResponse | None = None) -> None:
        resp = response or _OpenAIResponse(
            choices=[_Choice(message=_Message())],
            usage=_Usage(),
        )
        self.chat = _MockChat(resp)


class _MockAsyncCompletions:
    def __init__(self, response: _OpenAIResponse) -> None:
        self._response = response

    async def create(self, **kwargs: Any) -> _OpenAIResponse:
        return self._response


class _MockAsyncChat:
    def __init__(self, response: _OpenAIResponse) -> None:
        self.completions = _MockAsyncCompletions(response)


class _MockAsyncOpenAIClient:
    def __init__(self, response: _OpenAIResponse | None = None) -> None:
        resp = response or _OpenAIResponse(
            choices=[_Choice(message=_Message())],
            usage=_Usage(),
        )
        self.chat = _MockAsyncChat(resp)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompletionResult:
    def test_fields(self):
        r = CompletionResult(content="hi", input_tokens=10, output_tokens=5)
        assert r.content == "hi"
        assert r.input_tokens == 10
        assert r.output_tokens == 5


class TestOpenAIAdapter:
    def test_complete_extracts_content(self):
        client = _MockOpenAIClient()
        adapter = OpenAIAdapter(client)
        result = adapter.complete(model="m", messages=[], temperature=0.5, max_tokens=100)
        assert result.content == "hello from mock"

    def test_complete_extracts_tokens(self):
        client = _MockOpenAIClient()
        adapter = OpenAIAdapter(client)
        result = adapter.complete(model="m", messages=[], temperature=0.5, max_tokens=100)
        assert result.input_tokens == 42
        assert result.output_tokens == 17

    def test_complete_missing_usage(self):
        resp = _OpenAIResponse(choices=[_Choice(message=_Message())], usage=None)
        client = _MockOpenAIClient(resp)
        adapter = OpenAIAdapter(client)
        result = adapter.complete(model="m", messages=[], temperature=0.5, max_tokens=100)
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_complete_empty_content(self):
        resp = _OpenAIResponse(
            choices=[_Choice(message=_Message(content=""))],
            usage=_Usage(),
        )
        client = _MockOpenAIClient(resp)
        adapter = OpenAIAdapter(client)
        result = adapter.complete(model="m", messages=[], temperature=0.5, max_tokens=100)
        assert result.content == ""

    def test_complete_passes_params(self):
        client = _MockOpenAIClient()
        adapter = OpenAIAdapter(client)
        messages = [{"role": "user", "content": "hi"}]
        adapter.complete(model="gpt-5", messages=messages, temperature=0.7, max_tokens=500)
        call = client.chat.completions.calls[0]
        assert call["model"] == "gpt-5"
        assert call["messages"] == messages
        assert call["temperature"] == 0.7
        assert call["max_tokens"] == 500


class TestOpenAIAdapterAsync:
    @pytest.mark.asyncio
    async def test_acomplete(self):
        client = _MockAsyncOpenAIClient()
        adapter = OpenAIAdapter(client)
        result = await adapter.acomplete(model="m", messages=[], temperature=0.5, max_tokens=100)
        assert result.content == "hello from mock"
        assert result.input_tokens == 42
        assert result.output_tokens == 17


class TestWrapIfNeeded:
    def test_wraps_openai_client(self):
        client = _MockOpenAIClient()
        wrapped = wrap_if_needed(client)
        assert isinstance(wrapped, OpenAIAdapter)

    def test_passthrough_if_has_complete(self):
        class CustomClient:
            def complete(self, **kwargs: Any) -> CompletionResult:
                return CompletionResult(content="custom", input_tokens=0, output_tokens=0)

        client = CustomClient()
        wrapped = wrap_if_needed(client)
        assert wrapped is client

    def test_passthrough_unknown(self):
        """Objects without chat or complete are returned as-is."""
        obj = object()
        assert wrap_if_needed(obj) is obj


class TestCustomClient:
    """Verify that a custom client satisfying the protocol works end-to-end."""

    def test_custom_client_with_orchestrator(self):
        from rlm.config import RLMConfig
        from rlm.orchestrator import Orchestrator

        class SimpleClient:
            def complete(
                self,
                model: str,
                messages: list[dict[str, str]],
                temperature: float,
                max_tokens: int,
                **kwargs: Any,
            ) -> CompletionResult:
                return CompletionResult(
                    content="FINAL(custom answer)",
                    input_tokens=10,
                    output_tokens=5,
                )

        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(SimpleClient(), config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "custom answer"
