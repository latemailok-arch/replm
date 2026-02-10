"""Tests for the streaming orchestrator and StreamChunk."""

from __future__ import annotations

from typing import Any

import pytest

from replm.client import CompletionResult, ContentStream
from replm.config import RLMConfig
from replm.stream import StreamChunk, StreamOrchestrator
from replm.wrapper import RLMWrapper

# ---------------------------------------------------------------------------
# Async mock client with acomplete + optional astream
# ---------------------------------------------------------------------------


class AsyncMockClient:
    """Mock client implementing acomplete() and optionally astream()."""

    def __init__(self, responses: list[str], *, stream: bool = False) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._stream = stream

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append({"model": model, "messages": messages})
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append({"model": model, "messages": messages})
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)


class StreamingMockClient(AsyncMockClient):
    """Mock client that also supports astream()."""

    def astream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> ContentStream:
        self.calls.append({"model": model, "messages": messages})
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"

        async def _gen():
            # Simulate token-by-token streaming by yielding word by word
            words = text.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word

        stream = ContentStream(_gen())
        stream._set_usage(100, 50)
        return stream


# ---------------------------------------------------------------------------
# Tests: StreamChunk
# ---------------------------------------------------------------------------


class TestStreamChunk:
    def test_fields(self):
        chunk = StreamChunk(type="token", content="hello", iteration=1)
        assert chunk.type == "token"
        assert chunk.content == "hello"
        assert chunk.iteration == 1
        assert chunk.detail == {}

    def test_detail(self):
        chunk = StreamChunk(
            type="final_answer",
            content="done",
            iteration=3,
            detail={"response": "some_resp"},
        )
        assert chunk.detail["response"] == "some_resp"


# ---------------------------------------------------------------------------
# Tests: ContentStream
# ---------------------------------------------------------------------------


class TestContentStream:
    @pytest.mark.asyncio
    async def test_iterate_and_result(self):
        async def _gen():
            yield "hello"
            yield " world"

        stream = ContentStream(_gen())
        stream._set_usage(10, 5)
        parts = []
        async for delta in stream:
            parts.append(delta)
        assert parts == ["hello", " world"]
        result = stream.result
        assert result.content == "hello world"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_result_before_consume_raises(self):
        async def _gen():
            yield "x"

        stream = ContentStream(_gen())
        with pytest.raises(RuntimeError, match="not yet consumed"):
            _ = stream.result


# ---------------------------------------------------------------------------
# Tests: StreamOrchestrator (non-streaming fallback)
# ---------------------------------------------------------------------------


class TestStreamNonStreaming:
    @pytest.mark.asyncio
    async def test_immediate_final(self):
        """Non-streaming client yields tokens then final_answer."""
        client = AsyncMockClient(["FINAL(42)"])
        config = RLMConfig(max_iterations=5)
        orch = StreamOrchestrator(client, config, "m", "m")
        chunks: list[StreamChunk] = []
        async for chunk in orch.run("What?", "ctx"):
            chunks.append(chunk)

        types = [c.type for c in chunks]
        assert "iteration_start" in types
        assert "token" in types
        assert "final_answer" in types
        # Final chunk has the response
        final = [c for c in chunks if c.type == "final_answer"][0]
        assert final.content == "42"
        assert "response" in final.detail
        assert final.detail["response"].answer == "42"

    @pytest.mark.asyncio
    async def test_code_then_final(self):
        """Multi-iteration: code execution then final answer."""
        client = AsyncMockClient(
            [
                "```repl\nx = 42\nprint(x)\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = StreamOrchestrator(client, config, "m", "m")
        chunks: list[StreamChunk] = []
        async for chunk in orch.run("q", "ctx"):
            chunks.append(chunk)

        types = [c.type for c in chunks]
        assert "code_executed" in types
        assert types.count("iteration_start") == 2
        final = [c for c in chunks if c.type == "final_answer"][0]
        assert final.content == "done"

    @pytest.mark.asyncio
    async def test_response_has_tokens(self):
        """Token counts are tracked in the final response."""
        client = AsyncMockClient(["FINAL(answer)"])
        config = RLMConfig()
        orch = StreamOrchestrator(client, config, "m", "m")
        chunks = [c async for c in orch.run("q", "ctx")]
        resp = chunks[-1].detail["response"]
        assert resp.total_input_tokens > 0
        assert resp.total_output_tokens > 0


# ---------------------------------------------------------------------------
# Tests: StreamOrchestrator (with real streaming client)
# ---------------------------------------------------------------------------


class TestStreamWithStreaming:
    @pytest.mark.asyncio
    async def test_token_chunks_yielded(self):
        """Streaming client yields individual token chunks."""
        client = StreamingMockClient(["FINAL(hello world)"])
        config = RLMConfig(max_iterations=5)
        orch = StreamOrchestrator(client, config, "m", "m")
        chunks = [c async for c in orch.run("q", "ctx")]

        token_chunks = [c for c in chunks if c.type == "token"]
        # "FINAL(hello world)" split by spaces → "FINAL(hello", " world)"
        assert len(token_chunks) >= 2
        combined = "".join(c.content for c in token_chunks)
        assert combined == "FINAL(hello world)"

    @pytest.mark.asyncio
    async def test_multi_iteration_streaming(self):
        """Streaming works across multiple iterations."""
        client = StreamingMockClient(
            [
                "```repl\nprint('hi')\n```",
                "FINAL(ok)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = StreamOrchestrator(client, config, "m", "m")
        chunks = [c async for c in orch.run("q", "ctx")]

        types = [c.type for c in chunks]
        assert "code_executed" in types
        assert types.count("iteration_start") == 2


# ---------------------------------------------------------------------------
# Tests: RLMWrapper.astream_generate
# ---------------------------------------------------------------------------


class TestWrapperStreamGenerate:
    @pytest.mark.asyncio
    async def test_astream_generate(self):
        """RLMWrapper.astream_generate yields chunks and final response."""
        client = AsyncMockClient(["FINAL(streamed answer)"])
        wrapper = RLMWrapper(client, root_model="m")
        chunks = [c async for c in wrapper.astream_generate("q", "ctx")]

        final = [c for c in chunks if c.type == "final_answer"]
        assert len(final) == 1
        assert final[0].content == "streamed answer"
        assert final[0].detail["response"].answer == "streamed answer"

    @pytest.mark.asyncio
    async def test_astream_generate_multi_iteration(self):
        """astream_generate handles code execution iterations."""
        client = AsyncMockClient(
            [
                "```repl\nresult = 'computed'\n```",
                "FINAL_VAR(result)",
            ]
        )
        wrapper = RLMWrapper(client, root_model="m")
        chunks = [c async for c in wrapper.astream_generate("q", "ctx")]

        final = [c for c in chunks if c.type == "final_answer"]
        assert final[0].content == "computed"
