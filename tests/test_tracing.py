"""Tests for OpenTelemetry tracing instrumentation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from replm.client import CompletionResult
from replm.config import RLMConfig
from replm.orchestrator import Orchestrator
from replm.tracing import span

# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class MockClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tracer() -> tuple[MagicMock, list[MagicMock]]:
    """Create a mock tracer that records spans.

    Returns (tracer, spans_list) where spans_list collects all created spans.
    """
    spans: list[MagicMock] = []
    tracer = MagicMock()

    def _start_span(name: str) -> MagicMock:
        mock_span = MagicMock()
        mock_span.name = name
        mock_span._attributes: dict[str, Any] = {}

        def _set_attr(key: str, value: Any) -> None:
            mock_span._attributes[key] = value

        mock_span.set_attribute = _set_attr
        mock_span.__enter__ = lambda self: self
        mock_span.__exit__ = lambda self, *args: None
        spans.append(mock_span)
        return mock_span

    tracer.start_as_current_span = _start_span
    return tracer, spans


# ---------------------------------------------------------------------------
# Tests: No-op when OTel is absent
# ---------------------------------------------------------------------------


class TestNoOpTracing:
    def test_span_yields_none_without_otel(self):
        """When tracer is None, span() yields None and doesn't error."""
        with span("rlm.generate") as s:
            assert s is None

    def test_span_with_attributes_noop(self):
        """Passing attributes when tracer is None is safe."""
        with span("rlm.generate", {"rlm.model": "gpt-5.2"}) as s:
            assert s is None

    def test_orchestrator_runs_without_otel(self):
        """Orchestrator works fine without OTel installed."""
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(42)"])
        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "42"


# ---------------------------------------------------------------------------
# Tests: Span creation with mocked tracer
# ---------------------------------------------------------------------------


class TestTracingSpans:
    def test_generate_span_created(self):
        """Orchestrator creates an 'rlm.generate' span."""
        tracer, spans = _make_mock_tracer()
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(42)"])
        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(client, config, "test-model", "test-model")

        with patch("replm.tracing._tracer", tracer):
            resp = orch.run("What?", "ctx")

        assert resp.answer == "42"
        span_names = [s.name for s in spans]
        assert "rlm.generate" in span_names

    def test_generate_span_has_query_len(self):
        """The 'rlm.generate' span has rlm.query_len attribute."""
        tracer, spans = _make_mock_tracer()
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(42)"])
        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(client, config, "test-model", "test-model")

        with patch("replm.tracing._tracer", tracer):
            orch.run("Hello world", "ctx")

        generate_span = next(s for s in spans if s.name == "rlm.generate")
        assert generate_span._attributes["rlm.query_len"] == len("Hello world")
        assert generate_span._attributes["rlm.model"] == "test-model"

    def test_sub_call_span_created(self):
        """Sub-calls create 'rlm.sub_call' spans."""
        tracer, spans = _make_mock_tracer()
        # First response has code that calls llm_query; second is the sub-call
        # response; third is the final answer from the root model.
        client = MockClient(
            [
                "```repl\nresult = llm_query('sub question')\n```",
                "sub answer",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=0)
        orch = Orchestrator(client, config, "root-model", "sub-model")

        with patch("replm.tracing._tracer", tracer):
            resp = orch.run("q", "ctx")

        assert resp.answer == "done"
        span_names = [s.name for s in spans]
        assert "rlm.sub_call" in span_names

        sub_span = next(s for s in spans if s.name == "rlm.sub_call")
        assert sub_span._attributes["rlm.depth"] == 0
        assert sub_span._attributes["rlm.input_tokens"] == 100
        assert sub_span._attributes["rlm.output_tokens"] == 50


# ---------------------------------------------------------------------------
# Tests: Final span attributes
# ---------------------------------------------------------------------------


class TestTracingAttributes:
    def test_final_attributes_set(self):
        """Final response sets iterations, sub_calls, tokens, elapsed on span."""
        tracer, spans = _make_mock_tracer()
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(42)"])
        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(client, config, "m", "m")

        with patch("replm.tracing._tracer", tracer):
            orch.run("q", "ctx")

        generate_span = next(s for s in spans if s.name == "rlm.generate")
        attrs = generate_span._attributes
        assert attrs["rlm.iterations"] == 2
        assert attrs["rlm.sub_calls"] == 0
        assert attrs["rlm.total_input_tokens"] == 200
        assert attrs["rlm.total_output_tokens"] == 100
        assert "rlm.elapsed_seconds" in attrs
        assert attrs["rlm.elapsed_seconds"] >= 0

    def test_multi_iteration_attributes(self):
        """Span attributes reflect multiple iterations."""
        tracer, spans = _make_mock_tracer()
        client = MockClient(
            [
                "```repl\nx = 1\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")

        with patch("replm.tracing._tracer", tracer):
            orch.run("q", "ctx")

        generate_span = next(s for s in spans if s.name == "rlm.generate")
        assert generate_span._attributes["rlm.iterations"] == 2
        # 2 root calls × 100 input tokens each
        assert generate_span._attributes["rlm.total_input_tokens"] == 200


# ---------------------------------------------------------------------------
# Tests: Async orchestrator tracing
# ---------------------------------------------------------------------------


class TestAsyncTracing:
    @pytest.mark.asyncio
    async def test_async_generate_span(self):
        """AsyncOrchestrator creates an 'rlm.generate' span."""
        from replm.async_orchestrator import AsyncOrchestrator

        tracer, spans = _make_mock_tracer()
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(async answer)"])
        config = RLMConfig(max_iterations=3)
        orch = AsyncOrchestrator(client, config, "m", "m")

        with patch("replm.tracing._tracer", tracer):
            resp = await orch.run("q", "ctx")

        assert resp.answer == "async answer"
        span_names = [s.name for s in spans]
        assert "rlm.generate" in span_names

    @pytest.mark.asyncio
    async def test_async_final_attributes(self):
        """AsyncOrchestrator sets final attributes on the root span."""
        from replm.async_orchestrator import AsyncOrchestrator

        tracer, spans = _make_mock_tracer()
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(42)"])
        config = RLMConfig(max_iterations=3)
        orch = AsyncOrchestrator(client, config, "m", "m")

        with patch("replm.tracing._tracer", tracer):
            await orch.run("q", "ctx")

        generate_span = next(s for s in spans if s.name == "rlm.generate")
        attrs = generate_span._attributes
        assert attrs["rlm.iterations"] == 2
        assert attrs["rlm.total_input_tokens"] == 200
        assert attrs["rlm.total_output_tokens"] == 100
