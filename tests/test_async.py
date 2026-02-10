"""Tests for the async orchestrator and async sub-call manager."""

from __future__ import annotations

from typing import Any

import pytest

from rlm.async_orchestrator import AsyncOrchestrator
from rlm.client import CompletionResult
from rlm.config import RLMConfig
from rlm.types import RLMEvent
from rlm.wrapper import RLMWrapper

# ---------------------------------------------------------------------------
# Async mock client implementing LLMClient protocol
# ---------------------------------------------------------------------------


class AsyncMockClient:
    """Mock async LLM client for testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def acomplete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncBasicLoop:
    @pytest.mark.asyncio
    async def test_immediate_final(self):
        client = AsyncMockClient(["FINAL(42)"])
        config = RLMConfig(max_iterations=5)
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("What is the answer?", "some context")
        assert resp.answer == "42"
        assert resp.iterations == 1
        assert resp.sub_calls == 0

    @pytest.mark.asyncio
    async def test_code_then_final(self):
        client = AsyncMockClient(
            [
                "Let me check.\n```repl\nx = len(context)\nprint(x)\n```",
                "FINAL(The context has many characters)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("How long?", "hello world")
        assert resp.answer == "The context has many characters"
        assert resp.iterations == 2

    @pytest.mark.asyncio
    async def test_final_var(self):
        client = AsyncMockClient(
            [
                "```repl\nresult = 'computed answer'\n```",
                "FINAL_VAR(result)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("Compute.", "data")
        assert resp.answer == "computed answer"


class TestAsyncSubCalls:
    @pytest.mark.asyncio
    async def test_sub_call_tracked(self):
        """llm_query from REPL uses async sub-call manager."""
        client = AsyncMockClient(
            [
                '```repl\nresult = llm_query("summarize: " + context[:50])\nprint(result)\n```',
                "sub answer",
                "FINAL(Done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=0)
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("Summarize.", "A long document.")
        assert resp.sub_calls == 1
        assert resp.answer == "Done"

    @pytest.mark.asyncio
    async def test_batch_sub_calls(self):
        """llm_query_batch runs multiple sub-calls concurrently."""
        client = AsyncMockClient(
            [
                ('```repl\nresults = llm_query_batch(["q1", "q2", "q3"])\nprint(results)\n```'),
                "answer1",  # sub-call for q1
                "answer2",  # sub-call for q2
                "answer3",  # sub-call for q3
                "FINAL(all done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=0)
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("Batch.", "ctx")
        assert resp.sub_calls == 3
        assert resp.answer == "all done"


class TestAsyncTokenTracking:
    @pytest.mark.asyncio
    async def test_tokens_aggregated(self):
        client = AsyncMockClient(["FINAL(answer)"])
        config = RLMConfig()
        orch = AsyncOrchestrator(client, config, "m", "m")
        resp = await orch.run("q", "ctx")
        assert resp.total_input_tokens > 0
        assert resp.total_output_tokens > 0


class TestAsyncEvents:
    @pytest.mark.asyncio
    async def test_events_fired(self):
        events: list[str] = []
        client = AsyncMockClient(
            [
                "```repl\nprint('hi')\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = AsyncOrchestrator(client, config, "m", "m")
        await orch.run("q", "ctx", on_event=lambda e: events.append(e.type))
        assert "iteration_start" in events
        assert "code_generated" in events
        assert "code_executed" in events
        assert "final_answer" in events


class TestAsyncSystemPrompt:
    @pytest.mark.asyncio
    async def test_batch_fn_mentioned_in_prompt(self):
        """Async orchestrator includes llm_query_batch in the system prompt."""
        client = AsyncMockClient(["FINAL(ok)"])
        config = RLMConfig(max_iterations=5)
        orch = AsyncOrchestrator(client, config, "m", "m")
        await orch.run("q", "ctx")
        system_msg = client.calls[0]["messages"][0]["content"]
        assert "llm_query_batch" in system_msg

    @pytest.mark.asyncio
    async def test_no_subcalls_omits_batch(self):
        """No-sub-calls mode excludes llm_query_batch."""
        client = AsyncMockClient(["FINAL(ok)"])
        config = RLMConfig(max_iterations=5, enable_sub_calls=False)
        orch = AsyncOrchestrator(client, config, "m", "m")
        await orch.run("q", "ctx")
        system_msg = client.calls[0]["messages"][0]["content"]
        assert "llm_query_batch" not in system_msg
        assert "llm_query" not in system_msg


class TestAsyncSubCallEvents:
    @pytest.mark.asyncio
    async def test_sub_call_events_include_depth(self):
        events: list[RLMEvent] = []
        client = AsyncMockClient(
            [
                '```repl\nresult = llm_query("test")\n```',
                "sub answer",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=0)
        orch = AsyncOrchestrator(client, config, "m", "m")
        await orch.run("q", "ctx", on_event=events.append)
        sub_events = [e for e in events if e.type in ("sub_call_start", "sub_call_end")]
        assert len(sub_events) == 2
        for e in sub_events:
            assert e.depth == 0


class TestWrapperAgenerate:
    @pytest.mark.asyncio
    async def test_agenerate(self):
        client = AsyncMockClient(["FINAL(async answer)"])
        wrapper = RLMWrapper(client, root_model="m")
        resp = await wrapper.agenerate("q", "ctx")
        assert resp.answer == "async answer"
