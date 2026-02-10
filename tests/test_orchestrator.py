"""Integration tests for the orchestrator with a mock OpenAI client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm.config import RLMConfig
from rlm.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Mock OpenAI client
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class _Message:
    content: str = ""


@dataclass
class _Choice:
    message: _Message


@dataclass
class _CompletionResponse:
    choices: list[_Choice]
    usage: _Usage


class _MockCompletions:
    """Simulates ``client.chat.completions.create()``.

    Accepts a list of scripted responses.  Each call pops the first entry.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _CompletionResponse:
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return _CompletionResponse(
            choices=[_Choice(message=_Message(content=text))],
            usage=_Usage(),
        )


class _MockChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _MockCompletions(responses)


class MockClient:
    """Minimal mock of an OpenAI client."""

    def __init__(self, responses: list[str]) -> None:
        self.chat = _MockChat(responses)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicLoop:
    def test_immediate_final(self):
        """Model returns FINAL on the first iteration."""
        client = MockClient(["FINAL(42)"])
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("What is the answer?", "some context")
        assert resp.answer == "42"
        assert resp.iterations == 1
        assert resp.sub_calls == 0

    def test_code_then_final(self):
        """Model runs code first, then gives FINAL on the second iteration."""
        client = MockClient(
            [
                "Let me check.\n```repl\nx = len(context)\nprint(x)\n```",
                "FINAL(The context has many characters)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("How long is the context?", "hello world")
        assert resp.answer == "The context has many characters"
        assert resp.iterations == 2

    def test_final_var(self):
        """Model sets a variable and uses FINAL_VAR to return it."""
        client = MockClient(
            [
                "```repl\nresult = 'computed answer'\n```",
                "FINAL_VAR(result)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("Compute something.", "data")
        assert resp.answer == "computed answer"

    def test_max_iterations_nudge(self):
        """When max iterations are reached, a nudge message is sent."""
        # 3 iterations of "thinking" + 1 nudge response
        client = MockClient(
            [
                "```repl\nprint('step 1')\n```",
                "```repl\nprint('step 2')\n```",
                "```repl\nprint('step 3')\n```",
                "FINAL(forced answer)",
            ]
        )
        config = RLMConfig(max_iterations=3)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("What?", "ctx")
        assert "forced answer" in resp.answer

    def test_fallback_answer_variable(self):
        """When nudge still has no FINAL, fall back to known variable names."""
        client = MockClient(
            [
                "```repl\nanswer = 'my answer'\n```",
                "```repl\nprint('still thinking')\n```",
                "I don't know how to wrap up.",  # nudge response without FINAL
            ]
        )
        config = RLMConfig(max_iterations=2)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("What?", "ctx")
        assert resp.answer == "my answer"


class TestSubCalls:
    def test_sub_calls_tracked(self):
        """Sub-calls made via llm_query are counted."""
        # The root model asks to call llm_query; the sub-call also needs a response.
        # Our mock returns responses in order. The root model's code will call
        # llm_query which in turn calls client.chat.completions.create.
        #
        # Sequence:
        # 1. Root call → code that calls llm_query
        # 2. Sub-call (from llm_query) → "sub answer"
        # 3. Root call → FINAL
        client = MockClient(
            [
                '```repl\nresult = llm_query("summarize: " + context[:100])\nprint(result)\n```',
                "sub answer from LLM",  # this is the sub-call response
                "FINAL(Done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("Summarize.", "A long document about many things.")
        assert resp.sub_calls == 1
        assert resp.answer == "Done"


class TestTokenTracking:
    def test_tokens_aggregated(self):
        client = MockClient(["FINAL(answer)"])
        config = RLMConfig()
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.total_input_tokens > 0
        assert resp.total_output_tokens > 0


class TestEvents:
    def test_events_fired(self):
        events: list[str] = []
        client = MockClient(
            [
                "```repl\nprint('hi')\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        orch.run("q", "ctx", on_event=lambda e: events.append(e.type))
        assert "iteration_start" in events
        assert "code_generated" in events
        assert "code_executed" in events
        assert "final_answer" in events


class TestReplVariablesInResponse:
    def test_repl_variables_returned(self):
        client = MockClient(
            [
                "```repl\nmy_data = [1, 2, 3]\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert "my_data" in resp.repl_variables


class TestListContext:
    def test_list_context_accessible(self):
        client = MockClient(
            [
                "```repl\nprint(len(context))\nprint(context[0])\n```",
                "FINAL(ok)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", ["doc1", "doc2", "doc3"])
        assert resp.answer == "ok"
