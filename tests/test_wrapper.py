"""Tests for the public RLMWrapper API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from replm import RLMConfig, RLMWrapper


@dataclass
class _Usage:
    prompt_tokens: int = 50
    completion_tokens: int = 25


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
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def create(self, **kwargs: Any) -> _CompletionResponse:
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return _CompletionResponse(
            choices=[_Choice(message=_Message(content=text))],
            usage=_Usage(),
        )


class _MockChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _MockCompletions(responses)


class MockClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _MockChat(responses)


class TestRLMWrapper:
    def test_generate_returns_response(self):
        client = MockClient(
            [
                "```repl\nprint('ok')\n```",
                "FINAL(hello world)",
            ]
        )
        wrapper = RLMWrapper(client, root_model="test-model")
        resp = wrapper.generate("Say hello.", "context data")
        assert resp.answer == "hello world"
        assert resp.iterations == 2

    def test_sub_model_defaults_to_root(self):
        client = MockClient(["FINAL(ok)"])
        wrapper = RLMWrapper(client, root_model="my-model")
        assert wrapper._sub_model == "my-model"

    def test_custom_config(self):
        client = MockClient(["FINAL(ok)"])
        config = RLMConfig(max_iterations=3, verbose=False)
        wrapper = RLMWrapper(client, root_model="m", config=config)
        assert wrapper._config.max_iterations == 3

    def test_code_and_final_var_same_response(self):
        """Model outputs code + FINAL_VAR in one turn; code must run first."""
        client = MockClient(
            [
                "```repl\ncomputed = 'the answer'\n```\nFINAL_VAR(computed)",
            ]
        )
        wrapper = RLMWrapper(client, root_model="m")
        resp = wrapper.generate("q", "ctx")
        assert resp.answer == "the answer"
        assert resp.iterations == 1

    def test_on_event_callback(self):
        events: list[str] = []
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(done)"])
        wrapper = RLMWrapper(client, root_model="m")
        wrapper.generate("q", "ctx", on_event=lambda e: events.append(e.type))
        assert "final_answer" in events

    def test_cost_property(self):
        """response.cost computes from token counts and configured pricing."""
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(ok)"])
        config = RLMConfig(
            cost_per_input_token=2.50 / 1_000_000,
            cost_per_output_token=10.0 / 1_000_000,
        )
        wrapper = RLMWrapper(client, root_model="m", config=config)
        resp = wrapper.generate("q", "ctx")
        assert resp.cost > 0
        expected = (
            resp.total_input_tokens * config.cost_per_input_token
            + resp.total_output_tokens * config.cost_per_output_token
        )
        assert abs(resp.cost - expected) < 1e-12

    def test_cost_zero_without_pricing(self):
        """response.cost is 0.0 when no pricing is configured."""
        client = MockClient(["```repl\nprint('ok')\n```", "FINAL(ok)"])
        wrapper = RLMWrapper(client, root_model="m")
        resp = wrapper.generate("q", "ctx")
        assert resp.cost == 0.0

    def test_final_var_missing_variable_feedback(self):
        """When FINAL_VAR references nonexistent var, model gets feedback."""
        client = MockClient(
            [
                "FINAL_VAR(nonexistent)",
                "```repl\nresult = 'fixed'\n```\nFINAL_VAR(result)",
            ]
        )
        config = RLMConfig(max_iterations=5)
        wrapper = RLMWrapper(client, root_model="m", config=config)
        resp = wrapper.generate("q", "ctx")
        assert resp.answer == "fixed"
        assert resp.iterations == 2
