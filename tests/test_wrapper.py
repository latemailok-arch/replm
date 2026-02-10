"""Tests for the public RLMWrapper API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm import RLMConfig, RLMWrapper


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
        client = MockClient(["FINAL(hello world)"])
        wrapper = RLMWrapper(client, root_model="test-model")
        resp = wrapper.generate("Say hello.", "context data")
        assert resp.answer == "hello world"
        assert resp.iterations == 1

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
        client = MockClient(["FINAL(done)"])
        wrapper = RLMWrapper(client, root_model="m")
        wrapper.generate("q", "ctx", on_event=lambda e: events.append(e.type))
        assert "final_answer" in events
