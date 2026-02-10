"""Unit tests for rlm.sub_caller.SubCallManager."""

from __future__ import annotations

from typing import Any

import pytest

from replm.budget import SharedBudget
from replm.client import CompletionResult
from replm.config import RLMConfig
from replm.exceptions import MaxSubCallsExceeded
from replm.sub_caller import SubCallManager
from replm.types import RLMEvent

# -- Mock client implementing LLMClient protocol ----------------------------

# Use max_recursion_depth=0 for tests that verify plain (non-recursive) sub-call
# behaviour.  With the default (1), depth=0 triggers the recursive path which
# spins up a full inner Orchestrator — not what these unit tests intend.
_PLAIN_CONFIG = RLMConfig(max_recursion_depth=0)


class _MockClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> CompletionResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return CompletionResult(content="mock response", input_tokens=10, output_tokens=5)


# -- Tests -------------------------------------------------------------------


class TestMakeQueryFn:
    def test_returns_callable(self):
        mgr = SubCallManager(_MockClient(), _PLAIN_CONFIG, "test-model")
        fn = mgr.make_query_fn()
        assert callable(fn)

    def test_basic_call(self):
        client = _MockClient()
        mgr = SubCallManager(client, _PLAIN_CONFIG, "test-model")
        fn = mgr.make_query_fn()
        result = fn("hello world")
        assert result == "mock response"
        assert mgr.call_count == 1
        assert len(client.calls) == 1

    def test_model_passed_correctly(self):
        client = _MockClient()
        mgr = SubCallManager(client, _PLAIN_CONFIG, "my-special-model")
        fn = mgr.make_query_fn()
        fn("test")
        call = client.calls[0]
        assert call["model"] == "my-special-model"

    def test_temperature_from_config(self):
        client = _MockClient()
        config = RLMConfig(sub_temperature=0.7, max_recursion_depth=0)
        mgr = SubCallManager(client, config, "m")
        fn = mgr.make_query_fn()
        fn("test")
        call = client.calls[0]
        assert call["temperature"] == 0.7

    def test_max_tokens_from_config(self):
        client = _MockClient()
        config = RLMConfig(sub_max_tokens=4096, max_recursion_depth=0)
        mgr = SubCallManager(client, config, "m")
        fn = mgr.make_query_fn()
        fn("test")
        call = client.calls[0]
        assert call["max_tokens"] == 4096


class TestTokenTracking:
    def test_tokens_accumulated(self):
        mgr = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m")
        fn = mgr.make_query_fn()
        fn("a")
        fn("b")
        fn("c")
        assert mgr.call_count == 3
        assert mgr.total_input_tokens == 30  # 10 * 3
        assert mgr.total_output_tokens == 15  # 5 * 3


class TestInputTruncation:
    def test_long_input_truncated(self):
        client = _MockClient()
        config = RLMConfig(sub_call_max_input_chars=100, max_recursion_depth=0)
        mgr = SubCallManager(client, config, "m")
        fn = mgr.make_query_fn()
        fn("x" * 500)
        call = client.calls[0]
        user_msg = call["messages"][1]["content"]
        assert len(user_msg) == 100

    def test_short_input_not_truncated(self):
        client = _MockClient()
        config = RLMConfig(sub_call_max_input_chars=1000, max_recursion_depth=0)
        mgr = SubCallManager(client, config, "m")
        fn = mgr.make_query_fn()
        fn("short prompt")
        call = client.calls[0]
        user_msg = call["messages"][1]["content"]
        assert user_msg == "short prompt"


class TestMaxSubCalls:
    def test_raises_when_limit_reached(self):
        config = RLMConfig(max_sub_calls=2, max_recursion_depth=0)
        mgr = SubCallManager(_MockClient(), config, "m")
        fn = mgr.make_query_fn()
        fn("a")
        fn("b")
        with pytest.raises(MaxSubCallsExceeded) as exc_info:
            fn("c")
        assert exc_info.value.call_count == 2
        assert exc_info.value.max_sub_calls == 2

    def test_limit_of_one(self):
        config = RLMConfig(max_sub_calls=1, max_recursion_depth=0)
        mgr = SubCallManager(_MockClient(), config, "m")
        fn = mgr.make_query_fn()
        fn("a")
        with pytest.raises(MaxSubCallsExceeded):
            fn("b")


class TestSharedBudget:
    def test_budget_shared_between_managers(self):
        """Two managers sharing a budget accumulate into the same counters."""
        budget = SharedBudget(max_sub_calls=100)
        mgr1 = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m", budget=budget)
        mgr2 = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m", budget=budget)
        fn1 = mgr1.make_query_fn()
        fn2 = mgr2.make_query_fn()
        fn1("a")
        fn2("b")
        fn1("c")
        assert budget.call_count == 3
        assert mgr1.call_count == 3
        assert mgr2.call_count == 3
        assert budget.total_input_tokens == 30  # 10 * 3
        assert budget.total_output_tokens == 15  # 5 * 3

    def test_budget_limit_shared(self):
        """Shared budget enforces limit across multiple managers."""
        budget = SharedBudget(max_sub_calls=3)
        cfg = RLMConfig(max_recursion_depth=0)
        mgr1 = SubCallManager(_MockClient(), cfg, "m", budget=budget)
        mgr2 = SubCallManager(_MockClient(), cfg, "m", budget=budget)
        fn1 = mgr1.make_query_fn()
        fn2 = mgr2.make_query_fn()
        fn1("a")
        fn2("b")
        fn1("c")
        with pytest.raises(MaxSubCallsExceeded):
            fn2("d")

    def test_auto_creates_budget_when_none(self):
        """Manager creates its own budget when none is provided."""
        config = RLMConfig(max_sub_calls=10, max_recursion_depth=0)
        mgr = SubCallManager(_MockClient(), config, "m")
        assert mgr.budget.max_sub_calls == 10
        fn = mgr.make_query_fn()
        fn("a")
        assert mgr.budget.call_count == 1

    def test_budget_property_exposed(self):
        budget = SharedBudget(max_sub_calls=50)
        mgr = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m", budget=budget)
        assert mgr.budget is budget


class TestDepthDispatch:
    def test_depth_at_max_uses_plain_call(self):
        """At depth >= max_recursion_depth, uses plain LLM call."""
        client = _MockClient()
        config = RLMConfig(max_recursion_depth=1)
        mgr = SubCallManager(client, config, "m", depth=1)
        fn = mgr.make_query_fn()
        result = fn("test")
        assert result == "mock response"
        assert len(client.calls) == 1

    def test_depth_below_max_uses_recursive_call(self):
        """At depth < max_recursion_depth, uses recursive inner Orchestrator."""
        client = _MockClient()
        config = RLMConfig(max_recursion_depth=2, max_iterations=1)
        mgr = SubCallManager(client, config, "m", depth=0)
        fn = mgr.make_query_fn()
        fn("test")
        # Recursive path makes multiple client calls (inner orchestrator loop)
        assert len(client.calls) > 1


class TestEventCallbacks:
    def test_events_fired(self):
        events: list[RLMEvent] = []
        mgr = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m")
        mgr.set_event_callback(events.append)
        mgr.set_current_iteration(3)
        fn = mgr.make_query_fn()
        fn("hello")
        assert len(events) == 2
        assert events[0].type == "sub_call_start"
        assert events[0].iteration == 3
        assert events[1].type == "sub_call_end"

    def test_no_events_without_callback(self):
        mgr = SubCallManager(_MockClient(), _PLAIN_CONFIG, "m")
        fn = mgr.make_query_fn()
        fn("hello")  # should not raise

    def test_events_include_depth(self):
        events: list[RLMEvent] = []
        config = RLMConfig(max_recursion_depth=0)
        mgr = SubCallManager(_MockClient(), config, "m", depth=2)
        mgr.set_event_callback(events.append)
        mgr.set_current_iteration(1)
        fn = mgr.make_query_fn()
        fn("hello")
        assert events[0].depth == 2
        assert events[1].depth == 2
