"""Integration tests for the orchestrator with a mock OpenAI client."""

from __future__ import annotations

from typing import Any

from replm.client import CompletionResult
from replm.config import RLMConfig
from replm.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Mock client implementing the LLMClient protocol
# ---------------------------------------------------------------------------


class MockClient:
    """Mock LLM client for testing.

    Accepts a list of scripted responses.  Each ``complete()`` call pops the
    first entry.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
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
        text = self._responses.pop(0) if self._responses else "FINAL(fallback)"
        return CompletionResult(content=text, input_tokens=100, output_tokens=50)


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
        config = RLMConfig(max_iterations=5, max_recursion_depth=0)
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


class TestElapsedSeconds:
    def test_elapsed_seconds_positive(self):
        client = MockClient(["FINAL(answer)"])
        config = RLMConfig()
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.elapsed_seconds > 0


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


class TestNoSubCalls:
    def test_no_subcalls_mode_works(self):
        """In no-sub-calls mode, model can still use code and reach FINAL."""
        client = MockClient(
            [
                "```repl\nprint(type(context))\n```",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5, enable_sub_calls=False)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "done"
        assert resp.sub_calls == 0

    def test_no_subcalls_llm_query_unavailable(self):
        """Calling llm_query in no-sub-calls mode gives a NameError."""
        client = MockClient(
            [
                "```repl\ntry:\n    llm_query('test')\n"
                "except NameError:\n    print('no llm_query')\n```",
                "FINAL(correct)",
            ]
        )
        config = RLMConfig(max_iterations=5, enable_sub_calls=False)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "correct"

    def test_no_subcalls_system_prompt_omits_llm_query(self):
        """The system message should not mention llm_query."""
        client = MockClient(["FINAL(ok)"])
        config = RLMConfig(max_iterations=5, enable_sub_calls=False)
        orch = Orchestrator(client, config, "m", "m")
        orch.run("q", "ctx")
        system_msg = client.calls[0]["messages"][0]["content"]
        assert "llm_query" not in system_msg


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


class TestDeepRecursion:
    def test_depth_2_recursive_sub_call(self):
        """With max_recursion_depth=2, an llm_query triggers an inner
        Orchestrator that itself can run code before returning FINAL.

        Sequence of client.chat.completions.create calls:
        1. Outer root → code calling llm_query("inner prompt")
        2. Inner root (from inner Orchestrator) → FINAL(inner answer)
        3. Outer root → FINAL(outer answer)
        """
        client = MockClient(
            [
                # 1. Outer root: run code that calls llm_query
                '```repl\nresult = llm_query("inner prompt")\n```',
                # 2. Inner orchestrator root call → immediate FINAL
                "FINAL(inner answer)",
                # 3. Outer root: return final answer using the result
                "FINAL(outer answer)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=2)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("Do something recursive.", "ctx")
        assert resp.answer == "outer answer"
        assert resp.sub_calls == 1

    def test_depth_2_inner_runs_code(self):
        """Inner orchestrator executes code before returning FINAL.

        Sequence:
        1. Outer root → code calling llm_query
        2. Inner root → code (print)
        3. Inner root → FINAL(computed)
        4. Outer root → FINAL(done)
        """
        client = MockClient(
            [
                # 1. Outer root: call llm_query
                '```repl\nresult = llm_query("compute 2+2")\n```',
                # 2. Inner root: execute code
                "```repl\nval = 2 + 2\nprint(val)\n```",
                # 3. Inner root: return final
                "FINAL(4)",
                # 4. Outer root: return final
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=2)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("What is 2+2?", "ctx")
        assert resp.answer == "done"
        assert resp.sub_calls == 1

    def test_shared_budget_across_depths(self):
        """Sub-calls at different depths share a single budget."""
        from replm.budget import SharedBudget

        budget = SharedBudget(max_sub_calls=10)
        client = MockClient(
            [
                '```repl\nresult = llm_query("test")\n```',
                "FINAL(inner)",
                "FINAL(outer)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=2)
        orch = Orchestrator(client, config, "m", "m", budget=budget)
        resp = orch.run("q", "ctx")
        assert resp.answer == "outer"
        assert budget.call_count == 1

    def test_depth_1_uses_plain_sub_calls(self):
        """Default max_recursion_depth=1 makes plain (non-recursive) sub-calls."""
        client = MockClient(
            [
                '```repl\nresult = llm_query("summarize")\nprint(result)\n```',
                "plain sub answer",
                "FINAL(done)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=1)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        assert resp.answer == "done"
        assert resp.sub_calls == 1

    def test_recursive_token_tracking(self):
        """Inner orchestrator root tokens are included in the total."""
        client = MockClient(
            [
                '```repl\nresult = llm_query("test")\n```',
                "FINAL(inner)",
                "FINAL(outer)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=2)
        orch = Orchestrator(client, config, "m", "m")
        resp = orch.run("q", "ctx")
        # 3 LLM calls total: outer root x2, inner root x1.
        # Each call: 100 input + 50 output tokens (from mock).
        # Sub-call budget should include inner root tokens.
        assert resp.total_input_tokens == 300
        assert resp.total_output_tokens == 150

    def test_events_include_depth(self):
        """Events from sub-calls include the correct depth value."""
        from replm.types import RLMEvent

        events: list[RLMEvent] = []
        client = MockClient(
            [
                '```repl\nresult = llm_query("test")\n```',
                "FINAL(inner)",
                "FINAL(outer)",
            ]
        )
        config = RLMConfig(max_iterations=5, max_recursion_depth=2)
        orch = Orchestrator(client, config, "m", "m")
        orch.run("q", "ctx", on_event=events.append)
        sub_events = [e for e in events if e.type in ("sub_call_start", "sub_call_end")]
        assert len(sub_events) >= 2
        for e in sub_events:
            assert e.depth == 0  # sub-call originated at depth 0
