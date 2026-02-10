"""Tests for rlm.prompt."""

from rlm.config import RLMConfig
from rlm.prompt import build_nudge_prompt, build_root_system_prompt


class TestBuildRootSystemPrompt:
    def _build(self, **kwargs):
        config = RLMConfig(**kwargs)
        return build_root_system_prompt(
            context_type="string",
            context_total_length=50_000,
            context_lengths=[50_000],
            config=config,
        )

    def test_contains_context_metadata(self):
        prompt = self._build()
        assert "string" in prompt
        assert "50,000 total characters" in prompt
        assert "[50000]" in prompt

    def test_contains_repl_init(self):
        prompt = self._build()
        assert "`context` variable" in prompt
        assert "`llm_query` function" in prompt
        assert "`print()`" in prompt

    def test_contains_final_instructions(self):
        prompt = self._build()
        assert "FINAL(your final answer here)" in prompt
        assert "FINAL_VAR(variable_name)" in prompt

    def test_contains_code_examples(self):
        prompt = self._build()
        assert "```repl" in prompt
        assert "llm_query(" in prompt
        assert "chunk" in prompt

    def test_contains_batching_guidance(self):
        prompt = self._build()
        assert "10 documents per sub-LLM query" in prompt

    def test_contains_markdown_header_example(self):
        prompt = self._build()
        assert "Markdown headers" in prompt
        assert "re.split" in prompt
        assert "Summarize this" in prompt
        assert "FINAL_VAR(final_answer)" in prompt

    def test_contains_regex_example(self):
        prompt = self._build()
        assert "re.finditer" in prompt
        assert "climate|emissions|carbon" in prompt

    def test_sub_call_capacity_from_config(self):
        prompt = self._build(sub_call_max_input_chars=200_000)
        assert "200,000" in prompt

    def test_list_context_type(self):
        config = RLMConfig()
        prompt = build_root_system_prompt(
            context_type="list of 5 strings",
            context_total_length=100_000,
            context_lengths=[20000, 20000, 20000, 20000, 20000],
            config=config,
        )
        assert "list of 5 strings" in prompt
        assert "100,000 total characters" in prompt


class TestBuildNudgePrompt:
    def test_nudge_mentions_final(self):
        prompt = build_nudge_prompt()
        assert "FINAL" in prompt
        assert "FINAL_VAR" in prompt

    def test_nudge_mentions_max_iterations(self):
        prompt = build_nudge_prompt()
        assert "maximum number of iterations" in prompt
