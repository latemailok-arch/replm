"""Tests for rlm.prompt."""

import pytest

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

    def test_default_chunk_peek(self):
        prompt = self._build()
        assert "context[:10000]" in prompt

    def test_include_batch_fn(self):
        config = RLMConfig()
        prompt = build_root_system_prompt(
            context_type="string",
            context_total_length=1000,
            context_lengths=[1000],
            config=config,
            include_batch_fn=True,
        )
        assert "llm_query_batch" in prompt
        assert "parallel" in prompt

    def test_no_batch_fn_by_default(self):
        prompt = self._build()
        assert "llm_query_batch" not in prompt


class TestPromptVariants:
    def _build(self, variant, **kwargs):
        config = RLMConfig(prompt_variant=variant, **kwargs)
        return build_root_system_prompt(
            context_type="string",
            context_total_length=50_000,
            context_lengths=[50_000],
            config=config,
        )

    def test_default_variant_no_cost_warning(self):
        prompt = self._build("default")
        assert "incurs high runtime costs" not in prompt
        assert "~32k tokens" not in prompt

    def test_default_variant_has_standard_capacity(self):
        prompt = self._build("default")
        assert "500,000" in prompt
        assert "10 documents per sub-LLM query" in prompt
        assert "context[:10000]" in prompt

    def test_cost_warning_variant(self):
        prompt = self._build("cost_warning")
        assert "incurs high runtime costs" in prompt
        assert "~200k characters per call" in prompt
        assert "batching related information together" in prompt

    def test_cost_warning_retains_standard_capacity(self):
        prompt = self._build("cost_warning")
        assert "500,000" in prompt
        assert "context[:10000]" in prompt
        assert "10 documents per sub-LLM query" in prompt

    def test_cost_warning_no_32k(self):
        prompt = self._build("cost_warning")
        assert "~32k tokens" not in prompt

    def test_small_context_variant_token_warning(self):
        prompt = self._build("small_context")
        assert "~32k tokens" in prompt
        assert "be conservative" in prompt

    def test_small_context_variant_capacity(self):
        prompt = self._build("small_context")
        assert "~100k chars, roughly 32k tokens" in prompt

    def test_small_context_variant_batching(self):
        prompt = self._build("small_context")
        assert "2-3 documents per sub-LLM query" in prompt
        assert "10 documents per sub-LLM query" not in prompt

    def test_small_context_variant_chunk_peek(self):
        prompt = self._build("small_context")
        assert "context[:1000]" in prompt
        assert "context[:10000]" not in prompt

    def test_small_context_variant_cost_warning(self):
        prompt = self._build("small_context")
        assert "~10k-15k characters per call" in prompt
        assert "~32k token limit" in prompt

    def test_small_context_variant_final_instruction(self):
        prompt = self._build("small_context")
        assert "NOT in code or repl tags" in prompt

    def test_small_context_24k_chars(self):
        prompt = self._build("small_context")
        assert "~24k characters" in prompt

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt_variant"):
            RLMConfig(prompt_variant="nonexistent")


class TestNoSubCallsPrompt:
    def _build(self, **kwargs):
        config = RLMConfig(enable_sub_calls=False, **kwargs)
        return build_root_system_prompt(
            context_type="string",
            context_total_length=50_000,
            context_lengths=[50_000],
            config=config,
        )

    def test_omits_llm_query(self):
        prompt = self._build()
        assert "llm_query" not in prompt

    def test_omits_sub_llm_mentions(self):
        prompt = self._build()
        assert "sub-LLM" not in prompt
        assert "recursive" not in prompt.lower().split("query")[0]

    def test_has_context_variable(self):
        prompt = self._build()
        assert "`context` variable" in prompt

    def test_has_print_statement(self):
        prompt = self._build()
        assert "`print()`" in prompt

    def test_has_code_examples(self):
        prompt = self._build()
        assert "```repl" in prompt
        assert "regex" in prompt.lower() or "re." in prompt

    def test_has_buffer_examples(self):
        prompt = self._build()
        assert "buffers" in prompt

    def test_has_final_instructions(self):
        prompt = self._build()
        assert "FINAL(your final answer here)" in prompt
        assert "FINAL_VAR(variable_name)" in prompt

    def test_has_no_code_note(self):
        prompt = self._build()
        assert "you cannot write anything other than the final answer" in prompt

    def test_has_context_metadata(self):
        prompt = self._build()
        assert "50,000 total characters" in prompt


class TestBuildNudgePrompt:
    def test_nudge_mentions_final(self):
        prompt = build_nudge_prompt()
        assert "FINAL" in prompt
        assert "FINAL_VAR" in prompt

    def test_nudge_mentions_max_iterations(self):
        prompt = build_nudge_prompt()
        assert "maximum number of iterations" in prompt
