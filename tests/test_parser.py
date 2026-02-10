"""Tests for rlm.parser."""

from rlm.parser import parse_response


class TestCodeBlockExtraction:
    def test_single_repl_block(self):
        text = "Let me check.\n```repl\nprint(context[:100])\n```\nDone."
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 1
        assert "print(context[:100])" in parsed.code_blocks[0]
        assert not parsed.is_done

    def test_single_python_block(self):
        text = "```python\nx = 42\nprint(x)\n```"
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 1
        assert "x = 42" in parsed.code_blocks[0]

    def test_multiple_blocks(self):
        text = "Step 1:\n```repl\na = 1\n```\nStep 2:\n```python\nb = 2\n```"
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 2

    def test_no_code_blocks(self):
        text = "I need to think about this more."
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 0

    def test_empty_code_block(self):
        text = "```repl\n\n```"
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 1
        assert parsed.code_blocks[0].strip() == ""

    def test_ignores_other_language_fences(self):
        text = '```json\n{"key": "value"}\n```'
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 0


class TestFinalDirective:
    def test_simple_final(self):
        text = "After analysis:\nFINAL(The answer is 42)"
        parsed = parse_response(text)
        assert parsed.is_done
        assert parsed.final_answer == "The answer is 42"
        assert parsed.final_var is None

    def test_multiline_final(self):
        text = "FINAL(Line 1\nLine 2\nLine 3)"
        parsed = parse_response(text)
        assert parsed.is_done
        assert "Line 1" in parsed.final_answer
        assert "Line 3" in parsed.final_answer

    def test_final_var(self):
        text = "Done.\nFINAL_VAR(my_result)"
        parsed = parse_response(text)
        assert parsed.is_done
        assert parsed.final_var == "my_result"
        assert parsed.final_answer is None

    def test_final_var_with_underscores(self):
        text = "FINAL_VAR(final_answer_v2)"
        parsed = parse_response(text)
        assert parsed.final_var == "final_answer_v2"

    def test_final_inside_code_block_is_ignored(self):
        text = "```repl\nFinal = 'hello'\nprint(FINAL(nope))\n```\nStill working."
        parsed = parse_response(text)
        assert not parsed.is_done
        assert parsed.final_answer is None

    def test_final_after_code_block(self):
        text = "```repl\nx = 1\n```\nFINAL(The result is x=1)"
        parsed = parse_response(text)
        assert parsed.is_done
        assert parsed.final_answer == "The result is x=1"

    def test_final_with_whitespace(self):
        text = "FINAL(  trimmed answer  )"
        parsed = parse_response(text)
        assert parsed.is_done
        assert parsed.final_answer == "trimmed answer"

    def test_final_var_with_whitespace(self):
        text = "FINAL_VAR(  my_var  )"
        parsed = parse_response(text)
        assert parsed.final_var == "my_var"


class TestReasoning:
    def test_reasoning_extracted(self):
        text = "Let me think.\n```repl\nx=1\n```\nOkay, now I know."
        parsed = parse_response(text)
        assert "Let me think." in parsed.reasoning
        assert "Okay, now I know." in parsed.reasoning
        assert "x=1" not in parsed.reasoning


class TestEdgeCases:
    def test_empty_input(self):
        parsed = parse_response("")
        assert not parsed.is_done
        assert len(parsed.code_blocks) == 0

    def test_final_with_nested_parens(self):
        text = "FINAL(f(x) = (x+1))"
        parsed = parse_response(text)
        assert parsed.is_done
        # Should capture content correctly despite nested parens
        assert parsed.final_answer is not None

    def test_code_and_final_same_response(self):
        text = "```repl\nresult = compute()\n```\nFINAL_VAR(result)"
        parsed = parse_response(text)
        assert len(parsed.code_blocks) == 1
        assert parsed.is_done
        assert parsed.final_var == "result"
