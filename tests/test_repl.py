"""Tests for rlm.repl."""

from rlm.repl import REPLEnvironment


def _noop_llm_query(prompt: str) -> str:
    return f"echo: {prompt[:50]}"


class TestExecution:
    def test_simple_print(self):
        repl = REPLEnvironment("hello", _noop_llm_query)
        stdout, had_error = repl.execute("print('world')")
        assert stdout.strip() == "world"
        assert not had_error

    def test_variable_persistence(self):
        repl = REPLEnvironment("hello", _noop_llm_query)
        repl.execute("x = 42")
        stdout, _ = repl.execute("print(x)")
        assert "42" in stdout

    def test_context_available(self):
        repl = REPLEnvironment("my context data", _noop_llm_query)
        stdout, _ = repl.execute("print(len(context))")
        assert "15" in stdout

    def test_list_context(self):
        repl = REPLEnvironment(["a", "bb", "ccc"], _noop_llm_query)
        stdout, _ = repl.execute("print(len(context))")
        assert "3" in stdout

    def test_llm_query_callable(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, _ = repl.execute("result = llm_query('test prompt')\nprint(result)")
        assert "echo: test prompt" in stdout

    def test_preloaded_modules(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, _ = repl.execute("import re\nprint(re.findall(r'\\d+', 'a1b2c3'))")
        assert "['1', '2', '3']" in stdout

    def test_math_module(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, _ = repl.execute("print(math.sqrt(16))")
        assert "4.0" in stdout


class TestErrorHandling:
    def test_syntax_error_captured(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, had_error = repl.execute("if True print('oops')")
        assert had_error
        assert "SyntaxError" in stdout

    def test_runtime_error_captured(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, had_error = repl.execute("x = 1/0")
        assert had_error
        assert "ZeroDivisionError" in stdout

    def test_name_error_captured(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        stdout, had_error = repl.execute("print(undefined_var)")
        assert had_error
        assert "NameError" in stdout

    def test_error_does_not_break_subsequent_execution(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        repl.execute("x = 1/0")  # error
        stdout, had_error = repl.execute("print('still works')")
        assert not had_error
        assert "still works" in stdout


class TestTimeout:
    def test_timeout_detected(self):
        repl = REPLEnvironment("ctx", _noop_llm_query, timeout=1)
        stdout, had_error = repl.execute("import time; time.sleep(5)")
        assert had_error
        assert "timed out" in stdout


class TestNamespaceHelpers:
    def test_variable_names(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        repl.execute("my_var = 123")
        repl.execute("another = 'hello'")
        names = repl.variable_names
        assert "my_var" in names
        assert "another" in names
        assert "context" not in names
        assert "llm_query" not in names

    def test_has_variable(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        assert repl.has_variable("context")
        assert not repl.has_variable("nonexistent")
        repl.execute("x = 1")
        assert repl.has_variable("x")

    def test_get_variable(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        repl.execute("my_list = [1, 2, 3]")
        assert repl.get_variable("my_list") == [1, 2, 3]

    def test_variable_summaries(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        repl.execute("x = 42")
        summaries = repl.variable_summaries
        assert "x" in summaries
        assert "42" in summaries["x"]

    def test_long_variable_truncated_in_summary(self):
        repl = REPLEnvironment("ctx", _noop_llm_query)
        repl.execute("big = 'a' * 1000")
        summaries = repl.variable_summaries
        assert len(summaries["big"]) <= 203  # 200 + "..."


class TestNoLlmQuery:
    def test_repl_works_without_llm_query(self):
        repl = REPLEnvironment("hello world")
        stdout, had_error = repl.execute("print(len(context))")
        assert not had_error
        assert "11" in stdout

    def test_llm_query_raises_name_error(self):
        repl = REPLEnvironment("ctx")
        stdout, had_error = repl.execute("llm_query('test')")
        assert had_error
        assert "NameError" in stdout

    def test_context_still_accessible(self):
        repl = REPLEnvironment(["a", "bb", "ccc"])
        stdout, had_error = repl.execute("print(len(context))")
        assert not had_error
        assert "3" in stdout

    def test_modules_still_available(self):
        repl = REPLEnvironment("ctx")
        stdout, had_error = repl.execute("print(math.sqrt(9))")
        assert not had_error
        assert "3.0" in stdout

    def test_variable_names_exclude_llm_query(self):
        repl = REPLEnvironment("ctx")
        repl.execute("x = 42")
        assert "x" in repl.variable_names
        assert "llm_query" not in repl.variable_names
