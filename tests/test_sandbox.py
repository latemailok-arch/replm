"""Tests for sandbox modes (restricted, none)."""

import pytest

from rlm.config import RLMConfig
from rlm.repl import REPLEnvironment
from rlm.sandbox.restricted import ALLOWED_MODULES, build_safe_builtins

# ---------------------------------------------------------------------------
# Restricted mode — blocked operations
# ---------------------------------------------------------------------------


class TestRestrictedBlocked:
    def test_blocks_os_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import os")
        assert had_error
        assert "ImportError" in stdout

    def test_blocks_subprocess_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import subprocess")
        assert had_error
        assert "ImportError" in stdout

    def test_blocks_sys_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import sys")
        assert had_error
        assert "ImportError" in stdout

    def test_blocks_shutil_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import shutil")
        assert had_error
        assert "ImportError" in stdout

    def test_blocks_socket_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import socket")
        assert had_error
        assert "ImportError" in stdout

    def test_blocks_open(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("open('/tmp/test.txt')")
        assert had_error

    def test_blocks_eval(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("eval('1+1')")
        assert had_error

    def test_blocks_exec(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("exec('x = 1')")
        assert had_error

    def test_blocks_compile(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("compile('x = 1', '<string>', 'exec')")
        assert had_error


# ---------------------------------------------------------------------------
# Restricted mode — allowed operations
# ---------------------------------------------------------------------------


class TestRestrictedAllowed:
    def test_allows_re(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("print(re.findall(r'\\d+', 'a1b2'))")
        assert not had_error
        assert "['1', '2']" in stdout

    def test_allows_json(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("print(json.dumps({'a': 1}))")
        assert not had_error
        assert '"a"' in stdout

    def test_allows_math(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("print(math.sqrt(4))")
        assert not had_error
        assert "2.0" in stdout

    def test_allows_itertools_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute(
            "import itertools\nprint(list(itertools.chain([1], [2])))"
        )
        assert not had_error
        assert "[1, 2]" in stdout

    def test_allows_datetime_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("import datetime\nprint(type(datetime.date.today()))")
        assert not had_error
        assert "date" in stdout

    def test_allows_hashlib_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute(
            "import hashlib\nprint(hashlib.md5(b'test').hexdigest()[:8])"
        )
        assert not had_error
        assert len(stdout.strip()) == 8

    def test_allows_collections_counter(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute("print(collections.Counter('aabbc').most_common(1))")
        assert not had_error
        assert "a" in stdout

    def test_allows_functools_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute(
            "import functools\nprint(functools.reduce(lambda a, b: a+b, [1,2,3]))"
        )
        assert not had_error
        assert "6" in stdout

    def test_allows_urllib_parse_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="restricted")
        stdout, had_error = repl.execute(
            "from urllib.parse import urlparse\nprint(urlparse('http://x.com/a').path)"
        )
        assert not had_error
        assert "/a" in stdout

    def test_print_and_builtins_work(self):
        """Basic builtins like print, len, range, sorted should still work."""
        repl = REPLEnvironment("hello", sandbox_mode="restricted")
        stdout, had_error = repl.execute("print(len(context), sorted([3,1,2]))")
        assert not had_error
        assert "5" in stdout
        assert "[1, 2, 3]" in stdout


# ---------------------------------------------------------------------------
# None mode
# ---------------------------------------------------------------------------


class TestNoneMode:
    def test_allows_os_import(self):
        repl = REPLEnvironment("ctx", sandbox_mode="none")
        stdout, had_error = repl.execute("import os\nprint(os.getpid())")
        assert not had_error
        assert stdout.strip().isdigit()

    def test_allows_eval(self):
        repl = REPLEnvironment("ctx", sandbox_mode="none")
        stdout, had_error = repl.execute("print(eval('1+1'))")
        assert not had_error
        assert "2" in stdout

    def test_allows_open(self):
        repl = REPLEnvironment("ctx", sandbox_mode="none")
        stdout, had_error = repl.execute("import os; print(open.__name__)")
        assert not had_error
        assert "open" in stdout


# ---------------------------------------------------------------------------
# Default mode is restricted
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_config_is_restricted(self):
        assert RLMConfig().sandbox_mode == "restricted"

    def test_default_repl_is_restricted(self):
        repl = REPLEnvironment("ctx")
        stdout, had_error = repl.execute("import os")
        assert had_error
        assert "ImportError" in stdout

    def test_invalid_sandbox_mode_raises(self):
        with pytest.raises(ValueError, match="sandbox_mode"):
            RLMConfig(sandbox_mode="docker")


# ---------------------------------------------------------------------------
# build_safe_builtins unit tests
# ---------------------------------------------------------------------------


class TestBuildSafeBuiltins:
    def test_removes_blocked_builtins(self):
        safe = build_safe_builtins()
        assert "exec" not in safe
        assert "eval" not in safe
        assert "compile" not in safe
        assert "open" not in safe
        assert "breakpoint" not in safe

    def test_preserves_safe_builtins(self):
        safe = build_safe_builtins()
        assert "print" in safe
        assert "len" in safe
        assert "range" in safe
        assert "sorted" in safe
        assert "isinstance" in safe
        assert "str" in safe
        assert "int" in safe
        assert "list" in safe
        assert "dict" in safe

    def test_import_hook_replaces_dunder_import(self):
        safe = build_safe_builtins()
        assert "__import__" in safe
        # Should be our custom hook, not the real one
        assert safe["__import__"].__name__ == "_safe_import"


# ---------------------------------------------------------------------------
# ALLOWED_MODULES
# ---------------------------------------------------------------------------


class TestAllowedModules:
    def test_core_modules_present(self):
        for mod in ("re", "json", "math", "collections", "itertools", "datetime"):
            assert mod in ALLOWED_MODULES

    def test_dangerous_modules_absent(self):
        for mod in ("os", "subprocess", "sys", "shutil", "socket", "http"):
            assert mod not in ALLOWED_MODULES
