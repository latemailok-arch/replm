"""Persistent REPL environment for RLM code execution.

The context and all intermediate variables live here — never in the LLM's
context window.  The root model writes code that is ``exec``'d in this
namespace, and ``print()`` output is captured and returned.
"""

from __future__ import annotations

import collections
import io
import json
import math
import re
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

_INJECTED_NAMES = frozenset(
    {
        "context",
        "llm_query",
        "re",
        "json",
        "math",
        "collections",
        "__builtins__",
    }
)


class REPLEnvironment:
    """A persistent Python execution environment.

    Parameters
    ----------
    context:
        The user's (potentially very long) context string or list of strings.
    llm_query_fn:
        A callable ``(prompt: str) -> str`` that invokes a sub-LLM.
    timeout:
        Per-execution timeout in seconds.
    """

    def __init__(
        self,
        context: str | list[str],
        llm_query_fn: Callable[[str], str],
        timeout: int = 120,
    ) -> None:
        self._namespace: dict[str, Any] = {
            "context": context,
            "llm_query": llm_query_fn,
            # Pre-loaded standard-library modules.
            "re": re,
            "json": json,
            "math": math,
            "collections": collections,
        }
        self._timeout = timeout

    def execute(self, code: str) -> tuple[str, bool]:
        """Execute *code* in the persistent namespace.

        Returns ``(captured_stdout, had_error)``.

        * ``print()`` output is captured via :class:`io.StringIO`.
        * Variables persist across calls.
        * Respects the configured timeout.
        * Exceptions are caught and returned as part of stdout so the model
          can self-correct.
        """
        buf = io.StringIO()
        had_error = False

        def _run() -> None:
            nonlocal had_error
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                exec(code, self._namespace)  # noqa: S102
            except Exception:
                had_error = True
                buf.write(traceback.format_exc())
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            # We cannot forcibly kill the thread, but we signal the error.
            had_error = True
            buf.write(f"\n[Execution timed out after {self._timeout}s]")

        return buf.getvalue(), had_error

    # -- Namespace helpers ---------------------------------------------------

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the REPL namespace."""
        return self._namespace[name]

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in the namespace."""
        return name in self._namespace

    @property
    def variable_names(self) -> list[str]:
        """List user-created variables (excludes builtins and injected names)."""
        return [k for k in self._namespace if k not in _INJECTED_NAMES and not k.startswith("_")]

    @property
    def variable_summaries(self) -> dict[str, str]:
        """Map user-created variable names to short ``repr`` strings."""
        out: dict[str, str] = {}
        for name in self.variable_names:
            val = self._namespace[name]
            r = repr(val)
            if len(r) > 200:
                r = r[:200] + "..."
            out[name] = r
        return out
