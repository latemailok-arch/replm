"""Subprocess sandbox — runs REPL code in an isolated child process.

Provides real timeout enforcement via ``process.kill()`` and process-level
isolation.  Sub-call functions (``llm_query``, ``llm_query_batch``) are
proxied back to the parent through a ``multiprocessing.Pipe`` IPC channel.

IPC Protocol (tuples over ``multiprocessing.Connection``):

    Parent → Child:
        ("exec", code_str)              — execute code in the namespace
        ("llm_result", rid, answer)     — response to an llm_query request
        ("llm_batch_result", rid, list) — response to an llm_query_batch request

    Child → Parent:
        ("done", stdout, had_error, updated_vars)  — execution finished
        ("llm_query", rid, prompt)                  — sub-call request
        ("llm_query_batch", rid, prompts)           — batch sub-call request
"""

from __future__ import annotations

import io
import multiprocessing
import pickle
import sys
import traceback
import uuid
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from .restricted import build_safe_builtins


def _child_main(conn: Connection, initial_namespace: dict[str, Any]) -> None:  # noqa: C901
    """Entry point for the child process.  Waits for exec commands."""
    import collections
    import json
    import math
    import re

    namespace: dict[str, Any] = {
        **initial_namespace,
        "re": re,
        "json": json,
        "math": math,
        "collections": collections,
        "__builtins__": build_safe_builtins(),
    }

    # Inject llm_query / llm_query_batch stubs that send IPC messages.
    if initial_namespace.get("_has_llm_query"):

        def llm_query(prompt: str) -> str:
            rid = uuid.uuid4().hex[:8]
            conn.send(("llm_query", rid, prompt))
            # Block until parent responds.
            while True:
                msg = conn.recv()
                if msg[0] == "llm_result" and msg[1] == rid:
                    return msg[2]

        namespace["llm_query"] = llm_query

    if initial_namespace.get("_has_llm_query_batch"):

        def llm_query_batch(prompts: list[str]) -> list[str]:
            rid = uuid.uuid4().hex[:8]
            conn.send(("llm_query_batch", rid, prompts))
            while True:
                msg = conn.recv()
                if msg[0] == "llm_batch_result" and msg[1] == rid:
                    return msg[2]

        namespace["llm_query_batch"] = llm_query_batch

    # Remove internal flags from namespace.
    namespace.pop("_has_llm_query", None)
    namespace.pop("_has_llm_query_batch", None)

    # Main command loop.
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        if msg[0] != "exec":
            continue

        code = msg[1]
        buf = io.StringIO()
        had_error = False
        old_stdout = sys.stdout

        try:
            sys.stdout = buf
            exec(code, namespace)  # noqa: S102
        except Exception:
            had_error = True
            buf.write(traceback.format_exc())
        finally:
            sys.stdout = old_stdout

        # Collect user-created variables (serialisable only).
        injected = frozenset(
            {
                "context",
                "llm_query",
                "llm_query_batch",
                "re",
                "json",
                "math",
                "collections",
                "__builtins__",
            }
        )
        updated_vars: dict[str, Any] = {}
        for k, v in namespace.items():
            if k in injected or k.startswith("_"):
                continue
            try:
                pickle.dumps(v)
                updated_vars[k] = v
            except (pickle.PicklingError, TypeError, AttributeError):
                pass

        conn.send(("done", buf.getvalue(), had_error, updated_vars))


class SubprocessExecutor:
    """Runs code in a child process with IPC-based sub-call proxying.

    Parameters
    ----------
    timeout:
        Seconds to wait for each ``execute()`` call before killing the child.
    """

    def __init__(self, timeout: int = 120) -> None:
        self._timeout = timeout
        self._process: multiprocessing.Process | None = None
        self._conn: Connection | None = None

    def start(
        self,
        context: str | list[str],
        has_llm_query: bool = False,
        has_llm_query_batch: bool = False,
    ) -> None:
        """Spawn the child process with the initial namespace."""
        parent_conn, child_conn = multiprocessing.Pipe()
        initial_ns: dict[str, Any] = {
            "context": context,
            "_has_llm_query": has_llm_query,
            "_has_llm_query_batch": has_llm_query_batch,
        }
        self._process = multiprocessing.Process(
            target=_child_main,
            args=(child_conn, initial_ns),
            daemon=True,
        )
        self._process.start()
        child_conn.close()  # Parent doesn't use the child end.
        self._conn = parent_conn

    def execute(
        self,
        code: str,
        llm_query_fn: Callable[[str], str] | None = None,
        llm_query_batch_fn: Callable[[list[str]], list[str]] | None = None,
    ) -> tuple[str, bool, dict[str, Any]]:
        """Execute *code* in the child process.

        Returns ``(stdout, had_error, updated_vars)``.

        Proxies ``llm_query`` / ``llm_query_batch`` calls from the child back
        to the parent via IPC.  Kills the child on timeout.
        """
        conn = self._conn
        if conn is None or self._process is None or not self._process.is_alive():
            raise RuntimeError("Subprocess not started — call start() first")

        conn.send(("exec", code))

        # Message loop: wait for "done" while serving sub-call requests.
        while True:
            if not conn.poll(self._timeout):
                # Timeout — kill the child.
                self._process.kill()
                self._process.join(timeout=5)
                self._process = None
                self._conn = None
                return (f"\n[Execution timed out after {self._timeout}s]", True, {})

            msg = conn.recv()
            kind = msg[0]

            if kind == "done":
                return (msg[1], msg[2], msg[3])

            if kind == "llm_query" and llm_query_fn is not None:
                rid, prompt = msg[1], msg[2]
                try:
                    result = llm_query_fn(prompt)
                except Exception as exc:
                    result = f"[llm_query error: {exc}]"
                conn.send(("llm_result", rid, result))

            elif kind == "llm_query_batch" and llm_query_batch_fn is not None:
                rid, prompts = msg[1], msg[2]
                try:
                    results = llm_query_batch_fn(prompts)
                except Exception as exc:
                    results = [f"[llm_query_batch error: {exc}]"] * len(prompts)
                conn.send(("llm_batch_result", rid, results))

    def stop(self) -> None:
        """Kill the child process and clean up."""
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)
        self._process = None
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def is_alive(self) -> bool:
        """Check if the child process is running."""
        return self._process is not None and self._process.is_alive()
