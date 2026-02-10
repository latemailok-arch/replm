"""Restricted sandbox — safe builtins and import whitelist.

Provides an in-process sandbox that blocks dangerous operations (file I/O,
networking, subprocess spawning) while allowing the standard-library modules
needed for data processing.  This is the default sandbox mode.
"""

from __future__ import annotations

import builtins
from typing import Any

# Modules safe for data processing — no filesystem, network, or process access.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "re",
        "json",
        "math",
        "collections",
        "itertools",
        "functools",
        "string",
        "textwrap",
        "statistics",
        "decimal",
        "fractions",
        "operator",
        "copy",
        "dataclasses",
        "enum",
        "typing",
        "hashlib",
        "base64",
        "struct",
        "csv",
        "difflib",
        "heapq",
        "bisect",
        "random",
        "datetime",
        "pprint",
        "time",
        "unicodedata",
        "html",
        "urllib",
    }
)

# Builtins that allow code execution, file access, or interpreter control.
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "breakpoint",
        "exit",
        "quit",
        "input",
    }
)

_real_import = builtins.__import__


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,  # noqa: A002
    locals: dict[str, Any] | None = None,  # noqa: A002
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Import hook that only allows modules in :data:`ALLOWED_MODULES`."""
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_MODULES:
        raise ImportError(
            f"Module {name!r} is not allowed in restricted sandbox mode. "
            f"Allowed top-level packages: {sorted(ALLOWED_MODULES)}"
        )
    return _real_import(name, globals, locals, fromlist, level)


def build_safe_builtins() -> dict[str, Any]:
    """Return a ``__builtins__`` dict with dangerous entries removed.

    The returned dict replaces ``__import__`` with :func:`_safe_import`
    so only :data:`ALLOWED_MODULES` can be imported.
    """
    # __builtins__ can be either a dict or the builtins module itself.
    src = builtins.__dict__
    safe = {k: v for k, v in src.items() if k not in BLOCKED_BUILTINS}
    safe["__import__"] = _safe_import
    return safe
