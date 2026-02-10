"""OpenTelemetry instrumentation — auto-detected, no-op when OTel is absent.

When ``opentelemetry-api`` is installed, spans are emitted for:

- ``rlm.generate`` — root generation run
- ``rlm.iteration`` — each root-loop iteration
- ``rlm.sub_call`` — each sub-LLM call

When OTel is *not* installed, the ``span()`` context manager is a zero-cost
no-op so callers never need to check.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace

    _tracer: trace.Tracer | None = trace.get_tracer("rlm")
except Exception:  # ImportError, or OTel not configured
    _tracer = None


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that creates an OTel span if a tracer is available.

    When ``opentelemetry-api`` is not installed, this is a zero-cost no-op.

    Parameters
    ----------
    name:
        Span name (e.g. ``"rlm.generate"``).
    attributes:
        Initial span attributes.

    Yields
    ------
    The active span (or ``None`` if OTel is not available).  Callers can
    set additional attributes on the yielded span::

        with span("rlm.generate", {"model": "gpt-4.1"}) as s:
            ...
            if s is not None:
                s.set_attribute("rlm.iterations", 5)
    """
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as s:
        if attributes:
            for key, value in attributes.items():
                s.set_attribute(key, value)
        yield s
