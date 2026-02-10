"""Parse root-model output into code blocks and FINAL directives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches fenced code blocks tagged as ``repl`` or ``python``.
_CODE_BLOCK_RE = re.compile(
    r"```(?:repl|python)\s*\n(.*?)```",
    re.DOTALL,
)

# FINAL(…) outside code fences.  Handles multiline content and nested parens
# by using a greedy match up to the *last* closing paren that keeps the
# expression balanced.
_FINAL_RE = re.compile(r"FINAL\((.+)\)\s*$", re.DOTALL)

# FINAL_VAR(variable_name) — must be a valid Python identifier.
_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_]\w*)\s*\)")


@dataclass
class ParsedResponse:
    """Result of parsing a single root-model output turn."""

    reasoning: str
    """Non-code text (the model's natural-language reasoning)."""

    code_blocks: list[str] = field(default_factory=list)
    """Extracted code from ````` repl`` or ````` python`` fences."""

    final_answer: str | None = None
    """Content of ``FINAL(...)`` if present outside code fences."""

    final_var: str | None = None
    """Variable name from ``FINAL_VAR(...)`` if present outside code fences."""

    @property
    def is_done(self) -> bool:
        return self.final_answer is not None or self.final_var is not None


def _strip_code_fences(text: str) -> str:
    """Remove all fenced code blocks so we only search for FINAL outside them."""
    return _CODE_BLOCK_RE.sub("", text)


def _extract_final(text_without_code: str) -> tuple[str | None, str | None]:
    """Return (final_answer, final_var) from text that has no code fences."""
    # Check FINAL_VAR first (more specific).
    m = _FINAL_VAR_RE.search(text_without_code)
    if m:
        return None, m.group(1)

    # Then check FINAL(...).
    m = _FINAL_RE.search(text_without_code)
    if m:
        raw = m.group(1).strip()
        # Handle balanced parens: if the captured group ends with extra ')',
        # trim trailing unbalanced close-parens.
        depth = 0
        end = len(raw)
        for i, ch in enumerate(raw):
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth > 0:
                    depth -= 1
                else:
                    end = i
                    break
        return raw[:end].strip(), None

    return None, None


def parse_response(text: str) -> ParsedResponse:
    """Parse a root-model output turn.

    Extracts:
    1. Code blocks fenced with ````` repl`` or ````` python``.
    2. ``FINAL(answer text)`` — direct final answer (only outside code fences).
    3. ``FINAL_VAR(variable_name)`` — return a REPL variable as the answer.
    """
    code_blocks = _CODE_BLOCK_RE.findall(text)

    # Strip code fences before searching for FINAL directives so that
    # ``FINAL(...)`` *inside* a code block is ignored.
    stripped = _strip_code_fences(text)
    final_answer, final_var = _extract_final(stripped)

    # Everything that isn't a code block is considered reasoning.
    reasoning = _CODE_BLOCK_RE.sub("", text).strip()

    return ParsedResponse(
        reasoning=reasoning,
        code_blocks=code_blocks,
        final_answer=final_answer,
        final_var=final_var,
    )
