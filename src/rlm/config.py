"""Configuration for the RLM library."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RLMConfig:
    """Configuration for an RLM generation run.

    All fields have sensible defaults. Override only what you need.
    """

    max_iterations: int = 25
    """Max REPL loop iterations for the root model."""

    max_sub_calls: int = 500
    """Max total sub-LLM calls per generation."""

    max_recursion_depth: int = 1
    """Nesting depth (1 = sub-calls are plain LLM, no further recursion).

    .. note:: Reserved for v0.2.  Currently only depth=1 is supported.
       Values > 1 are accepted but have no effect yet.
    """

    enable_sub_calls: bool = True
    """Whether to enable sub-LLM calls (``llm_query``) in the REPL.

    When ``False``, uses a simplified system prompt without sub-call
    examples, and does not inject ``llm_query`` into the REPL environment.
    Set to ``False`` to reproduce the "RLM (no sub-calls)" ablation from
    the paper.
    """

    metadata_prefix_chars: int = 1000
    """Characters of stdout / content shown to the root model."""

    sub_call_max_input_chars: int = 500_000
    """Approximate max characters per sub-call input."""

    temperature: float = 0.6
    """Root model temperature."""

    sub_temperature: float = 0.4
    """Sub-call temperature."""

    root_max_tokens: int = 16_384
    """Max output tokens per root iteration."""

    sub_max_tokens: int = 8_192
    """Max output tokens per sub-call."""

    sandbox_timeout: int = 120
    """Timeout (seconds) per REPL execution."""

    prompt_variant: str = "default"
    """Prompt variant controlling model-specific system prompt behaviour.

    Supported values:

    - ``"default"``: Standard prompt (best for GPT-5, GPT-4.1, Claude, etc.)
    - ``"cost_warning"``: Adds a warning about sub-call costs and batching
      guidance.  Best for models that tend to make excessive sub-calls
      (e.g. Qwen3-Coder).
    - ``"small_context"``: Reduces context-limit descriptions to ~32k tokens,
      uses smaller chunk sizes in examples, and adds token-limit warnings.
      Best for small models (e.g. Qwen3-8B).
    """

    verbose: bool = False
    """Print debug logs to stderr."""

    cost_per_input_token: float = 0.0
    """Cost per input token in USD.  Set this to enable ``response.cost``.

    Example: for a model charging $2.50 / 1M input tokens, set to ``2.50 / 1_000_000``.
    """

    cost_per_output_token: float = 0.0
    """Cost per output token in USD.  Set this to enable ``response.cost``.

    Example: for a model charging $10 / 1M output tokens, set to ``10.0 / 1_000_000``.
    """

    _extras: dict[str, object] = field(default_factory=dict, repr=False)
    """Reserved for future extensions without breaking the API."""

    _VALID_PROMPT_VARIANTS: frozenset[str] = field(
        default=frozenset({"default", "cost_warning", "small_context"}),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.prompt_variant not in self._VALID_PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant {self.prompt_variant!r}. "
                f"Must be one of {sorted(self._VALID_PROMPT_VARIANTS)}"
            )
