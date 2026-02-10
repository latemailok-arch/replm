"""Benchmark configuration — reads env vars, builds clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rlm import RLMConfig, RLMWrapper

# Load .env from benchmarks/ or examples/ directory.
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / "examples" / ".env")


@dataclass
class BenchmarkConfig:
    """All settings for a benchmark run."""

    api_key: str = field(
        default_factory=lambda: os.environ["API-KEY"],
    )
    base_url: str = field(
        default_factory=lambda: os.environ["BASE-URL"],
    )
    root_model: str = field(
        default_factory=lambda: os.environ.get(
            "ROOT-MODEL",
            os.environ.get("GPT-OSS-120b-MODEL", ""),  # noqa: SIM112
        ),
    )
    sub_model: str = field(
        default_factory=lambda: os.environ.get(
            "SUB-MODEL",
            os.environ.get("QWEN3-235b-a22b-instruct-MODEL", ""),  # noqa: SIM112
        ),
    )

    # RLM tuning.
    max_iterations: int = 25
    max_sub_calls: int = 500
    temperature: float = 0.6
    sub_temperature: float = 0.4
    verbose: bool = False

    # Retry / timeout.
    max_retries: int = 3
    retry_delay: float = 5.0
    api_timeout: float = 300.0

    # Output directory for results.
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")

    # Approximate character limit for base-mode context (~128K tokens).
    base_context_limit_chars: int = 500_000

    def make_openai_client(self) -> OpenAI:
        """Create an OpenAI client."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.api_timeout,
        )

    def make_rlm_wrapper(self) -> RLMWrapper:
        """Create a fully configured RLMWrapper."""
        return RLMWrapper(
            client=self.make_openai_client(),
            root_model=self.root_model,
            sub_model=self.sub_model or None,
            config=RLMConfig(
                max_iterations=self.max_iterations,
                max_sub_calls=self.max_sub_calls,
                temperature=self.temperature,
                sub_temperature=self.sub_temperature,
                verbose=self.verbose,
            ),
        )
