"""rlm — Recursive Language Models for processing arbitrarily long prompts.

Turn any OpenAI-compatible client into a Recursive Language Model that
offloads context into a REPL and enables symbolic recursion via sub-LLM calls.

Basic usage::

    from openai import OpenAI
    from rlm import RLMWrapper, RLMConfig

    client = RLMWrapper(OpenAI(api_key="sk-..."), root_model="gpt-4.1")
    response = client.generate("Summarize this.", very_long_text)
    print(response.answer)
"""

from .async_orchestrator import AsyncOrchestrator
from .budget import SharedBudget
from .config import RLMConfig
from .exceptions import MaxSubCallsExceeded, RLMError
from .types import HistoryEntry, RLMEvent, RLMResponse
from .wrapper import RLMWrapper

__all__ = [
    "RLMWrapper",
    "RLMConfig",
    "RLMResponse",
    "RLMEvent",
    "HistoryEntry",
    "SharedBudget",
    "AsyncOrchestrator",
    "RLMError",
    "MaxSubCallsExceeded",
]

__version__ = "0.1.0"
