"""Basic example: use RLM to analyze a long text.

Requires an OpenAI API key in the OPENAI_API_KEY environment variable.

    OPENAI_API_KEY=sk-... python examples/basic_usage.py
"""

import os

from openai import OpenAI

from rlm import RLMConfig, RLMWrapper

client = RLMWrapper(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    root_model="gpt-4.1-mini",
    sub_model="gpt-4.1-mini",
    config=RLMConfig(verbose=True),
)

# Generate a synthetic long context for demonstration.
long_text = (
    "The quick brown fox jumps over the lazy dog. " * 5000
    + "SECRET: The magic number is 7. "
    + "The quick brown fox jumps over the lazy dog. " * 5000
)

response = client.generate(
    query="What is the magic number hidden in the text?",
    context=long_text,
    on_event=lambda e: print(f"  [{e.type}] {e.preview[:80]}"),
)

print(f"\nAnswer: {response.answer}")
print(f"Iterations: {response.iterations}")
print(f"Sub-calls: {response.sub_calls}")
print(f"Tokens (in/out): {response.total_input_tokens}/{response.total_output_tokens}")
