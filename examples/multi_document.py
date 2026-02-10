"""Example: process a list of documents with RLM.

Demonstrates passing ``context`` as a list of strings, where each
element represents a separate document.

Requires a .env file in this directory (see .env.example).

    python examples/multi_document.py
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rlm import RLMConfig, RLMWrapper

load_dotenv(Path(__file__).with_name(".env"))

client = RLMWrapper(
    OpenAI(),
    root_model="gpt-4.1-mini",
    sub_model="gpt-4.1-mini",
    config=RLMConfig(max_iterations=15, verbose=True),
)

# Simulate a list of documents.
documents = [
    f"Document {i}: {'Lorem ipsum dolor sit amet. ' * 200}" for i in range(50)
]
# Hide a fact in document 37.
documents[37] = (
    "Document 37: The annual revenue of Acme Corp in 2024 was $4.2 billion. "
    + "This was driven primarily by growth in the cloud services division. " * 100
)

response = client.generate(
    query="What was the annual revenue of Acme Corp in 2024?",
    context=documents,
    on_event=lambda e: print(f"  [{e.type}] {e.preview[:80]}"),
)

print(f"\nAnswer: {response.answer}")
print(f"Iterations: {response.iterations}")
print(f"Sub-calls: {response.sub_calls}")
