# rlm

**Recursive Language Models** — process arbitrarily long prompts by offloading context into a REPL and enabling symbolic recursion via sub-LLM calls.

Based on the paper [*Recursive Language Models*](https://arxiv.org/abs/2512.24601) (Zhang, Kraska & Khattab, 2025).

---

## What is an RLM?

Standard LLMs break down when prompts exceed their context window, and quality degrades well before the hard limit (**context rot**). An RLM fixes this by treating the prompt as a variable in a persistent REPL environment rather than feeding it into the model's token budget. The model writes code to peek at, decompose, and recursively call itself over slices of the context.

```
            User prompt (arbitrarily long)
                     │
                     ▼
         ┌───────────────────────┐
         │    REPL Environment   │
         │   context = <prompt>  │◄── model writes code here
         │   llm_query(...)      │    (peek, chunk, sub-call)
         └───────┬───────────────┘
                 │
            only metadata
            (length, prefix)
                 │
                 ▼
         ┌───────────────────────┐
         │      Root LLM         │
         │  generates code +     │
         │  reasoning each turn  │
         └───────────────────────┘
```

Key properties:
- **The full prompt never enters the LLM's context window.** Only metadata (length, a short prefix) does.
- **Stdout is truncated** before being shown to the root model, forcing it to use variables and sub-calls.
- **Symbolic recursion:** the `llm_query()` function is callable inside REPL code, so the model can launch O(|P|) or even O(|P|^2) sub-processes over programmatic slices of the input.

## Installation

```bash
pip install rlm
```

Or with development dependencies:

```bash
pip install rlm[dev]
```

**Requirements:** Python 3.10+ and an OpenAI-compatible client (`openai` package).

## Quick Start

```python
from openai import OpenAI
from rlm import RLMWrapper, RLMConfig

client = RLMWrapper(
    OpenAI(api_key="sk-..."),
    root_model="gpt-4.1",
)

response = client.generate(
    query="What is the main argument of this book?",
    context=very_long_text,  # string or list[str]
)

print(response.answer)       # final answer string
print(response.iterations)   # REPL loop iterations used
print(response.sub_calls)    # sub-LLM calls made
print(response.total_input_tokens)
print(response.total_output_tokens)
```

## Advanced Usage

### Separate root and sub-call models

Use a powerful model for orchestration and a cheaper one for sub-calls:

```python
client = RLMWrapper(
    OpenAI(api_key="sk-..."),
    root_model="gpt-4.1",
    sub_model="gpt-4.1-mini",
    config=RLMConfig(
        max_iterations=30,
        max_sub_calls=1000,
        verbose=True,
    ),
)
```

### Streaming events for observability

```python
def on_event(event):
    print(f"[iter {event.iteration}] {event.type}: {event.preview[:80]}")

response = client.generate(
    query="Find all entities mentioned in these documents.",
    context=list_of_documents,
    on_event=on_event,
)
```

### OpenAI-compatible providers

Works with any provider that exposes the OpenAI chat completions API:

```python
# Together AI
client = RLMWrapper(
    OpenAI(api_key="...", base_url="https://api.together.xyz/v1"),
    root_model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
)

# Fireworks
client = RLMWrapper(
    OpenAI(api_key="fw-...", base_url="https://api.fireworks.ai/inference/v1"),
    root_model="accounts/fireworks/models/qwen3-coder-480b-a35b",
)
```

### Multi-document context

Pass a list of strings to process many documents:

```python
response = client.generate(
    query="Which documents mention climate change?",
    context=["doc 1 text...", "doc 2 text...", ...],
)
```

## Configuration

All options live in `RLMConfig`:

| Parameter | Default | Description |
|---|---|---|
| `max_iterations` | `25` | Max REPL loop iterations for the root model |
| `max_sub_calls` | `500` | Max total sub-LLM calls per generation |
| `max_recursion_depth` | `1` | Nesting depth (1 = sub-calls are plain LLM) |
| `metadata_prefix_chars` | `1000` | Characters of stdout shown to the root model |
| `sub_call_max_input_chars` | `500000` | Max chars per sub-call input |
| `temperature` | `0.6` | Root model temperature |
| `sub_temperature` | `0.4` | Sub-call temperature |
| `root_max_tokens` | `16384` | Max output tokens per root iteration |
| `sub_max_tokens` | `8192` | Max output tokens per sub-call |
| `sandbox_timeout` | `120` | Timeout (seconds) per REPL execution |
| `verbose` | `False` | Print debug logs |

## Response Object

`RLMResponse` contains:

- `answer` — the final answer string
- `iterations` — number of root loop iterations
- `sub_calls` — total sub-LLM invocations
- `total_input_tokens` / `total_output_tokens` — aggregated token usage
- `history` — full execution trace (`list[HistoryEntry]`)
- `repl_variables` — final REPL state (variable names to repr strings)

## Architecture

```
src/rlm/
├── __init__.py         # Public API
├── wrapper.py          # RLMWrapper — main entry point
├── orchestrator.py     # Root REPL loop (Algorithm 1)
├── repl.py             # REPL environment: exec, variables
├── sub_caller.py       # Sub-LLM call manager
├── parser.py           # Parse code blocks + FINAL directives
├── prompt.py           # System prompt templates
├── metadata.py         # Truncation logic
├── config.py           # RLMConfig dataclass
├── types.py            # RLMResponse, RLMEvent, HistoryEntry
├── exceptions.py       # RLMError hierarchy
└── sandbox/            # Future: pluggable sandboxing backends
```

## Development

```bash
git clone https://github.com/replm/rlm.git
cd rlm
pip install -e ".[dev]"
pytest
```

### Running tests

```bash
pytest                     # all tests
pytest tests/test_parser.py   # specific module
pytest -v --tb=short       # verbose with short tracebacks
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Security Note

The REPL executes model-generated code via `exec()` in a restricted namespace. For production use, consider running the REPL in a sandboxed environment (Docker, E2B, etc.). The `sandbox/` module is a placeholder for pluggable backends.

## Roadmap

- Async support with parallel sub-calls
- Pluggable sandboxing backends (Docker, E2B)
- Streaming of root model output
- Caching of sub-call results
- Deeper recursion (sub-RLMs instead of plain sub-LLMs)
- Per-model prompt tuning

## Citation

```bibtex
@article{zhang2025rlm,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025}
}
```

## License

MIT
