# replm = REPL + LM

**What is replm?**
replm is a lightweight Python library that wraps any OpenAI-compatible client and turns your LLM into an RLM.

**Recursive Language Models** — process arbitrarily long prompts by offloading context into a REPL and enabling symbolic recursion via sub-LLM calls.

Based on the paper [_Recursive Language Models_](https://arxiv.org/abs/2512.24601) (Zhang, Kraska & Khattab, 2025).

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
uv add replm
```

Or from source with development dependencies:

```bash
uv sync --group dev
```

**Requirements:** Python 3.10+. The `openai` package is needed for OpenAI-compatible providers; alternatively, implement the `LLMClient` protocol for any other backend.

## Quick Start

```python
from openai import OpenAI
from replm import RLMWrapper, RLMConfig

client = RLMWrapper(
    OpenAI(api_key="sk-..."),
    root_model="gpt-5.2",
)

response = client.generate(
    query="What is the main argument of this book?",
    context=very_long_text,  # string or list[str]
)

print(response.answer)            # final answer string
print(response.iterations)        # REPL loop iterations used
print(response.sub_calls)         # sub-LLM calls made
print(response.elapsed_seconds)   # wall-clock time
print(response.cost)              # USD cost (if pricing configured)
```

## Advanced Usage

### Separate root and sub-call models

Use a powerful model for orchestration and a cheaper one for sub-calls:

```python
client = RLMWrapper(
    OpenAI(api_key="sk-..."),
    root_model="gpt-5.2",
    sub_model="gpt-5-mini",
    config=RLMConfig(
        max_iterations=30,
        max_sub_calls=1000,
        verbose=True,
    ),
)
```

### Async generation

Use `agenerate()` with an async client for concurrent sub-calls via `llm_query_batch`:

```python
from openai import AsyncOpenAI
from replm import RLMWrapper

client = RLMWrapper(
    AsyncOpenAI(api_key="sk-..."),
    root_model="gpt-5.2",
    sub_model="gpt-5-mini",
)

response = await client.agenerate(
    query="Summarize all documents.",
    context=list_of_documents,
)
```

### Token-by-token streaming

Stream root model tokens as they arrive using `astream_generate()`:

```python
from openai import AsyncOpenAI
from replm import RLMWrapper

client = RLMWrapper(AsyncOpenAI(api_key="sk-..."), root_model="gpt-5.2")

async for chunk in client.astream_generate("Summarize.", very_long_text):
    if chunk.type == "token":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "final_answer":
        response = chunk.detail["response"]
        print(f"\n\nTokens: {response.total_input_tokens + response.total_output_tokens}")
```

Chunk types: `"token"`, `"iteration_start"`, `"code_executed"`, `"final_answer"`.

If the client supports native streaming (has an `astream()` method), tokens arrive in real-time. Otherwise, falls back to `acomplete()` and yields the full response as a single chunk.

### Event callbacks for observability

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

### Custom providers

For non-OpenAI backends, implement the `LLMClient` protocol directly:

```python
from replm import RLMWrapper, CompletionResult

class MyClient:
    def complete(self, model, messages, temperature, max_tokens):
        # Call your LLM here
        return CompletionResult(content="...", input_tokens=0, output_tokens=0)

client = RLMWrapper(MyClient(), root_model="my-model")
```

OpenAI SDK clients are auto-wrapped in `OpenAIAdapter` — no changes needed for existing code.

### Multi-document context

Pass a list of strings to process many documents:

```python
response = client.generate(
    query="Which documents mention climate change?",
    context=["doc 1 text...", "doc 2 text...", ...],
)
```

### Cost tracking

Configure per-token pricing to get cost estimates:

```python
config = RLMConfig(
    cost_per_input_token=2.50 / 1_000_000,
    cost_per_output_token=10.0 / 1_000_000,
)
client = RLMWrapper(OpenAI(api_key="sk-..."), root_model="gpt-5.2", config=config)
response = client.generate(query="...", context=long_text)
print(f"Cost: ${response.cost:.4f}")
```

### Sub-call caching

Avoid redundant API calls when the same sub-call prompt is issued multiple times within a single generation:

```python
config = RLMConfig(cache_sub_calls=True)
```

Cache hits are free — they don't count against the sub-call budget. The cache is per-generation (not persisted across calls) and uses LRU eviction with a 10,000-entry default.

### OpenTelemetry tracing

Install the optional tracing dependency to get automatic span instrumentation:

```bash
uv add "replm[tracing]"
```

When `opentelemetry-api` is installed, spans are emitted automatically:

- `rlm.generate` — root generation run (attributes: query length, model, iterations, tokens, elapsed time)
- `rlm.sub_call` — each sub-LLM call (attributes: depth, prompt length, tokens)

When OTel is not installed, tracing is a zero-cost no-op — no code changes needed.

### No-sub-calls ablation

Reproduce the paper's "RLM (no sub-calls)" ablation — the model uses REPL code only, no `llm_query`:

```python
config = RLMConfig(enable_sub_calls=False)
```

## Configuration

All options live in `RLMConfig`:

| Parameter                  | Default        | Description                                         |
| -------------------------- | -------------- | --------------------------------------------------- |
| `max_iterations`           | `25`           | Max REPL loop iterations for the root model         |
| `max_sub_calls`            | `500`          | Max total sub-LLM calls per generation              |
| `max_recursion_depth`      | `1`            | Nesting depth (1 = plain sub-calls, 2+ = recursive) |
| `cache_sub_calls`          | `False`        | Cache identical sub-call prompts within a run       |
| `enable_sub_calls`         | `True`         | Set `False` for the no-sub-calls ablation           |
| `metadata_prefix_chars`    | `1000`         | Characters of stdout shown to the root model        |
| `sub_call_max_input_chars` | `500000`       | Max chars per sub-call input                        |
| `temperature`              | `0.6`          | Root model temperature                              |
| `sub_temperature`          | `0.4`          | Sub-call temperature                                |
| `reasoning_effort`         | `None`         | Root model reasoning effort (`"low"`, `"medium"`, `"high"`) |
| `root_max_tokens`          | `16384`        | Max output tokens per root iteration                |
| `sub_max_tokens`           | `8192`         | Max output tokens per sub-call                      |
| `sandbox_timeout`          | `120`          | Timeout (seconds) per REPL execution                |
| `sandbox_mode`             | `"restricted"` | `"restricted"`, `"subprocess"`, or `"none"`         |
| `prompt_variant`           | `"default"`    | `"default"`, `"cost_warning"`, or `"small_context"` |
| `cost_per_input_token`     | `0.0`          | USD per input token (enables `response.cost`)       |
| `cost_per_output_token`    | `0.0`          | USD per output token (enables `response.cost`)      |
| `verbose`                  | `False`        | Print debug logs                                    |

## Response Object

`RLMResponse` contains:

- `answer` — the final answer string
- `iterations` — number of root loop iterations
- `sub_calls` — total sub-LLM invocations
- `total_input_tokens` / `total_output_tokens` — aggregated token usage
- `cache_hits` — sub-call cache hits (when `cache_sub_calls=True`)
- `cost` — estimated USD cost (based on configured per-token pricing)
- `elapsed_seconds` — wall-clock time for the generation
- `history` — full execution trace (`list[HistoryEntry]`)
- `repl_variables` — final REPL state (variable names to repr strings)

## Sandboxing

The REPL executes model-generated code, so sandboxing is on by default. Three modes are available via `sandbox_mode`:

### `"restricted"` (default)

In-process sandbox that blocks dangerous operations while allowing the standard-library modules needed for data processing:

- **Blocked:** `os`, `subprocess`, `sys`, `shutil`, `socket`, file I/O (`open`), code execution (`eval`, `exec`, `compile`), and all other non-whitelisted modules
- **Allowed:** `re`, `json`, `math`, `collections`, `itertools`, `functools`, `datetime`, `hashlib`, `csv`, `statistics`, `random`, `textwrap`, `copy`, `base64`, `urllib.parse`, and [more](src/replm/sandbox/restricted.py)

Zero overhead — runs in the same process with a restricted `__builtins__` dict and a custom import hook.

### `"subprocess"`

Full process isolation. Code runs in a child process via `multiprocessing`:

- Real timeout enforcement — `process.kill()` terminates stuck code
- Auto-recovery — a new child is spawned after a timeout, with user variables restored
- `llm_query` and `llm_query_batch` are proxied back to the parent through IPC
- Restricted builtins are also applied inside the child

```python
config = RLMConfig(sandbox_mode="subprocess", sandbox_timeout=30)
```

### `"none"`

No restrictions. Code runs with full access to the Python runtime. Use only in trusted environments or when you need access to blocked modules.

```python
config = RLMConfig(sandbox_mode="none")
```

## Architecture

```
src/replm/
├── __init__.py            # Public API
├── wrapper.py             # RLMWrapper — main entry point
├── client.py              # LLMClient protocol + OpenAIAdapter
├── orchestrator.py        # Root REPL loop (Algorithm 1)
├── async_orchestrator.py  # Async variant with concurrent sub-calls
├── stream.py              # StreamOrchestrator + StreamChunk (token streaming)
├── repl.py                # REPL environment: exec, variables
├── sub_caller.py          # Sub-LLM call manager (sync)
├── async_sub_caller.py    # Sub-LLM call manager (async)
├── budget.py              # SharedBudget for global sub-call limits
├── cache.py               # LRU cache for sub-call responses
├── tracing.py             # OpenTelemetry spans (no-op when OTel absent)
├── parser.py              # Parse code blocks + FINAL directives
├── prompt.py              # System prompt templates (Appendix C.1)
├── metadata.py            # Truncation logic
├── config.py              # RLMConfig dataclass
├── types.py               # RLMResponse, RLMEvent, HistoryEntry
├── exceptions.py          # RLMError hierarchy
└── sandbox/
    ├── __init__.py            # Sandbox public API
    ├── restricted.py          # Safe builtins + import whitelist
    └── subprocess_executor.py # Child process with IPC
```

## Development

```bash
git clone https://github.com/dschulmeist/replm.git
cd replm
uv sync --group dev
uv run pytest
```

### Running tests

```bash
uv run pytest                        # all tests
uv run pytest tests/test_parser.py   # specific module
uv run pytest -v --tb=short          # verbose with short tracebacks
```

### Linting

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Roadmap

- External sandboxing backends (Docker, E2B)

## Citations

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
