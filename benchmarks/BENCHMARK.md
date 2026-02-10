# RLM Benchmark Suite

## Why Benchmark?

The core claim of Recursive Language Models (Zhang, Kraska & Khattab 2025) is that offloading context into a REPL environment and enabling recursive sub-LLM calls allows models to process arbitrarily long inputs — outperforming base LLMs that degrade as context grows.

We want to empirically validate this claim using our `rlm` library. Specifically, we measure:

1. **Does accuracy hold as context scales?** Base LLMs degrade with longer inputs. Does the RLM maintain performance?
2. **How does task complexity interact with context length?** Simple tasks (find a single fact) vs. complex tasks (classify every entry) should show different degradation curves.
3. **What are the cost and iteration trade-offs?** RLMs use more API calls but potentially fewer total tokens than feeding everything into one prompt.

## Benchmark Design

We replicate two benchmark categories from the paper, using synthetic data with known ground truth:

### Benchmark 1: S-NIAH (Needle-in-a-Haystack)

**Based on:** RULER (Hsieh et al 2024) — Single Needle-in-a-Haystack

**Complexity:** O(1) — the "needle" is a fixed-size fact regardless of context length.

**How it works:**
- A single fact ("The special magic number for '{key}' is: {value}") is embedded at a random position in a long haystack of diverse filler text.
- The model is asked to find and return the value.
- Context sizes: 32K, 65K, 130K, 260K, 500K, 1M characters.
- 20 tasks per size (120 total).

**What we expect:**
- **Base model:** Works at small sizes, degrades or fails as context approaches/exceeds the context window (~128K tokens for most models).
- **RLM:** Uses REPL code (e.g., `re.search()`) to find the needle in the `context` variable. Should remain accurate regardless of size.

**Scoring:** The model's response must contain the correct 7-digit number.

### Benchmark 2: Synthetic Aggregation (OOLONG-lite)

**Based on:** OOLONG trec_coarse (Bertsch et al 2025)

**Complexity:** O(n) — every entry must be processed to answer correctly.

**How it works:**
- N data entries are generated, each containing a trivia question with a hidden semantic category (entity, location, numeric_value, description, abbreviation, human_being).
- The category is NOT in the data — the model must infer it from the question's semantics.
- Tasks ask either:
  - **Comparison:** "Is label 'A' more common than label 'B'?"
  - **Count:** "How many entries have label 'A'?"
- Entry counts: 100, 500, 1000, 2000, 5000.
- 20 tasks per size (100 total).

**What we expect:**
- **Base model:** May handle 100 entries but increasingly fails at 1000+ as it cannot reliably classify and aggregate all entries within a single prompt.
- **RLM:** Chunks the data, uses sub-LLM calls to classify each chunk, then aggregates programmatically. Should scale linearly with entry count.

**Scoring:** Comparison tasks use exact match on "more/less/same". Count tasks use ±5% numeric tolerance.

## Comparison Modes

Each benchmark runs in two modes:

1. **Base mode:** Direct `chat.completions.create()` with the full context in the user message. Tests the raw LLM capability. If context exceeds the model's limit (~500K chars), the task is marked `context_exceeded`.

2. **RLM mode:** Uses `RLMWrapper.generate(query, context)`. The context is stored as a REPL variable — only metadata is shown to the root model. The model writes code to explore and process the context, optionally making recursive sub-LLM calls.

## Running

```bash
# Install benchmark dependencies
uv pip install -e ".[benchmarks]"

# Copy and fill in your API credentials
cp benchmarks/.env.example benchmarks/.env

# Run everything
python -m benchmarks.run_all

# Run a specific benchmark
python -m benchmarks.s_niah.run --mode both
python -m benchmarks.aggregation.run --mode rlm

# Run a subset of sizes (for quick testing)
python -m benchmarks.s_niah.run --mode rlm --sizes 32K 65K

# Generate plots from existing results
python -m benchmarks.run_all --plot-only
```

## Resumption

The runner saves results to JSON after every task. If a run is interrupted, re-running the same command will skip already-completed tasks and continue where it left off.

## Output

Results are saved to `benchmarks/results/`:

```
results/
├── s_niah_base.json          # 120 task results
├── s_niah_rlm.json           # 120 task results
├── aggregation_base.json     # 100 task results
├── aggregation_rlm.json      # 100 task results
└── plots/
    ├── s_niah_accuracy.png       # Accuracy vs context size
    ├── s_niah_tokens.png         # Token usage comparison
    ├── s_niah_iterations.png     # RLM iterations & sub-calls
    ├── aggregation_accuracy.png
    ├── aggregation_tokens.png
    └── aggregation_iterations.png
```

Each result JSON contains per-task metrics: predicted answer, correctness, score, iterations, sub-calls, token counts, cost, and wall-clock time.

## Metrics Collected

| Metric | Description |
| --- | --- |
| `correct` | Whether the predicted answer matches ground truth |
| `score` | 0.0 or 1.0 (exact match / tolerance) |
| `iterations` | Number of REPL loop iterations (RLM only) |
| `sub_calls` | Number of sub-LLM invocations (RLM only) |
| `total_input_tokens` | Aggregated input tokens across all LLM calls |
| `total_output_tokens` | Aggregated output tokens |
| `cost` | Estimated USD cost (if pricing configured) |
| `elapsed_seconds` | Wall-clock time per task |

## References

- Zhang, Kraska & Khattab. *Recursive Language Models.* arXiv:2512.24601, 2025.
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024.
- Bertsch et al. *Oolong: Investigating What Makes Transfer Learning Hard with Controlled Studies.* arXiv:2511.02817, 2025.
