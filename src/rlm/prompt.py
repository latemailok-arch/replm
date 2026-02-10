"""System prompt templates for the root RLM model.

Based on Appendix C.1 of the RLM paper (Zhang, Kraska, Khattab 2025).
Supports three prompt variants:

- ``"default"`` — standard prompt (GPT-5, GPT-4.1, Claude, etc.)
- ``"cost_warning"`` — adds sub-call cost guidance (Qwen3-Coder)
- ``"small_context"`` — reduces limits to ~32k tokens (Qwen3-8B)
"""

from __future__ import annotations

from .config import RLMConfig


def build_root_system_prompt(
    context_type: str,
    context_total_length: int,
    context_lengths: list[int],
    config: RLMConfig,
    *,
    include_batch_fn: bool = False,
) -> str:
    """Build the system prompt for the root model.

    Parameters correspond to metadata about the user's context — the actual
    content never enters the prompt.  Delegates to the no-sub-calls prompt
    when ``config.enable_sub_calls`` is ``False``.
    """
    if not config.enable_sub_calls:
        return _build_no_subcalls_prompt(
            context_type, context_total_length, context_lengths, config
        )

    variant = config.prompt_variant
    lengths_str = str(context_lengths)

    # --- Variant-dependent values -------------------------------------------
    if variant == "small_context":
        capacity_str = "~100k chars, roughly 32k tokens"
        batching_advice = (
            "For example, a viable strategy is to feed 2-3 documents per sub-LLM query."
        )
        chunk_peek = "context[:1000]"
    else:
        capacity_str = f"{config.sub_call_max_input_chars:,}"
        batching_advice = (
            "For example, a viable strategy is to feed 10 documents per sub-LLM query."
        )
        chunk_peek = "context[:10000]"

    # --- Assemble sections --------------------------------------------------
    parts: list[str] = []

    # Opening paragraph
    parts.append(
        "You are tasked with answering a query with associated context. "
        "You can access, transform, and analyze this context interactively in a REPL "
        "environment that can recursively query sub-LLMs, which you are strongly "
        "encouraged to use as much as possible. You will be queried iteratively until "
        "you provide a final answer."
    )

    # Small-context window warning (paper diff 1c)
    if variant == "small_context":
        parts.append(
            "IMPORTANT: You have a total context window of approximately ~32k tokens. "
            "Be very careful about context length limits. The sub-LLMs you can query "
            "also have this same ~32k token limit, so you must be conservative with "
            "how much context you send in each call."
        )

    # Context metadata
    parts.append(
        f"Your context is a {context_type} with {context_total_length:,} total "
        f"characters, and is broken up into chunks of char lengths: {lengths_str}."
    )

    # REPL initialization
    repl_init = (
        "The REPL environment is initialized with:\n\n"
        "1. A `context` variable that contains extremely important information about "
        "your query. You should check the content of the `context` variable to "
        "understand what you are working with. Make sure you look through it "
        "sufficiently as you answer your query.\n\n"
        f"2. A `llm_query` function that allows you to query an LLM (that can handle "
        f"around {capacity_str} chars) inside your REPL environment."
    )
    if include_batch_fn:
        repl_init += (
            " You also have `llm_query_batch(prompts: list[str]) -> list[str]` for "
            "running multiple queries in parallel — prefer this when making many "
            "independent sub-calls."
        )
    repl_init += (
        "\n\n"
        "3. The ability to use `print()` statements to view the output of your REPL "
        "code and continue your reasoning."
    )
    parts.append(repl_init)

    # Cost warning (paper diff 1b — Qwen3-Coder)
    if variant == "cost_warning":
        parts.append(
            "IMPORTANT: Be very careful about using 'llm_query' as it incurs high "
            "runtime costs. Always batch as much information as reasonably possible "
            "into each call (aim for around ~200k characters per call). For example, "
            "if you have 1000 lines of information to process, it's much better to "
            "split into chunks of 5 and call 'llm_query' on each chunk (200 calls "
            "total) rather than making 1000 individual calls. Minimize the number of "
            "'llm_query' calls by batching related information together."
        )
    elif variant == "small_context":
        parts.append(
            "IMPORTANT: Be very careful about using 'llm_query' as it incurs high "
            "runtime costs. Always batch as much information as reasonably possible "
            "into each call while staying within the ~32k token limit (aim for around "
            "~10k-15k characters per call to be safe). For example, if you have 1000 "
            "lines of information to process, it's much better to split into chunks "
            "of 50-100 and call 'llm_query' on each chunk (10-20 calls total) rather "
            "than making 1000 individual calls. Minimize the number of 'llm_query' "
            "calls by batching related information together, but always respect the "
            "~32k token limit."
        )

    # Truncation warning + strategy guidance
    parts.append(
        "You will only be able to see truncated outputs from the REPL environment, so "
        "you should use the query LLM function on variables you want to analyze. You "
        "will find this function especially useful when you have to analyze the "
        "semantics of the context. Use these variables as buffers to build up your "
        "final answer."
    )

    parts.append(
        "Make sure to explicitly look through the entire context in REPL before "
        "answering your query. An example strategy is to first look at the context and "
        "figure out a chunking strategy, then break up the context into smart chunks, "
        "and query an LLM per chunk with a particular question and save the answers to "
        "a buffer, then query an LLM with all the buffers to produce your final answer."
    )

    # Batching guidance
    if variant == "small_context":
        parts.append(
            "You can use the REPL environment to help you understand your context, "
            "especially if it is huge. Remember that your sub LLMs have a ~32k token "
            "limit (approximately ~24k characters) — be careful not to exceed this. "
            f"{batching_advice} Analyze your input data and see if it is sufficient "
            "to just fit it in a few sub-LLM calls!"
        )
    else:
        parts.append(
            "You can use the REPL environment to help you understand your context, "
            f"especially if it is huge. Remember that your sub LLMs are powerful — "
            f"they can fit around {capacity_str} characters in their context window, "
            f"so don't be afraid to put a lot of context into them. {batching_advice} "
            "Analyze your input data and see if it is sufficient to just fit it in a "
            "few sub-LLM calls!"
        )

    # Code examples
    parts.append(
        "When you want to execute Python code in the REPL environment, wrap it in "
        "triple backticks with the `repl` language identifier. For example:\n\n"
        f"```repl\n"
        f"chunk = {chunk_peek}\n"
        f'answer = llm_query(f"What is the magic number in the context? '
        f'Here is the chunk: {{chunk}}")\n'
        f"print(answer)\n"
        f"```"
    )

    parts.append(
        "As an example, suppose you're trying to answer a question about a book. You "
        "can iteratively chunk the context section by section, query an LLM on that "
        "chunk, and track relevant information in a buffer:\n\n"
        "```repl\n"
        'query = "In Harry Potter, did Gryffindor win the House Cup because they led?"\n'
        "for i, section in enumerate(context):\n"
        "    if i == len(context) - 1:\n"
        "        buffer = llm_query(\n"
        '            f"You are on the last section. So far: {buffers}. "\n'
        '            f"Answer {query}. Section: {section}"\n'
        "        )\n"
        "    else:\n"
        "        buffer = llm_query(\n"
        '            f"Section {i} of {len(context)}. "\n'
        '            f"Gather info for {query}. Section: {section}"\n'
        "        )\n"
        '    print(f"After section {i}: {buffer}")\n'
        "```"
    )

    parts.append(
        "As another example, when the context is moderately sized, a simple strategy is "
        "to combine chunks and recursively query an LLM over them:\n\n"
        "```repl\n"
        'query = "How many jobs did Fitzgerald have?"\n'
        "chunk_size = len(context) // 10\n"
        "answers = []\n"
        "for i in range(10):\n"
        "    start = i * chunk_size\n"
        "    end = (i + 1) * chunk_size if i < 9 else len(context)\n"
        "    chunk_str = context[start:end] if isinstance(context, str)"
        ' else "\\n".join(context[start:end])\n'
        "    answer = llm_query(\n"
        '        f"Try to answer: {query}\\nDocuments:\\n{chunk_str}\\n"\n'
        '        f"Only answer if confident."\n'
        "    )\n"
        "    answers.append(answer)\n"
        '    print(f"Chunk {i}: {answer}")\n'
        "final_answer = llm_query(\n"
        '    f"Aggregate answers for: {query}\\n\\n" + "\\n".join(answers)\n'
        ")\n"
        "```"
    )

    parts.append(
        "As a final example, after analyzing the context and realizing it is separated "
        "by Markdown headers, you can maintain state through buffers by chunking the "
        "context by headers and iteratively querying an LLM over each section:\n\n"
        "```repl\n"
        "import re\n"
        "sections = re.split(r'### (.+)', context)\n"
        "buffers = []\n"
        "for i in range(1, len(sections), 2):\n"
        "    header = sections[i]\n"
        "    info = sections[i + 1]\n"
        '    summary = llm_query(f"Summarize this {header} section: {info}")\n'
        '    buffers.append(f"{header}: {summary}")\n'
        "final_answer = llm_query(\n"
        '    f"Based on these summaries, answer the original query: {query}\\n\\nSummaries:\\n"\n'
        '    + "\\n".join(buffers)\n'
        ")\n"
        "```\n\n"
        "In the next step, we can return FINAL_VAR(final_answer)."
    )

    # FINAL instructions
    final_instruction = (
        "IMPORTANT: When you are done with the iterative process, you MUST provide a "
        "final answer inside a FINAL function when you have completed your task, NOT "
        "in code"
    )
    if variant == "small_context":
        final_instruction += " or repl tags"
    final_instruction += (
        ". Do not use these tags unless you have completed your task. You have "
        "two options:\n\n"
        "1. Use FINAL(your final answer here) to provide the answer directly\n"
        "2. Use FINAL_VAR(variable_name) to return a variable you have created in the "
        "REPL environment as your final output"
    )
    parts.append(final_instruction)

    parts.append(
        "Think step by step carefully, plan, and execute this plan immediately in your "
        'response — do not just say "I will do this". Output to the REPL environment '
        "and recursive LLMs as much as possible. Remember to explicitly answer the "
        "original query in your final answer."
    )

    return "\n\n".join(parts)


def _build_no_subcalls_prompt(
    context_type: str,
    context_total_length: int,
    context_lengths: list[int],
    config: RLMConfig,
) -> str:
    """Build the system prompt for the no-sub-calls ablation.

    Based on paper Appendix C.1 prompt (2): REPL without ``llm_query``.
    The model can only use code execution (regex, string ops, etc.) to
    interact with the context.
    """
    lengths_str = str(context_lengths)

    parts: list[str] = []

    parts.append(
        "You are tasked with answering a query with associated context. "
        "You can access, transform, and analyze this context interactively in a REPL "
        "environment, which you are strongly encouraged to use as much as possible. "
        "You will be queried iteratively until you provide a final answer."
    )

    parts.append(
        f"Your context is a {context_type} with {context_total_length:,} total "
        f"characters, and is broken up into chunks of char lengths: {lengths_str}."
    )

    parts.append(
        "The REPL environment is initialized with:\n\n"
        "1. A `context` variable that contains extremely important information about "
        "your query. You should check the content of the `context` variable to "
        "understand what you are working with. Make sure you look through it "
        "sufficiently as you answer your query.\n\n"
        "2. The ability to use `print()` statements to view the output of your REPL "
        "code and continue your reasoning."
    )

    parts.append(
        "You will only be able to see truncated outputs from the REPL environment "
        "to not overflow the context window. Use these variables as buffers to build "
        "up your final answer."
    )

    parts.append(
        "Make sure to explicitly look through the entire context in REPL before "
        "answering your query. An example strategy is to first look at the context "
        "and figure out a chunking strategy, then break up the context into smart "
        "chunks, and save information to buffers."
    )

    parts.append(
        "You can use the REPL environment to help you understand your context, "
        "especially if it is huge."
    )

    parts.append(
        "When you want to execute Python code in the REPL environment, wrap it in "
        "triple backticks with the `repl` language identifier. For example, "
        "say we want to peek at the first 10000 characters of the context:\n\n"
        "```repl\n"
        "chunk = context[:10000]\n"
        'print(f"First 10000 characters of context: {chunk}")\n'
        "```"
    )

    parts.append(
        "As another example, after analyzing the context and realizing we need to "
        "search for specific topics, we can use regex to find relevant sections and "
        "maintain state through buffers:\n\n"
        "```repl\n"
        "import re\n"
        'query_terms = ["magic", "number"]\n'
        "relevant_sections = []\n"
        "buffers = []\n"
        "\n"
        "# Search for sections containing our query terms\n"
        "for i, chunk in enumerate(context):\n"
        "    chunk_text = str(chunk).lower()\n"
        "    if any(term in chunk_text for term in query_terms):\n"
        "        relevant_sections.append((i, chunk))\n"
        "\n"
        "# Process each relevant section and print findings\n"
        "for section_idx, section_content in relevant_sections:\n"
        '    print(f"Found relevant section {section_idx}:")\n'
        '    print(f"Content: {section_content[:500]}...")\n'
        '    buffers.append(f"Section {section_idx}: Contains references")\n'
        "\n"
        'print(f"Total relevant sections found: {len(relevant_sections)}")\n'
        'print("Summary of findings:")\n'
        "for buffer in buffers:\n"
        '    print(f"- {buffer}")\n'
        "```"
    )

    parts.append(
        "IMPORTANT: When you are done with the iterative process, you MUST provide a "
        "final answer inside a FINAL function when you have completed your task, NOT "
        "in code. Do not use these tags unless you have completed your task. You have "
        "two options:\n\n"
        "1. Use FINAL(your final answer here) to provide the answer directly\n"
        "2. Use FINAL_VAR(variable_name) to return a variable you have created in the "
        "REPL environment as your final output\n\n"
        "Note: If you are ready to provide a final answer, you cannot write anything "
        "other than the final answer in the FINAL or FINAL_VAR tags."
    )

    parts.append(
        "Think step by step carefully, plan, and execute this plan immediately in your "
        'response — do not just say "I will do this". Output to the REPL environment '
        "as much as possible. Remember to explicitly answer the original query in "
        "your final answer."
    )

    return "\n\n".join(parts)


def build_nudge_prompt() -> str:
    """Return a prompt that asks the model to finalize its answer."""
    return (
        "You have reached the maximum number of iterations. "
        "Please provide your final answer now using "
        "FINAL(your answer) or FINAL_VAR(variable_name)."
    )
