"""Plotting utilities for benchmark results.

Generates accuracy and token usage charts comparing base vs RLM.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .results import BenchmarkResults


def plot_accuracy_vs_size(
    base_results: BenchmarkResults,
    rlm_results: BenchmarkResults,
    title: str,
    output_path: Path,
) -> None:
    """Line chart: accuracy (y) vs context size (x) for base vs RLM."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, label, color, marker in [
        (base_results, "Base (direct LLM)", "#d62728", "o"),
        (rlm_results, "RLM", "#2ca02c", "s"),
    ]:
        groups: dict[str, list[float]] = defaultdict(list)
        for t in results.tasks:
            groups[t.context_size_label].append(t.score)

        labels = sorted(groups.keys(), key=_size_sort_key)
        accs = [sum(groups[lbl]) / len(groups[lbl]) * 100 if groups[lbl] else 0 for lbl in labels]
        ax.plot(labels, accs, marker=marker, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Context Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tokens_vs_size(
    base_results: BenchmarkResults,
    rlm_results: BenchmarkResults,
    title: str,
    output_path: Path,
) -> None:
    """Line chart: mean total tokens vs context size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, label, color in [
        (base_results, "Base", "#d62728"),
        (rlm_results, "RLM", "#2ca02c"),
    ]:
        groups: dict[str, list[int]] = defaultdict(list)
        for t in results.tasks:
            if t.error != "context_exceeded":
                groups[t.context_size_label].append(t.total_input_tokens + t.total_output_tokens)

        labels = sorted(groups.keys(), key=_size_sort_key)
        means = [sum(groups[lbl]) / len(groups[lbl]) if groups[lbl] else 0 for lbl in labels]
        ax.plot(labels, means, marker="o", label=label, color=color, linewidth=2)

    ax.set_xlabel("Context Size", fontsize=12)
    ax.set_ylabel("Mean Total Tokens", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_iterations_vs_size(
    rlm_results: BenchmarkResults,
    title: str,
    output_path: Path,
) -> None:
    """Bar chart: mean iterations + sub-calls for RLM runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups_iter: dict[str, list[int]] = defaultdict(list)
    groups_sub: dict[str, list[int]] = defaultdict(list)

    for t in rlm_results.tasks:
        if t.iterations is not None:
            groups_iter[t.context_size_label].append(t.iterations)
        if t.sub_calls is not None:
            groups_sub[t.context_size_label].append(t.sub_calls)

    labels = sorted(groups_iter.keys(), key=_size_sort_key)
    mean_iter = [
        sum(groups_iter[lbl]) / len(groups_iter[lbl]) if groups_iter[lbl] else 0 for lbl in labels
    ]
    mean_sub = [
        sum(groups_sub[lbl]) / len(groups_sub[lbl]) if groups_sub[lbl] else 0 for lbl in labels
    ]

    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mean_iter, width, label="Iterations", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], mean_sub, width, label="Sub-calls", color="#ff7f0e")

    ax.set_xlabel("Context Size", fontsize=12)
    ax.set_ylabel("Mean Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _size_sort_key(label: str) -> int:
    s = label.replace("K", "000").replace("M", "000000").replace("_entries", "")
    try:
        return int(s)
    except ValueError:
        return 0
