"""Tokenize the full GRPO dataset and report token-length statistics.

Usage:
    uv run python dev/dataset_token_stats.py
    uv run python dev/dataset_token_stats.py --enable-thinking   # with <think> prefix (default)
    uv run python dev/dataset_token_stats.py --no-thinking       # without <think> prefix
    uv run python dev/dataset_token_stats.py --system-prompt "You are helpful."
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ouro_rl.data import CHAT_TEMPLATE, format_prompt, load_math_train


def compute_stats(lengths: list[int]) -> dict[str, float]:
    a = np.array(lengths)
    return {
        "count": len(a),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset token length statistics")
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    parser.add_argument("--dataset", default="qwedsacf/competition_math")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="For context budget calculation")
    parser.add_argument("--output-dir", default="outputs/dev")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE

    # Load dataset.
    dataset = load_math_train(args.dataset)
    problems = dataset["problem"]
    print(f"Dataset: {args.dataset} — {len(problems)} problems")
    print(f"Model: {args.model}")
    print(f"enable_thinking: {args.enable_thinking}")
    print(f"system_prompt: {args.system_prompt!r}")
    print()

    # Tokenize all prompts.
    prompt_lengths: list[int] = []
    for problem in problems:
        prompt_str = format_prompt(problem, tokenizer, system_prompt=args.system_prompt, enable_thinking=args.enable_thinking)
        token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_lengths.append(len(token_ids))

    stats = compute_stats(prompt_lengths)

    # Context budget: prompt + max_new_tokens.
    budget_stats = {
        f"prompt+{args.max_new_tokens}_min": stats["min"] + args.max_new_tokens,
        f"prompt+{args.max_new_tokens}_max": stats["max"] + args.max_new_tokens,
        f"prompt+{args.max_new_tokens}_mean": stats["mean"] + args.max_new_tokens,
        f"prompt+{args.max_new_tokens}_p95": stats["p95"] + args.max_new_tokens,
        f"prompt+{args.max_new_tokens}_p99": stats["p99"] + args.max_new_tokens,
    }

    # Length distribution buckets.
    a = np.array(prompt_lengths)
    buckets = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, np.inf]
    bucket_labels = []
    bucket_counts = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        label = f"{int(lo)}-{int(hi) if hi != np.inf else '+'}"
        count = int(((a >= lo) & (a < hi)).sum())
        bucket_labels.append(label)
        bucket_counts.append(count)

    # Print results.
    print("=" * 60)
    print("PROMPT TOKEN LENGTH STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>10}: {v:>10.1f}" if isinstance(v, float) else f"  {k:>10}: {v:>10}")
    print()
    print("CONTEXT BUDGET (prompt + max_new_tokens)")
    print("-" * 60)
    for k, v in budget_stats.items():
        print(f"  {k:>30}: {v:>10.1f}" if isinstance(v, float) else f"  {k:>30}: {v:>10}")
    print()
    print("LENGTH DISTRIBUTION")
    print("-" * 60)
    for label, count in zip(bucket_labels, bucket_counts):
        pct = count / len(prompt_lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>10}: {count:>6} ({pct:5.1f}%) {bar}")

    # Save results.
    results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "enable_thinking": args.enable_thinking,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt_token_stats": stats,
        "context_budget": budget_stats,
        "distribution": dict(zip(bucket_labels, bucket_counts)),
    }
    out_path = output_dir / "dataset_token_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
