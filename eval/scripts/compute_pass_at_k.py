"""Compute pass@k from lm-eval sample logs.

Usage:
    python eval/scripts/compute_pass_at_k.py eval/outputs/MODEL/samples_TASK_TIMESTAMP.jsonl
    python eval/scripts/compute_pass_at_k.py eval/outputs/MODEL/samples_TASK_TIMESTAMP.jsonl --k 10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def compute_pass_at_k(samples_path: Path, k: int, filter_name: str | None = None):
    # doc_id -> list of exact_match scores, per filter
    by_doc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(samples_path) as f:
        for line in f:
            sample = json.loads(line)
            fname = sample.get("filter", "unknown")
            if filter_name and fname != filter_name:
                continue
            by_doc[fname][sample["doc_id"]].append(sample.get("exact_match", 0.0))

    for fname, docs in sorted(by_doc.items()):
        n_docs = len(docs)
        n_solved = sum(1 for scores in docs.values() if any(s >= 1.0 for s in scores))
        repeats = [len(scores) for scores in docs.values()]
        min_r, max_r = min(repeats), max(repeats)

        if max_r < k:
            print(f"  [{fname}] WARNING: only {max_r} repeats but k={k}")

        pass_at_k = n_solved / n_docs
        avg_acc = sum(s for scores in docs.values() for s in scores) / sum(repeats)
        print(f"  [{fname}] pass@{k}: {pass_at_k:.1%} ({n_solved}/{n_docs})  avg_acc: {avg_acc:.1%}  repeats: {min_r}-{max_r}")


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k from lm-eval sample logs")
    parser.add_argument("samples", type=Path, help="Path to samples JSONL file")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--filter", type=str, default=None, help="Only compute for this filter name")
    args = parser.parse_args()

    print(f"pass@{args.k} — {args.samples.name}")
    compute_pass_at_k(args.samples, args.k, args.filter)


if __name__ == "__main__":
    main()
