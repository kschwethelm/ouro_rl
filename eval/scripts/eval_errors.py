"""Analyze failed samples from an lm-eval samples JSONL file.

Usage:
    uv run python eval/scripts/eval_errors.py <samples.jsonl> [--filter boxed] [--verbose]
"""

import argparse
import json
from pathlib import Path


def analyze_errors(path: str, filter_name: str | None = None, verbose: bool = False) -> None:
    samples = Path(path)
    all_samples: list[dict] = []
    errors: list[dict] = []

    with samples.open() as f:
        for line in f:
            s = json.loads(line)
            if filter_name and s.get("filter") != filter_name:
                continue
            all_samples.append(s)
            if s.get("exact_match") == 0.0:
                errors.append(s)

    n_total = len(all_samples)
    n_correct = n_total - len(errors)

    # Categorize errors
    invalid_extraction: list[dict] = []  # regex couldn't extract anything
    wrong_answer: list[dict] = []  # extracted an answer, but it was wrong

    for s in errors:
        filtered = s["filtered_resps"]
        # lm-eval returns ["[invalid]"] when the regex/filter fails to extract
        if filtered == ["[invalid]"]:
            invalid_extraction.append(s)
        else:
            wrong_answer.append(s)

    # Print summary
    filter_label = f" (filter={filter_name})" if filter_name else ""
    print(f"=== Eval Error Analysis{filter_label} ===")
    print(f"Total samples:      {n_total}")
    print(f"Correct:            {n_correct} ({100 * n_correct / n_total:.1f}%)")
    print(f"Wrong:              {len(errors)} ({100 * len(errors) / n_total:.1f}%)")
    print(f"  Wrong answer:     {len(wrong_answer)} (extracted answer, but incorrect)")
    print(f"  Invalid extract:  {len(invalid_extraction)} (regex failed — likely truncated)")
    print()

    # Check if invalid extractions are due to max token truncation
    if invalid_extraction:
        resp_lengths = [len(s["resps"][0][0]) for s in invalid_extraction]
        max_len = max(resp_lengths)
        min_len = min(resp_lengths)
        avg_len = sum(resp_lengths) / len(resp_lengths)
        print("--- Invalid extraction response lengths ---")
        print(f"  min={min_len}  avg={avg_len:.0f}  max={max_len}")

        # Check how many have \boxed anywhere in the response
        has_boxed = sum(1 for s in invalid_extraction if "\\boxed" in s["resps"][0][0])
        print(f"  Contains \\boxed but still failed: {has_boxed}/{len(invalid_extraction)}")
        no_boxed = len(invalid_extraction) - has_boxed
        print(f"  No \\boxed at all (truncated?):    {no_boxed}/{len(invalid_extraction)}")
        print()

    # Show wrong answer details
    if wrong_answer and verbose:
        print(f"--- Wrong answers ({len(wrong_answer)}) ---")
        for s in wrong_answer:
            question = s["doc"]["question"]
            if len(question) > 100:
                question = question[:100] + "..."
            extracted = s["filtered_resps"][0] if s["filtered_resps"] else "???"
            print(f"  doc_id={s['doc_id']:>4}  target={s['target']:<12} got={extracted:<12} q={question}")
        print()

    # Show invalid extraction details
    if invalid_extraction and verbose:
        print(f"--- Invalid extractions ({len(invalid_extraction)}) ---")
        for s in invalid_extraction:
            question = s["doc"]["question"]
            if len(question) > 100:
                question = question[:100] + "..."
            resp = s["resps"][0][0]
            resp_tail = resp[-150:] if len(resp) > 150 else resp
            print(f"  doc_id={s['doc_id']:>4}  target={s['target']:<12} resp_len={len(resp)}")
            print(f"    q: {question}")
            print(f"    tail: ...{resp_tail}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to samples JSONL file")
    parser.add_argument("--filter", default=None, help="Only show errors for this filter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show individual error details")
    args = parser.parse_args()
    analyze_errors(args.path, args.filter, args.verbose)
