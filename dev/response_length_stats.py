"""Generate responses at temperature=1 on competition_math and report length statistics.

Uses vLLM for fast generation, then reports token-level response length
distribution, truncation rate, and accuracy.

Usage:
    uv run python dev/response_length_stats.py                          # 100 problems, 1 response each
    uv run python dev/response_length_stats.py --num-samples 500        # more problems
    uv run python dev/response_length_stats.py --num-responses 4        # multiple responses per problem
    uv run python dev/response_length_stats.py --max-new-tokens 4096    # longer budget
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ouro_rl.data import CHAT_TEMPLATE, format_prompt, load_math_train
from ouro_rl.patches import CORRECT_EOS_TOKEN_ID, patch_ouro
from ouro_rl.reward import score_answer


def compute_stats(lengths: list[int]) -> dict[str, float]:
    a = np.array(lengths)
    return {
        "count": len(a),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p10": int(np.percentile(a, 10)),
        "p25": int(np.percentile(a, 25)),
        "p75": int(np.percentile(a, 75)),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Response length statistics at temperature=1")
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    parser.add_argument("--dataset", default="qwedsacf/competition_math")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of problems to sample (0 = all)")
    parser.add_argument("--num-responses", type=int, default=1, help="Responses per problem")
    parser.add_argument("--max-new-tokens", type=int, default=7896)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/dev")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load patches.
    patch_ouro()

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE

    # Load dataset.
    dataset = load_math_train(args.dataset)
    problems = dataset["problem"]
    solutions = dataset["solution"]

    # Sample problems.
    if args.num_samples > 0 and args.num_samples < len(problems):
        indices = random.sample(range(len(problems)), args.num_samples)
    else:
        indices = list(range(len(problems)))
        args.num_samples = len(problems)

    sampled_problems = [problems[i] for i in indices]
    sampled_solutions = [solutions[i] for i in indices]

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Problems: {len(sampled_problems)} / {len(problems)}")
    print(f"Responses per problem: {args.num_responses}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"enable_thinking: {args.enable_thinking}")
    print()

    # Format prompts.
    prompts = [
        format_prompt(p, tokenizer, system_prompt=args.system_prompt, enable_thinking=args.enable_thinking)
        for p in sampled_problems
    ]
    prompt_token_ids = [tokenizer.encode(p) for p in prompts]

    # Generate with vLLM.
    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[CORRECT_EOS_TOKEN_ID],
        skip_special_tokens=False,
    )

    print("Generating responses...")
    gen_start = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        skip_tokenizer_init=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params,
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    gen_time = time.time() - gen_start
    print(f"Generation complete in {gen_time:.1f}s\n")

    # Collect per-response data.
    all_lengths: list[int] = []
    completed_lengths: list[int] = []
    truncated_lengths: list[int] = []
    num_correct = 0
    num_total = 0
    per_problem: list[dict] = []

    for prompt_idx, output in enumerate(outputs):
        problem_data = {
            "problem_idx": indices[prompt_idx],
            "prompt_tokens": len(output.prompt_token_ids),
            "responses": [],
        }
        for resp in output.outputs:
            resp_len = len(resp.token_ids)
            resp_ids = list(resp.token_ids)
            completed = resp_ids[-1] == CORRECT_EOS_TOKEN_ID if resp_ids else False
            text = tokenizer.decode(resp_ids)
            correct = score_answer(text, sampled_solutions[prompt_idx]) == 1.0

            all_lengths.append(resp_len)
            if completed:
                completed_lengths.append(resp_len)
            else:
                truncated_lengths.append(resp_len)
            if correct:
                num_correct += 1
            num_total += 1

            problem_data["responses"].append(
                {
                    "length": resp_len,
                    "completed": completed,
                    "correct": correct,
                }
            )
        per_problem.append(problem_data)

    # Print statistics.
    print("=" * 70)
    print("RESPONSE LENGTH STATISTICS (all responses)")
    print("=" * 70)
    stats = compute_stats(all_lengths)
    for k, v in stats.items():
        print(f"  {k:>10}: {v:>10.1f}" if isinstance(v, float) else f"  {k:>10}: {v:>10}")

    truncation_rate = len(truncated_lengths) / len(all_lengths) * 100
    accuracy = num_correct / num_total * 100
    print()
    print(f"  Truncated: {len(truncated_lengths)}/{len(all_lengths)} ({truncation_rate:.1f}%)")
    print(f"  Accuracy:  {num_correct}/{num_total} ({accuracy:.1f}%)")

    if completed_lengths:
        print()
        print("-" * 70)
        print("COMPLETED RESPONSES (finished with EOS)")
        print("-" * 70)
        completed_stats = compute_stats(completed_lengths)
        for k, v in completed_stats.items():
            print(f"  {k:>10}: {v:>10.1f}" if isinstance(v, float) else f"  {k:>10}: {v:>10}")

    if truncated_lengths:
        print()
        print("-" * 70)
        print("TRUNCATED RESPONSES (hit max_new_tokens)")
        print("-" * 70)
        truncated_stats = compute_stats(truncated_lengths)
        for k, v in truncated_stats.items():
            print(f"  {k:>10}: {v:>10.1f}" if isinstance(v, float) else f"  {k:>10}: {v:>10}")

    # Length distribution buckets.
    print()
    print("-" * 70)
    print("LENGTH DISTRIBUTION")
    print("-" * 70)
    a = np.array(all_lengths)
    buckets = [0, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, np.inf]
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        label = f"{int(lo)}-{int(hi) if hi != np.inf else '+'}"
        count = int(((a >= lo) & (a < hi)).sum())
        pct = count / len(a) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>10}: {count:>6} ({pct:5.1f}%) {bar}")

    # Save results.
    results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "num_responses": args.num_responses,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "enable_thinking": args.enable_thinking,
            "system_prompt": args.system_prompt,
            "seed": args.seed,
        },
        "timing": {"generation_seconds": gen_time},
        "summary": {
            "all_responses": stats,
            "completed_responses": compute_stats(completed_lengths) if completed_lengths else None,
            "truncated_responses": compute_stats(truncated_lengths) if truncated_lengths else None,
            "truncation_rate": truncation_rate,
            "accuracy": accuracy,
        },
        "per_problem": per_problem,
    }
    out_path = output_dir / "response_length_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
