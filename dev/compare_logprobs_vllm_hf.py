"""Compare log-probabilities between vLLM and HF Transformers for Ouro.

Generates a few sequences with vLLM (capturing per-token log-probs), then
recomputes log-probs through HF Transformers on the same prompt+response
tokens. Reports the numerical mismatch to determine whether vLLM log-probs
can safely replace the `old_log_probs` forward pass in GRPO training.

If the max absolute deviation is well within the clipping range (e.g. ratio
deviation < 0.01 vs clip_eps = 0.2), vLLM log-probs are safe to use as the
frozen anchor for multi-iteration GRPO.

Usage:
    uv run python dev/compare_logprobs_vllm_hf.py
    uv run python dev/compare_logprobs_vllm_hf.py --model ByteDance/Ouro-1.4B-Thinking
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from ouro_rl.data import format_prompt
from ouro_rl.patches import patch_ouro_post_load

PROBLEMS = [
    "What is 7 * 8?",
    "Find the derivative of f(x) = 3x^2 + 2x - 5.",
    "If a triangle has sides of length 3, 4, and 5, what is its area?",
    "Solve for x: 2x + 7 = 15",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare vLLM vs HF log-probs")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=2816)
    p.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Step 1: Generate with vLLM and capture log-probs ---
    print("=" * 70)
    print("Step 1: Generating with vLLM (capturing per-token log-probs)")
    print("=" * 70)

    prompts = [format_prompt(prob, tokenizer) for prob in PROBLEMS]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=0,  # Return log-prob of the sampled token only.
    )

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    vllm_outputs = llm.generate(prompts, sampling_params)
    del llm
    torch.cuda.empty_cache()

    # Extract per-token log-probs from vLLM outputs.
    vllm_data: list[dict] = []
    for i, output in enumerate(vllm_outputs):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)

        # Each logprobs entry is dict[token_id -> Logprob].
        # We want the log-prob of the actually sampled token.
        lp_values = []
        for pos, lp_dict in enumerate(completion.logprobs):
            sampled_id = token_ids[pos]
            lp_values.append(lp_dict[sampled_id].logprob)

        vllm_data.append(
            {
                "problem": PROBLEMS[i],
                "prompt": prompts[i],
                "response_token_ids": token_ids,
                "response_text": completion.text,
                "vllm_logprobs": lp_values,
            }
        )
        print(f"\n[{i}] {PROBLEMS[i]}")
        print(f"    Generated {len(token_ids)} tokens")
        print(f"    Response: {completion.text[:100]}...")

    # --- Step 2: Recompute log-probs with HF Transformers ---
    print("\n" + "=" * 70)
    print("Step 2: Recomputing log-probs with HF Transformers")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    patch_ouro_post_load(model, tokenizer)

    device = next(model.parameters()).device

    results: list[dict] = []
    for i, data in enumerate(vllm_data):
        # Tokenize the prompt.
        prompt_ids = tokenizer.encode(data["prompt"], add_special_tokens=False)
        response_ids = data["response_token_ids"]

        # Concatenate prompt + response.
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Forward pass through HF model (no grad).
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, vocab)

        # Compute per-token log-probs (shifted: logits[t] predicts token[t+1]).
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

        # Extract response-only log-probs.
        # Response starts at position len(prompt_ids) in the full sequence.
        # In the shifted view, that's position len(prompt_ids) - 1.
        resp_start_shifted = len(prompt_ids) - 1
        hf_logprobs = token_lp[0, resp_start_shifted:].float().cpu().tolist()

        # Align lengths (HF might have one fewer if response is at the end).
        vllm_lp = data["vllm_logprobs"]
        n = min(len(hf_logprobs), len(vllm_lp))
        hf_lp = hf_logprobs[:n]
        vllm_lp = vllm_lp[:n]

        # Compute differences.
        diffs = [h - v for h, v in zip(hf_lp, vllm_lp)]
        abs_diffs = [abs(d) for d in diffs]
        ratios = [float(torch.exp(torch.tensor(d))) for d in diffs]

        result = {
            "problem": PROBLEMS[i],
            "num_tokens": n,
            "mean_abs_diff": sum(abs_diffs) / len(abs_diffs),
            "max_abs_diff": max(abs_diffs),
            "mean_ratio": sum(ratios) / len(ratios),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "ratio_std": float(torch.tensor(ratios).std()),
        }
        results.append(result)

        print(f"\n[{i}] {PROBLEMS[i]} ({n} tokens)")
        print(f"    Mean |diff|:  {result['mean_abs_diff']:.6f}")
        print(f"    Max  |diff|:  {result['max_abs_diff']:.6f}")
        print(f"    Ratio range:  [{result['min_ratio']:.6f}, {result['max_ratio']:.6f}]")
        print(f"    Mean ratio:   {result['mean_ratio']:.6f}  (ideal: 1.0)")

        # Show worst-case tokens.
        worst_idx = abs_diffs.index(max(abs_diffs))
        print(
            f"    Worst token:  pos={worst_idx}, "
            f"vLLM={vllm_lp[worst_idx]:.6f}, "
            f"HF={hf_lp[worst_idx]:.6f}, "
            f"diff={diffs[worst_idx]:.6f}, "
            f"ratio={ratios[worst_idx]:.6f}"
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_max_diffs = [r["max_abs_diff"] for r in results]
    all_max_ratios = [r["max_ratio"] for r in results]
    all_min_ratios = [r["min_ratio"] for r in results]
    overall_max_diff = max(all_max_diffs)
    overall_ratio_range = (min(all_min_ratios), max(all_max_ratios))

    print(f"\nOverall max |log-prob diff|: {overall_max_diff:.6f}")
    print(f"Overall ratio range:        [{overall_ratio_range[0]:.6f}, {overall_ratio_range[1]:.6f}]")
    print("Clip range (eps=0.2):       [0.800000, 1.200000]")

    if overall_ratio_range[0] > 0.99 and overall_ratio_range[1] < 1.01:
        print("\n-> Mismatch is NEGLIGIBLE. vLLM log-probs are safe to use as old_log_probs.")
    elif overall_ratio_range[0] > 0.95 and overall_ratio_range[1] < 1.05:
        print("\n-> Mismatch is SMALL but non-trivial. Likely safe with clip_eps=0.2, but monitor mean_ratio during training.")
    else:
        print(
            "\n-> Mismatch is SIGNIFICANT. Using vLLM log-probs as old_log_probs "
            "would cause spurious clipping. Keep HF forward pass."
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "results": results,
                    "overall_max_log_diff": overall_max_diff,
                    "overall_ratio_range": list(overall_ratio_range),
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
