"""Track latent states during response generation and visualize average distances per UT step.

This script:
1. Loads one or more test cases from GSM8K
2. Generates responses using the Ouro Universal Transformer
3. Captures the normed hidden state after each UT step for ALL tokens (input + output)
   - Input tokens: captured during prefill pass
   - Output tokens: captured during autoregressive decoding
4. Plots average L2 distance, cosine similarity, and KL divergence per UT step for:
   - Input sequence only (prompt tokens)
   - Output sequence only (generated tokens)

The KL divergence measures how the output distribution (after lm_head) changes
between consecutive UT steps: KL(p_{i+1} || p_i).

Example:
    uv run python dev/analysis/visualize_latent_state_dist_per_loop.py --num-samples 5
    uv run python dev/analysis/visualize_latent_state_dist_per_loop.py --n-loops 8 --num-samples 3
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from common import (
    compute_intermediate_logits,
    format_prompt,
    generate_with_latent_tracking,
    load_gsm8k,
    load_model,
)


def compute_distance_metrics(
    latent_states_per_token: list[list[torch.Tensor]],
    total_ut_steps: int,
    num_input_tokens: int,
) -> dict:
    """Compute L2, cosine, relative L2, and state norm metrics per UT step transition.

    Returns dict with keys 'l2', 'cosine', 'relative_l2', 'state_norms', each containing:
        'full': (means, stds, medians, q25, q75) arrays of shape (transitions,) or (steps,)
        'input': same, for input tokens only
        'output': same, for output tokens only
    """
    num_tokens = len(latent_states_per_token)
    num_transitions = total_ut_steps - 1

    l2_distances = np.zeros((num_tokens, num_transitions))
    cosine_sims = np.zeros((num_tokens, num_transitions))
    angular_distances = np.zeros((num_tokens, num_transitions))
    relative_l2_distances = np.zeros((num_tokens, num_transitions))
    state_norms = np.zeros((num_tokens, total_ut_steps))

    for token_idx, states in enumerate(latent_states_per_token):
        for step_idx in range(total_ut_steps):
            state = states[step_idx].flatten()
            state_norms[token_idx, step_idx] = torch.norm(state, p=2).item()

            if step_idx < num_transitions:
                state2 = states[step_idx + 1].flatten()
                l2_dist = torch.norm(state2 - state, p=2).item()
                l2_distances[token_idx, step_idx] = l2_dist

                norm = state_norms[token_idx, step_idx]
                relative_l2_distances[token_idx, step_idx] = l2_dist / norm if norm > 0 else 0

                cos_sim = F.cosine_similarity(state.unsqueeze(0), state2.unsqueeze(0)).item()
                cosine_sims[token_idx, step_idx] = cos_sim
                angular_distances[token_idx, step_idx] = math.acos(max(-1.0, min(1.0, cos_sim))) / math.pi

    def compute_stats(data: np.ndarray, axis: int = 0):
        return (
            np.mean(data, axis=axis),
            np.std(data, axis=axis),
            np.median(data, axis=axis),
            np.percentile(data, 25, axis=axis),
            np.percentile(data, 75, axis=axis),
        )

    def split_stats(data: np.ndarray):
        full = compute_stats(data)
        inp = compute_stats(data[:num_input_tokens]) if num_input_tokens > 0 else (None,) * 5
        out = compute_stats(data[num_input_tokens:]) if num_input_tokens < num_tokens else (None,) * 5
        return {"full": full, "input": inp, "output": out}

    return {
        "l2": split_stats(l2_distances),
        "cosine": split_stats(cosine_sims),
        "angular": split_stats(angular_distances),
        "relative_l2": split_stats(relative_l2_distances),
        "state_norms": split_stats(state_norms),
    }


def compute_output_kl_divergence(
    intermediate_logits: list[torch.Tensor],
    num_input_tokens: int,
) -> dict:
    """Compute KL(p_{i+1} || p_i) per token per UT step transition.

    Args:
        intermediate_logits: List of total_ut_steps tensors, each (total_tokens, vocab_size).
        num_input_tokens: Number of prompt tokens (for input/output split).

    Returns:
        Dict with 'full', 'input', 'output' keys, each (means, stds, medians, q25, q75).
    """
    num_steps = len(intermediate_logits)
    num_tokens = intermediate_logits[0].shape[0]
    kl_divergences = np.zeros((num_tokens, num_steps - 1))

    prev_log_probs = None
    for step_idx in range(num_steps):
        log_probs = F.log_softmax(intermediate_logits[step_idx].float(), dim=-1)
        if prev_log_probs is not None:
            kl = F.kl_div(prev_log_probs, log_probs, reduction="none", log_target=True).sum(dim=-1)
            kl_divergences[:, step_idx - 1] = kl.numpy()
        prev_log_probs = log_probs

    def compute_stats(data: np.ndarray, axis: int = 0):
        return (
            np.mean(data, axis=axis),
            np.std(data, axis=axis),
            np.median(data, axis=axis),
            np.percentile(data, 25, axis=axis),
            np.percentile(data, 75, axis=axis),
        )

    full = compute_stats(kl_divergences)
    inp = compute_stats(kl_divergences[:num_input_tokens]) if num_input_tokens > 0 else (None,) * 5
    out = compute_stats(kl_divergences[num_input_tokens:]) if num_input_tokens < num_tokens else (None,) * 5
    return {"full": full, "input": inp, "output": out}


def aggregate_metrics_across_samples(all_metrics: list[dict]) -> dict:
    """Aggregate metrics from multiple samples by averaging means/stds across samples."""
    if not all_metrics:
        return {}

    def collect_and_aggregate(metric_key: str, seq_type: str):
        means_list, stds_list, medians_list, q25_list, q75_list = [], [], [], [], []

        for metrics in all_metrics:
            stats = metrics[metric_key][seq_type]
            if stats[0] is not None:
                means, stds, medians, q25, q75 = stats
                means_list.append(means)
                stds_list.append(stds)
                medians_list.append(medians)
                q25_list.append(q25)
                q75_list.append(q75)

        if not means_list:
            return (None,) * 5

        return (
            np.mean(means_list, axis=0),
            np.sqrt(np.mean(np.array(stds_list) ** 2, axis=0)),
            np.median(medians_list, axis=0),
            np.mean(q25_list, axis=0),
            np.mean(q75_list, axis=0),
        )

    metric_keys = ["l2", "cosine", "angular", "relative_l2", "state_norms", "kl_divergence"]
    result = {}
    for key in metric_keys:
        result[key] = {seq: collect_and_aggregate(key, seq) for seq in ("full", "input", "output")}
    return result


def plot_distance_curves(
    metrics: dict,
    total_ut_steps: int,
    num_input_tokens: int,
    num_output_tokens: int,
    question: str,
    num_samples: int = 1,
    log_scale_l2: bool = False,
):
    """Plot comprehensive distance metrics per UT step.

    Creates eight subplots: L2, relative L2, state norms, cosine similarity,
    and KL divergence — each with mean±std and median+IQR views.
    """
    transitions = np.arange(1, total_ut_steps)  # For between-step metrics
    steps = np.arange(0, total_ut_steps)  # For per-step metrics (norms)

    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

    colors = {"input": "#E69F00", "output": "#009E73"}
    linestyles = {"input": "--", "output": "-."}
    markers = {"input": "s", "output": "D"}

    samples_text = f", averaged over {num_samples} samples" if num_samples > 1 else ""

    def make_label(seq_type: str) -> str:
        n = num_input_tokens if seq_type == "input" else num_output_tokens
        return f"{seq_type.capitalize()} (n={n})"

    def plot_mean_std(ax, metric_data, x, title, ylabel, use_log=False):
        for seq_type in ("input", "output"):
            means, stds, _, _, _ = metric_data[seq_type]
            if means is None:
                continue
            ax.plot(
                x,
                means,
                label=make_label(seq_type),
                color=colors[seq_type],
                marker=markers[seq_type],
                linestyle=linestyles[seq_type],
                markersize=5,
            )
            ax.fill_between(
                x, np.maximum(means - stds, 1e-8) if use_log else means - stds, means + stds, alpha=0.2, color=colors[seq_type]
            )
        ax.set_xlabel("UT Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

    def plot_median_iqr(ax, metric_data, x, title, ylabel, use_log=False):
        for seq_type in ("input", "output"):
            _, _, medians, q25, q75 = metric_data[seq_type]
            if medians is None:
                continue
            ax.plot(
                x,
                medians,
                label=make_label(seq_type),
                color=colors[seq_type],
                marker=markers[seq_type],
                linestyle=linestyles[seq_type],
                markersize=5,
            )
            ax.fill_between(x, np.maximum(q25, 1e-8) if use_log else q25, q75, alpha=0.2, color=colors[seq_type])
        ax.set_xlabel("UT Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

    # Row 1: L2 Distance
    plot_mean_std(
        fig.add_subplot(gs[0, 0]),
        metrics["l2"],
        transitions,
        f"L2 Distance: Mean ± Std\n(total_ut_steps={total_ut_steps}{samples_text})",
        "L2 Distance (mean ± std)",
        use_log=log_scale_l2,
    )
    plot_median_iqr(
        fig.add_subplot(gs[0, 1]),
        metrics["l2"],
        transitions,
        "L2 Distance: Median + IQR\n(more robust to outliers)",
        "L2 Distance (median, IQR)",
        use_log=log_scale_l2,
    )

    # Row 2: Relative L2 + State Norms
    plot_median_iqr(
        fig.add_subplot(gs[1, 0]),
        metrics["relative_l2"],
        transitions,
        "Relative L2: ||s_{t+1} - s_t|| / ||s_t||\n(normalized by state magnitude)",
        "Relative L2 Distance",
    )
    plot_median_iqr(
        fig.add_subplot(gs[1, 1]),
        metrics["state_norms"],
        steps,
        "State Norms Across UT Steps (Log Scale)\n(magnitude of hidden states)",
        "State Norm ||s_t||",
        use_log=True,
    )

    # Row 3: Cosine Similarity
    plot_mean_std(
        fig.add_subplot(gs[2, 0]),
        metrics["cosine"],
        transitions,
        f"Cosine Similarity: Mean ± Std\nQuestion: {question[:50]}...",
        "Cosine Similarity (mean ± std)",
    )
    plot_median_iqr(
        fig.add_subplot(gs[2, 1]),
        metrics["cosine"],
        transitions,
        "Cosine Similarity: Median + IQR\n(more robust to outliers)",
        "Cosine Similarity (median, IQR)",
    )

    # Row 4: Angular Distance
    plot_mean_std(
        fig.add_subplot(gs[3, 0]),
        metrics["angular"],
        transitions,
        f"Angular Distance: Mean ± Std\n(d = arccos(cos_sim) / π, range [0,1]{samples_text})",
        "Angular Distance",
    )
    plot_median_iqr(
        fig.add_subplot(gs[3, 1]),
        metrics["angular"],
        transitions,
        "Angular Distance: Median + IQR\n(more robust to outliers)",
        "Angular Distance",
    )

    # Row 5: KL Divergence
    plot_mean_std(
        fig.add_subplot(gs[4, 0]),
        metrics["kl_divergence"],
        transitions,
        f"KL(p_{{i+1}} || p_i): Mean ± Std (Log Scale)\n(output distribution change per step{samples_text})",
        "KL Divergence (mean ± std)",
        use_log=True,
    )
    plot_median_iqr(
        fig.add_subplot(gs[4, 1]),
        metrics["kl_divergence"],
        transitions,
        "KL(p_{i+1} || p_i): Median + IQR (Log Scale)\n(more robust to outliers)",
        "KL Divergence (median, IQR)",
        use_log=True,
    )

    # Save
    plots_dir = Path(__file__).resolve().parent / "outputs"
    plots_dir.mkdir(exist_ok=True)
    samples_suffix = f"_avg{num_samples}" if num_samples > 1 else ""
    logscale_suffix = "_logscale" if log_scale_l2 else ""
    output_path = plots_dir / f"gsm8k_latent_distances_per_loop_steps{total_ut_steps}{samples_suffix}{logscale_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Track and plot average latent distances per UT step")
    parser.add_argument("--model", type=str, default=None, help="Model name or path (default: Ouro-1.4B-Thinking)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to start from (default: 0)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to average over (default: 1)")
    parser.add_argument("--n-loops", type=int, default=None, help="Override total_ut_steps (default: use model config)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0, greedy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--enable-thinking", action="store_true", default=False, help="Enable thinking mode (default: disabled)"
    )
    parser.add_argument("--log-scale-l2", action="store_true", help="Use log scale for L2 distance y-axis")
    args = parser.parse_args()

    # Load model
    model_name = args.model or None
    load_kwargs = {"model_name": model_name} if model_name else {}
    print(f"Loading model{f': {model_name}' if model_name else ''}...")
    model, tokenizer = load_model(**load_kwargs)

    total_ut_steps = args.n_loops or model.config.total_ut_steps
    print(f"Using total_ut_steps={total_ut_steps}")

    # Load dataset
    print("Loading GSM8K test set...")
    gsm8k = load_gsm8k()

    # Process samples
    all_metrics = []
    total_input_tokens = 0
    total_output_tokens = 0
    first_question = None

    for sample_offset in range(args.num_samples):
        sample_idx = args.sample_idx + sample_offset
        sample = gsm8k[sample_idx]
        question = sample["question"]

        if sample_offset == 0:
            first_question = question

        print(f"\nProcessing sample {sample_idx} ({sample_offset + 1}/{args.num_samples})")
        print(f"Question: {question[:80]}...")

        prompt_ids = format_prompt(question, tokenizer, enable_thinking=args.enable_thinking)
        print(f"Prompt length: {len(prompt_ids)} tokens")

        generated_tokens, input_latent_states, output_latent_states, _, _ = generate_with_latent_tracking(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed + sample_offset,
            n_loops=args.n_loops,
        )

        num_input = len(input_latent_states)
        num_output = len(output_latent_states)
        print(f"Generated {len(generated_tokens)} tokens, captured states for {num_input} input + {num_output} output tokens")
        print(f"Response: {tokenizer.decode(generated_tokens)[:100]}...")

        total_input_tokens += num_input
        total_output_tokens += num_output

        all_states = input_latent_states + output_latent_states
        if all_states:
            metrics = compute_distance_metrics(all_states, total_ut_steps, num_input)

            # Compute KL divergence from intermediate logits
            print("Computing intermediate logits for KL divergence...")
            intermediate_logits = compute_intermediate_logits(model, input_latent_states, output_latent_states)
            metrics["kl_divergence"] = compute_output_kl_divergence(intermediate_logits, num_input)

            all_metrics.append(metrics)
        else:
            print(f"Warning: No latent states captured for sample {sample_idx}")

    # Aggregate and plot
    if all_metrics:
        if args.num_samples > 1:
            print(f"\nAggregating metrics across {len(all_metrics)} samples...")
            final_metrics = aggregate_metrics_across_samples(all_metrics)
            avg_input = total_input_tokens // args.num_samples
            avg_output = total_output_tokens // args.num_samples
        else:
            final_metrics = all_metrics[0]
            avg_input = total_input_tokens
            avg_output = total_output_tokens

        plot_distance_curves(
            metrics=final_metrics,
            total_ut_steps=total_ut_steps,
            num_input_tokens=avg_input,
            num_output_tokens=avg_output,
            question=first_question,
            num_samples=args.num_samples,
            log_scale_l2=args.log_scale_l2,
        )
    else:
        print("No latent states captured for plotting")


if __name__ == "__main__":
    main()
