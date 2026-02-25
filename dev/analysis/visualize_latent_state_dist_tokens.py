"""Track latent states during response generation and visualize L2 distances per token.

This script:
1. Loads one test case from GSM8K
2. Generates a response using the Ouro Universal Transformer
3. Captures the normed hidden state after each UT step for ALL tokens
4. Plots a heatmap showing L2 distance between consecutive UT steps per token

Example:
    uv run python dev/analysis/visualize_latent_state_dist_tokens.py
    uv run python dev/analysis/visualize_latent_state_dist_tokens.py --n-loops 8 --sample-idx 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    format_prompt,
    generate_with_latent_tracking,
    load_gsm8k,
    load_model,
)
from matplotlib.colors import LogNorm


def plot_latent_state_distances(
    input_tokens: list[int],
    output_tokens: list[int],
    input_latent_states: list[list[torch.Tensor]],
    output_latent_states: list[list[torch.Tensor]],
    tokenizer,
    total_ut_steps: int,
    input_exit_pdf: list[list[float]] | None = None,
    output_exit_pdf: list[list[float]] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot heatmaps of L2 distances between consecutive UT steps for input and output tokens.

    Creates side-by-side heatmaps with shared colorbar:
    - Left: input token positions × UT step transitions
    - Right: output token positions × UT step transitions

    If exit_pdf data is provided, overlays cumulative exit probability as text
    annotations and marks the step where the gate would exit (cumulative >= 0.5)
    with a red border.
    """
    num_transitions = total_ut_steps - 1

    def compute_distances(latent_states_per_token: list[list[torch.Tensor]]) -> np.ndarray:
        num_tokens = len(latent_states_per_token)
        distances = np.zeros((num_tokens, num_transitions))
        for token_idx, states in enumerate(latent_states_per_token):
            for t in range(num_transitions):
                s1 = states[t].flatten()
                s2 = states[t + 1].flatten()
                distances[token_idx, t] = torch.norm(s2 - s1, p=2).item()
        return distances

    def compute_cumulative_exit(exit_pdf: list[list[float]]) -> np.ndarray:
        """Convert per-step exit PDF to cumulative exit probability."""
        arr = np.array(exit_pdf)  # (num_tokens, total_ut_steps)
        return np.cumsum(arr, axis=1)

    input_distances = compute_distances(input_latent_states) if input_latent_states else None
    output_distances = compute_distances(output_latent_states) if output_latent_states else None

    # Shared color scale
    all_distances = [d for d in (input_distances, output_distances) if d is not None]
    shared_vmin = vmin or min(d[d > 0].min() for d in all_distances if d[d > 0].size > 0)
    shared_vmax = vmax or max(d.max() for d in all_distances)
    shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)

    # Compute cumulative exit probabilities
    input_cum_exit = compute_cumulative_exit(input_exit_pdf) if input_exit_pdf else None
    output_cum_exit = compute_cumulative_exit(output_exit_pdf) if output_exit_pdf else None

    # Scale figure height to token count
    n_input = len(input_latent_states) if input_latent_states else 0
    n_output = len(output_latent_states) if output_latent_states else 0
    max_tokens = max(n_input, n_output, 1)
    cell_h = 0.18
    fig_height = max(6, min(24, max_tokens * cell_h + 2.5))

    # Add extra column for exit gate strip per panel
    has_gate = input_cum_exit is not None or output_cum_exit is not None
    if has_gate:
        fig = plt.figure(figsize=(34, fig_height))
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.12, 1, 0.12, 0.03], wspace=0.08)
    else:
        fig = plt.figure(figsize=(28, fig_height))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.12)

    last_im = None

    def annotate_exit_gate(ax_gate, cum_exit: np.ndarray, n_tokens: int):
        """Draw a heatmap strip showing cumulative exit probability per UT step."""
        # cum_exit shape: (n_tokens, total_ut_steps)
        im_gate = ax_gate.imshow(cum_exit, aspect="auto", cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
        ax_gate.set_xticks(range(total_ut_steps))
        ax_gate.set_xticklabels([str(i) for i in range(total_ut_steps)], fontsize=8)
        ax_gate.set_xlabel("UT Step", fontsize=9)
        ax_gate.set_title("Cum. Exit P", fontsize=10)
        ax_gate.set_yticks(range(n_tokens))
        ax_gate.set_yticklabels([])  # token labels are on the main heatmap

        # Annotate cells with probability and mark exit step
        tick_fontsize = max(3, min(6, 200 // max(n_tokens, 1)))
        for tok_idx in range(n_tokens):
            exit_step = None
            for step in range(total_ut_steps):
                p = cum_exit[tok_idx, step]
                ax_gate.text(
                    step,
                    tok_idx,
                    f"{p:.2f}",
                    ha="center",
                    va="center",
                    fontsize=tick_fontsize,
                    color="white" if p < 0.5 else "black",
                )
                # Mark first step where cumulative exit >= 0.5
                if exit_step is None and p >= 0.5:
                    exit_step = step
            if exit_step is not None:
                rect = plt.Rectangle(
                    (exit_step - 0.5, tok_idx - 0.5),
                    1,
                    1,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_gate.add_patch(rect)
        return im_gate

    # Left panel: Input tokens
    if input_distances is not None and n_input > 0:
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(input_distances, aspect="auto", cmap="viridis", interpolation="nearest", norm=shared_norm)
        ax.set_xlabel("UT Step Transition", fontsize=11)
        ax.set_ylabel("Input Token Position", fontsize=11)
        ax.set_title(f"Input Tokens (n={n_input})\nL2 Distance Between UT Steps", fontsize=12)
        ax.set_xticks(range(num_transitions))
        ax.set_xticklabels([f"{i}\u2192{i + 1}" for i in range(num_transitions)])
        input_token_texts = [tokenizer.decode([t]) for t in input_tokens[:n_input]]
        ax.set_yticks(range(n_input))
        tick_fontsize = max(3, min(7, 200 // n_input))
        ax.set_yticklabels([t[:15] for t in input_token_texts], fontsize=tick_fontsize)
        last_im = im

        if has_gate and input_cum_exit is not None:
            ax_gate = fig.add_subplot(gs[0, 1])
            annotate_exit_gate(ax_gate, input_cum_exit, n_input)

    # Right panel: Output tokens
    if output_distances is not None and n_output > 0:
        col_main = 2 if has_gate else 1
        ax = fig.add_subplot(gs[0, col_main])
        im = ax.imshow(output_distances, aspect="auto", cmap="viridis", interpolation="nearest", norm=shared_norm)
        ax.set_xlabel("UT Step Transition", fontsize=11)
        ax.set_ylabel("Output Token Position", fontsize=11)
        ax.set_title(f"Output Tokens (n={n_output})\nL2 Distance Between UT Steps", fontsize=12)
        ax.set_xticks(range(num_transitions))
        ax.set_xticklabels([f"{i}\u2192{i + 1}" for i in range(num_transitions)])
        output_token_texts = [tokenizer.decode([t]) for t in output_tokens[:n_output]]
        ax.set_yticks(range(n_output))
        tick_fontsize = max(3, min(7, 200 // n_output))
        ax.set_yticklabels([t[:15] for t in output_token_texts], fontsize=tick_fontsize)
        last_im = im

        if has_gate and output_cum_exit is not None:
            col_gate = 3 if has_gate else None
            ax_gate = fig.add_subplot(gs[0, col_gate])
            annotate_exit_gate(ax_gate, output_cum_exit, n_output)

    # Shared colorbar
    if last_im is not None:
        col_cbar = 4 if has_gate else 2
        cbar_ax = fig.add_subplot(gs[0, col_cbar])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("L2 Distance", fontsize=10)

    plt.suptitle(
        f"Latent State L2 Distances (total_ut_steps={total_ut_steps})"
        + (" | Red box = exit step (cum. P \u2265 0.5)" if has_gate else ""),
        fontsize=14,
        y=0.995,
    )

    # Save
    plots_dir = Path(__file__).resolve().parent / "outputs"
    plots_dir.mkdir(exist_ok=True)
    output_path = plots_dir / f"gsm8k_latent_state_distances_steps{total_ut_steps}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Track and plot per-token latent state distances")
    parser.add_argument("--model", type=str, default=None, help="Model name or path (default: Ouro-1.4B-Thinking)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to process (default: 0)")
    parser.add_argument("--n-loops", type=int, default=None, help="Override total_ut_steps (default: use model config)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0, greedy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--enable-thinking", action="store_true", default=False, help="Enable thinking mode (default: disabled)"
    )
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for colorbar (default: auto)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colorbar (default: auto)")
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

    sample = gsm8k[args.sample_idx]
    question = sample["question"]
    print(f"\nProcessing sample {args.sample_idx}")
    print(f"Question: {question}")

    prompt_ids = format_prompt(question, tokenizer, enable_thinking=args.enable_thinking)
    print(f"Prompt length: {len(prompt_ids)} tokens")

    generated_tokens, input_latent_states, output_latent_states, input_exit_pdf, output_exit_pdf = (
        generate_with_latent_tracking(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            n_loops=args.n_loops,
        )
    )

    response = tokenizer.decode(generated_tokens)
    print(f"\nGenerated {len(generated_tokens)} tokens")
    print(f"Captured states for {len(input_latent_states)} input + {len(output_latent_states)} output tokens")
    print(f"Exit gate data: {len(input_exit_pdf)} input + {len(output_exit_pdf)} output tokens")
    print(f"Response: {response}")

    if input_latent_states or output_latent_states:
        plot_latent_state_distances(
            input_tokens=prompt_ids,
            output_tokens=generated_tokens,
            input_latent_states=input_latent_states,
            output_latent_states=output_latent_states,
            tokenizer=tokenizer,
            total_ut_steps=total_ut_steps,
            input_exit_pdf=input_exit_pdf or None,
            output_exit_pdf=output_exit_pdf or None,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    else:
        print("No latent states captured for plotting")


if __name__ == "__main__":
    main()
