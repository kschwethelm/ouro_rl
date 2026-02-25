"""Measure per-layer angular distance across UT steps for the Ouro model.

Replicates the ETD paper's (arXiv:2510.07358) key analysis: computing the
angular distance d(l, l+1) between consecutive layer representations to
identify encoder/thinking/decoder functional regimes.

For Ouro (a Universal Transformer that loops all layers), we plot one curve
per UT step to see how the layer-level pattern evolves across recurrence steps.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from common import (
    format_prompt,
    forward_with_layer_tracking,
    load_gsm8k,
    load_model,
)
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "outputs"


def compute_angular_distances(
    states: list[list[torch.Tensor]],
) -> np.ndarray:
    """Compute angular distance between consecutive layer representations.

    Args:
        states: states[ut_step][layer_boundary] = tensor of shape (D,).
            N+1 boundaries per UT step (input to layer 0, output of each layer).

    Returns:
        Array of shape (total_ut_steps, num_transitions) where
        num_transitions = num_layers (N transitions from N+1 boundaries).
        Values in [0, 1] (normalized by 1/pi).
    """
    total_ut_steps = len(states)
    num_transitions = len(states[0]) - 1
    distances = np.zeros((total_ut_steps, num_transitions))

    for step in range(total_ut_steps):
        for t in range(num_transitions):
            x_l = states[step][t].float()
            x_l1 = states[step][t + 1].float()
            cos_sim = F.cosine_similarity(x_l.unsqueeze(0), x_l1.unsqueeze(0)).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))  # clamp for numerical safety
            distances[step, t] = math.acos(cos_sim) / math.pi

    return distances


def plot_angular_distances(
    mean_distances: np.ndarray,
    std_distances: np.ndarray | None,
    num_samples: int,
    total_ut_steps: int,
    output_path: Path,
    kneedle_knees: list[int] | None = None,
):
    """Plot angular distance vs layer transition, one line per UT step.

    Args:
        mean_distances: Shape (total_ut_steps, num_transitions).
        std_distances: Shape (total_ut_steps, num_transitions), or None.
        num_samples: Number of samples averaged over.
        total_ut_steps: Number of UT steps.
        output_path: Where to save the plot.
        kneedle_knees: Layer indices where Kneedle detected boundaries.
    """
    num_transitions = mean_distances.shape[1]
    x = np.arange(num_transitions)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, total_ut_steps))

    fig, ax = plt.subplots(figsize=(12, 5))

    for step in range(total_ut_steps):
        label = f"UT step {step}"
        ax.plot(x, mean_distances[step], color=colors[step], label=label, linewidth=1.8, marker="o", markersize=3)
        if std_distances is not None:
            ax.fill_between(
                x,
                mean_distances[step] - std_distances[step],
                mean_distances[step] + std_distances[step],
                alpha=0.15,
                color=colors[step],
            )

    if kneedle_knees:
        for knee in kneedle_knees:
            ax.axvline(x=knee, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
            ax.text(knee + 0.3, ax.get_ylim()[1] * 0.95, f"knee={knee}", color="red", fontsize=8, va="top")

    ax.set_xlabel("Layer transition (l -> l+1)", fontsize=11)
    ax.set_ylabel("Angular distance d(l, l+1)", fontsize=11)
    ax.set_title(
        f"Per-layer angular distance across UT steps (avg over {num_samples} GSM8K samples)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, num_transitions - 0.5)
    ax.set_xticks(range(0, num_transitions, max(1, num_transitions // 16)))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def try_kneedle(mean_distances: np.ndarray, step: int = 0) -> list[int]:
    """Run Kneedle algorithm on step 0 angular distances to detect E/T/D boundaries.

    Returns list of knee layer indices, or empty list if kneedle not installed.
    """
    try:
        from kneed import KneeLocator
    except ImportError:
        print("kneed not installed, skipping Kneedle boundary detection")
        return []

    y = mean_distances[step]
    x = np.arange(len(y))

    # Detect encoder boundary (decreasing convex -> knee)
    knees = []
    kneedle = KneeLocator(
        x,
        y,
        curve="convex",
        direction="decreasing",
        interp_method="polynomial",
        polynomial_degree=2,
        online=True,
    )
    if kneedle.knee is not None:
        knees.append(int(kneedle.knee))
        print(f"Kneedle encoder boundary (step {step}): layer {kneedle.knee}")

    # Detect decoder boundary (from the right, increasing concave -> knee)
    kneedle_rev = KneeLocator(
        x,
        y,
        curve="concave",
        direction="increasing",
        interp_method="polynomial",
        polynomial_degree=2,
        online=True,
    )
    if kneedle_rev.knee is not None:
        knees.append(int(kneedle_rev.knee))
        print(f"Kneedle decoder boundary (step {step}): layer {kneedle_rev.knee}")

    return knees


def main():
    parser = argparse.ArgumentParser(description="Measure per-layer angular distance across UT steps")
    parser.add_argument("--model", type=str, default=None, help="Model name/path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of GSM8K samples to average over")
    parser.add_argument("--n-loops", type=int, default=None, help="Override total_ut_steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode in prompt")
    parser.add_argument("--kneedle", action="store_true", help="Run Kneedle algorithm to detect E/T/D boundaries")
    args = parser.parse_args()

    load_kwargs = {}
    if args.model:
        load_kwargs["model_name"] = args.model
    model, tokenizer = load_model(**load_kwargs)

    total_ut_steps = args.n_loops or model.config.total_ut_steps
    num_layers = model.config.num_hidden_layers

    print(f"Model: {model.config._name_or_path}")
    print(f"Layers: {num_layers}, UT steps: {total_ut_steps}")
    print(f"Samples: {args.num_samples}")

    dataset = load_gsm8k()

    # Collect angular distances across samples
    all_distances: list[np.ndarray] = []

    for i in tqdm(range(args.num_samples), desc="Computing angular distances"):
        prompt_ids = format_prompt(
            dataset[i]["question"],
            tokenizer,
            enable_thinking=args.enable_thinking,
        )
        states = forward_with_layer_tracking(model, prompt_ids, n_loops=args.n_loops)
        distances = compute_angular_distances(states)
        all_distances.append(distances)

    # Aggregate: mean and std across samples
    stacked = np.stack(all_distances)  # (num_samples, total_ut_steps, num_transitions)
    mean_distances = stacked.mean(axis=0)
    std_distances = stacked.std(axis=0)

    # Print summary
    for step in range(total_ut_steps):
        d = mean_distances[step]
        print(
            f"UT step {step}: mean={d.mean():.4f}, min={d.min():.4f} (layer {d.argmin()}), max={d.max():.4f} (layer {d.argmax()})"
        )

    # Optional Kneedle detection
    knees = try_kneedle(mean_distances) if args.kneedle else None

    # Plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_avg{args.num_samples}" if args.num_samples > 1 else ""
    output_path = OUTPUT_DIR / f"angular_distance_steps{total_ut_steps}{suffix}.png"

    plot_angular_distances(
        mean_distances,
        std_distances,
        args.num_samples,
        total_ut_steps,
        output_path,
        kneedle_knees=knees,
    )


if __name__ == "__main__":
    main()
