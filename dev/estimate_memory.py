"""Estimate GPU memory requirements for GRPO training on Ouro.

Prints a breakdown of model params, optimizer state, activations (with and
without gradient checkpointing), and the peak memory per GPU for different
micro-batch / sequence-length configurations.

Usage:
    uv run python dev/estimate_memory.py
    uv run python dev/estimate_memory.py --train-micro-batch 1 --max-seq-len 4096
    uv run python dev/estimate_memory.py --gradient-checkpointing
"""

from __future__ import annotations

import argparse

# ── Model architecture ──────────────────────────────────────────────────────
# Ouro-1.4B-Thinking defaults from configuration_ouro.py + HF model card.
# Universal Transformer: 28 physical layers × 4 UT steps = 112 effective layers,
# but weights are shared so param count = 28 physical layers.

DEFAULTS = {
    "vocab_size": 49152,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_hidden_layers": 24,  # physical layers (weight-shared across UT steps)
    "num_attention_heads": 16,
    "num_key_value_heads": 16,  # MHA (not GQA)
    "total_ut_steps": 4,
    "tie_word_embeddings": False,
}


def count_params(cfg: dict) -> dict[str, int]:
    """Count parameters by component."""
    h = cfg["hidden_size"]
    ff = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = h // n_heads
    vocab = cfg["vocab_size"]

    embed = vocab * h  # token embeddings
    lm_head = 0 if cfg["tie_word_embeddings"] else vocab * h

    # Per-layer:
    #   QKV projections: h*(n_heads*head_dim) + h*(n_kv*head_dim) + h*(n_kv*head_dim)
    #   O projection:    (n_heads*head_dim)*h
    #   MLP: gate h->ff, up h->ff, down ff->h  (no bias)
    #   RMSNorm: 4 norms per layer (pre-attn, post-attn, pre-mlp, post-mlp) = 4*h
    q = h * (n_heads * head_dim)
    k = h * (n_kv * head_dim)
    v = h * (n_kv * head_dim)
    o = (n_heads * head_dim) * h
    attn_per_layer = q + k + v + o

    mlp_per_layer = h * ff + h * ff + ff * h  # gate + up + down

    # Ouro has 4 RMSNorm per decoder layer:
    #   input_layernorm, input_layernorm_2, post_attention_layernorm, post_attention_layernorm_2
    norm_per_layer = 4 * h

    layer_total = attn_per_layer + mlp_per_layer + norm_per_layer
    all_layers = n_layers * layer_total

    # Final norm + early exit gate
    final_norm = h
    early_exit_gate = h * 1 + 1  # Linear(h, 1) with bias

    total = embed + lm_head + all_layers + final_norm + early_exit_gate

    return {
        "embedding": embed,
        "lm_head": lm_head,
        "per_layer_attn": attn_per_layer,
        "per_layer_mlp": mlp_per_layer,
        "per_layer_norm": norm_per_layer,
        "per_layer_total": layer_total,
        "all_layers": all_layers,
        "final_norm": final_norm,
        "early_exit_gate": early_exit_gate + 1,
        "total": total,
    }


def bytes_per_param(dtype: str) -> int:
    return {"float32": 4, "bfloat16": 2, "float16": 2}[dtype]


def fmt_gb(nbytes: int | float) -> str:
    return f"{nbytes / 2**30:.2f} GB"


def fmt_mb(nbytes: int | float) -> str:
    return f"{nbytes / 2**20:.1f} MB"


def estimate_activation_memory(
    cfg: dict,
    micro_batch: int,
    seq_len: int,
    dtype_bytes: int,
    gradient_checkpointing: bool,
) -> dict[str, float]:
    """Estimate peak activation memory for a forward+backward pass.

    This follows the analysis from "Reducing Activation Recomputation in Large
    Transformer Models" (Korthikanti et al., 2022).

    For a single transformer layer, the stored activations include:
    - Input to layer (for residual): B*S*H
    - Attention: QKV, softmax output, dropout mask, context
    - MLP: input, gate, up, activation
    - LayerNorms: inputs

    With gradient checkpointing, only the layer input is stored; everything
    else is recomputed during backward.
    """
    B = micro_batch
    S = seq_len
    H = cfg["hidden_size"]
    FF = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"]
    ut_steps = cfg["total_ut_steps"]
    head_dim = H // n_heads
    d = dtype_bytes

    # Effective layers = physical layers × UT steps (Ouro's universal transformer
    # reuses weights but the activations for each UT step are distinct)
    effective_layers = n_layers * ut_steps

    if gradient_checkpointing:
        # Only store input hidden states per checkpointed segment.
        # With per-layer checkpointing: store 1 activation per layer boundary.
        per_layer = B * S * H * d  # layer input
        # During recomputation of one layer, we temporarily need the full
        # layer activations. This adds ~1 layer's worth of non-checkpointed memory.
        recompute_buffer = _per_layer_activation(B, S, H, FF, n_heads, n_kv, head_dim, d)

        total_stored = effective_layers * per_layer
        peak = total_stored + recompute_buffer
    else:
        per_layer = _per_layer_activation(B, S, H, FF, n_heads, n_kv, head_dim, d)
        peak = effective_layers * per_layer

    # Logits: B * S * vocab * dtype (materialized for loss computation)
    logits = B * S * cfg["vocab_size"] * d

    # log_softmax output (same shape as logits) — needed for NLL gradient
    log_softmax = logits

    return {
        "per_layer": per_layer if not gradient_checkpointing else B * S * H * d,
        "effective_layers": effective_layers,
        "logits": logits,
        "log_softmax": log_softmax,
        "peak_activations": peak + logits + log_softmax,
    }


def _per_layer_activation(B: int, S: int, H: int, FF: int, n_heads: int, n_kv: int, head_dim: int, d: int) -> float:
    """Activation memory for one transformer layer (no checkpointing)."""
    # Attention block:
    #   - Q, K, V projections output: B*S*(n_heads*head_dim + 2*n_kv*head_dim)
    #   - Attention scores: B*n_heads*S*S (stored as float32 by flash attn? No, FA doesn't store.)
    #     With FlashAttention: O(B*n_heads*S) workspace, but we still save the output B*S*H
    #   - Attention output (before o_proj): B*S*H
    #   - Residual: B*S*H
    attn_qkv = B * S * (n_heads + 2 * n_kv) * head_dim * d
    # FlashAttention: no S*S matrix stored; only output + softmax LSE
    attn_out = B * S * H * d
    softmax_lse = B * n_heads * S * 4  # float32 logsumexp per head
    attn_total = attn_qkv + attn_out + softmax_lse

    # Norms (4 per layer): save input for backward
    norms = 4 * B * S * H * d

    # MLP: gate_proj, up_proj, act_fn(gate)*up (saved for down_proj backward)
    mlp_gate_up = 2 * B * S * FF * d  # gate + up intermediate results
    mlp_act = B * S * FF * d  # act_fn(gate) * up — input to down_proj

    # Residual connections: 2 per layer (post-attn, post-mlp)
    residuals = 2 * B * S * H * d

    return attn_total + norms + mlp_gate_up + mlp_act + residuals


def activation_bytes_per_token(cfg: dict, dtype_bytes: int, gradient_checkpointing: bool) -> float:
    """Return bytes of activation memory per token for micro_batch=1.

    All terms are O(S) with FlashAttention (no S×S matrix), so activation
    memory = coeff * S, making max pack_len analytically solvable.
    """
    H = cfg["hidden_size"]
    FF = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = H // n_heads
    effective_layers = cfg["num_hidden_layers"] * cfg["total_ut_steps"]
    d = dtype_bytes

    # Per-layer activation cost per token (S=1, B=1)
    attn_qkv = (n_heads + 2 * n_kv) * head_dim * d
    attn_out = H * d
    softmax_lse = n_heads * 4  # float32 LSE per head (B=1)
    norms = 4 * H * d
    mlp_gate_up = 2 * FF * d
    mlp_act = FF * d
    residuals = 2 * H * d
    full_layer_cost = attn_qkv + attn_out + softmax_lse + norms + mlp_gate_up + mlp_act + residuals

    if gradient_checkpointing:
        # Store only layer input (H*d per token); recompute one layer on backward.
        stored_cost = effective_layers * H * d
        recompute_cost = full_layer_cost  # peak during recompute of one layer
        layer_cost = stored_cost + recompute_cost
    else:
        layer_cost = effective_layers * full_layer_cost

    # Logits + log_softmax (materialized for NLL loss)
    logits_cost = 2 * cfg["vocab_size"] * d

    return layer_cost + logits_cost


def solve_max_pack_len(
    cfg: dict,
    fixed_bytes: float,
    gpu_memory_gb: float,
    dtype_bytes: int,
    gradient_checkpointing: bool,
    cuda_overhead_gb: float = 0.5,
) -> int:
    """Solve for the largest pack_len that fits in GPU memory (micro_batch=1)."""
    available = (gpu_memory_gb - cuda_overhead_gb) * 2**30
    activation_budget = available - fixed_bytes
    coeff = activation_bytes_per_token(cfg, dtype_bytes, gradient_checkpointing)
    if coeff <= 0 or activation_budget <= 0:
        return 0
    return int(activation_budget / coeff)


def estimate_optimizer_memory(num_params: int, dtype_bytes: int) -> dict[str, float]:
    """AdamW optimizer state: 2 fp32 states (mean, variance) + fp32 param copy."""
    # AdamW stores:
    #   - fp32 copy of params (if training in bf16): num_params * 4
    #   - exp_avg (momentum): num_params * 4
    #   - exp_avg_sq (variance): num_params * 4
    fp32_copy = num_params * 4 if dtype_bytes < 4 else 0
    exp_avg = num_params * 4
    exp_avg_sq = num_params * 4
    return {
        "fp32_param_copy": fp32_copy,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "total": fp32_copy + exp_avg + exp_avg_sq,
    }


def estimate_gradient_memory(num_params: int, dtype_bytes: int) -> int:
    """Gradients stored in same dtype as params."""
    return num_params * dtype_bytes


def main():
    parser = argparse.ArgumentParser(description="Estimate GRPO training memory")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-seq-len", type=int, default=6144, help="Max sequence length")
    parser.add_argument("--train-micro-batch", type=int, default=2, help="Training micro-batch size")
    parser.add_argument("--log-prob-micro-batch", type=int, default=4, help="Log-prob micro-batch size")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-memory-gb", type=float, default=80.0, help="Per-GPU memory in GB")

    # Allow overriding model dims
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)

    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    if args.hidden_size:
        cfg["hidden_size"] = args.hidden_size
    if args.intermediate_size:
        cfg["intermediate_size"] = args.intermediate_size
    if args.num_hidden_layers:
        cfg["num_hidden_layers"] = args.num_hidden_layers

    d = bytes_per_param(args.dtype)

    # ── 1. Parameters ────────────────────────────────────────────────────
    params = count_params(cfg)
    param_bytes = params["total"] * d

    print("=" * 72)
    print("GRPO MEMORY ESTIMATION — Ouro-1.4B-Thinking")
    print("=" * 72)

    print(
        f"\nModel: {cfg['num_hidden_layers']} layers × {cfg['total_ut_steps']} UT steps "
        f"= {cfg['num_hidden_layers'] * cfg['total_ut_steps']} effective layers"
    )
    print(
        f"  hidden={cfg['hidden_size']}, ff={cfg['intermediate_size']}, "
        f"heads={cfg['num_attention_heads']}, kv_heads={cfg['num_key_value_heads']}"
    )
    print(f"  dtype={args.dtype} ({d}B/param), tie_embeddings={cfg['tie_word_embeddings']}")

    print(f"\n{'─' * 72}")
    print("1. MODEL PARAMETERS")
    print(f"{'─' * 72}")
    print(f"  Embedding:          {params['embedding']:>12,}  ({fmt_mb(params['embedding'] * d)})")
    print(
        f"  LM head:            {params['lm_head']:>12,}  ({fmt_mb(params['lm_head'] * d)})"
        f"{'  (tied)' if cfg['tie_word_embeddings'] else ''}"
    )
    print(f"  Per-layer attn:     {params['per_layer_attn']:>12,}  ({fmt_mb(params['per_layer_attn'] * d)})")
    print(f"  Per-layer MLP:      {params['per_layer_mlp']:>12,}  ({fmt_mb(params['per_layer_mlp'] * d)})")
    print(f"  Per-layer norm:     {params['per_layer_norm']:>12,}  ({fmt_mb(params['per_layer_norm'] * d)})")
    print(f"  Per-layer total:    {params['per_layer_total']:>12,}  ({fmt_mb(params['per_layer_total'] * d)})")
    print(f"  All {cfg['num_hidden_layers']} layers:       {params['all_layers']:>12,}  ({fmt_gb(params['all_layers'] * d)})")
    print("  ────────────────────────────────────")
    print(f"  Total params:       {params['total']:>12,}  ({fmt_gb(param_bytes)})")

    # ── 2. Optimizer ─────────────────────────────────────────────────────
    opt = estimate_optimizer_memory(params["total"], d)

    print(f"\n{'─' * 72}")
    print("2. OPTIMIZER STATE (AdamW)")
    print(f"{'─' * 72}")
    if opt["fp32_param_copy"]:
        print(f"  FP32 param copy:    {fmt_gb(opt['fp32_param_copy'])}")
    print(f"  exp_avg (momentum): {fmt_gb(opt['exp_avg'])}")
    print(f"  exp_avg_sq (var):   {fmt_gb(opt['exp_avg_sq'])}")
    print("  ────────────────────────────────────")
    print(f"  Total optimizer:    {fmt_gb(opt['total'])}")

    # ── 3. Gradients ─────────────────────────────────────────────────────
    grad_bytes = estimate_gradient_memory(params["total"], d)

    print(f"\n{'─' * 72}")
    print("3. GRADIENTS")
    print(f"{'─' * 72}")
    print(f"  Gradients ({args.dtype}): {fmt_gb(grad_bytes)}")

    # ── 4. Activations ───────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("4. ACTIVATION MEMORY (forward + backward)")
    print(f"{'─' * 72}")

    configs_to_show = [
        (args.train_micro_batch, args.max_seq_len, args.gradient_checkpointing),
    ]
    # Also show alternatives for comparison
    alternatives = [
        (1, args.max_seq_len, False),
        (args.train_micro_batch, args.max_seq_len, True),
        (1, args.max_seq_len, True),
    ]
    for alt in alternatives:
        if alt not in configs_to_show:
            configs_to_show.append(alt)

    for mb, sl, gc in configs_to_show:
        act = estimate_activation_memory(cfg, mb, sl, d, gc)
        tag = (
            " ◄ CURRENT"
            if (mb == args.train_micro_batch and sl == args.max_seq_len and gc == args.gradient_checkpointing)
            else ""
        )
        gc_str = "+gc" if gc else ""
        print(f"\n  micro_batch={mb}, seq_len={sl} {gc_str}{tag}")
        print(f"    Per-layer stored:  {fmt_mb(act['per_layer'])}")
        print(f"    Effective layers:  {act['effective_layers']}")
        print(f"    Logits:            {fmt_mb(act['logits'])}")
        print(f"    Log-softmax:       {fmt_mb(act['log_softmax'])}")
        print(f"    Peak activations:  {fmt_gb(act['peak_activations'])}")

    # ── 5. Summary ───────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("5. PEAK MEMORY PER GPU (training forward+backward)")
    print(f"{'─' * 72}")
    print(f"  GPU capacity: {args.gpu_memory_gb:.0f} GB × {args.num_gpus} GPUs")
    print()

    # Fixed costs (same for all configs)
    fixed = param_bytes + opt["total"] + grad_bytes
    print("  Fixed costs:")
    print(f"    Model params:     {fmt_gb(param_bytes)}")
    print(f"    Optimizer:        {fmt_gb(opt['total'])}")
    print(f"    Gradients:        {fmt_gb(grad_bytes)}")
    print("    ──────────────────────────────")
    print(f"    Subtotal:         {fmt_gb(fixed)}")
    print()

    gpu_bytes = args.gpu_memory_gb * 2**30
    cuda_overhead = 0.5 * 2**30  # ~500MB CUDA context overhead

    for mb, sl, gc in configs_to_show:
        act = estimate_activation_memory(cfg, mb, sl, d, gc)
        peak = fixed + act["peak_activations"]
        available = gpu_bytes - cuda_overhead
        headroom = available - peak
        gc_str = "+gc" if gc else ""
        tag = (
            " ◄ CURRENT"
            if (mb == args.train_micro_batch and sl == args.max_seq_len and gc == args.gradient_checkpointing)
            else ""
        )
        status = "OK" if headroom > 0 else "OOM"
        print(f"  mb={mb}, seq={sl} {gc_str:>4}{tag}")
        print(f"    Activations:  {fmt_gb(act['peak_activations'])}")
        print(f"    Total peak:   {fmt_gb(peak)}")
        print(f"    Headroom:     {'+' if headroom > 0 else ''}{fmt_gb(headroom)}  [{status}]")
        print()

    # ── 6. Log-prob pass (inference, no grad) ────────────────────────────
    print(f"{'─' * 72}")
    print("6. LOG-PROB PASS (inference, no gradients)")
    print(f"{'─' * 72}")
    # No optimizer, no gradients, no activation storage for backward
    # Just model params + KV cache + logits
    lp_mb = args.log_prob_micro_batch
    kv_per_layer = (
        2 * lp_mb * args.max_seq_len * (cfg["num_key_value_heads"] * (cfg["hidden_size"] // cfg["num_attention_heads"])) * d
    )
    kv_total = kv_per_layer * cfg["num_hidden_layers"] * cfg["total_ut_steps"]
    logits_mem = lp_mb * args.max_seq_len * cfg["vocab_size"] * d
    lp_peak = param_bytes + kv_total + logits_mem
    print(f"  micro_batch={lp_mb}, seq_len={args.max_seq_len}")
    print(f"    Model params:   {fmt_gb(param_bytes)}")
    print(f"    KV cache:       {fmt_gb(kv_total)}")
    print(f"    Logits:         {fmt_gb(logits_mem)}")
    print(f"    Peak:           {fmt_gb(lp_peak)}")
    print()

    current_act = estimate_activation_memory(cfg, args.train_micro_batch, args.max_seq_len, d, args.gradient_checkpointing)
    current_peak = fixed + current_act["peak_activations"]

    # ── 7. Packed training: max pack_len ─────────────────────────────────
    print(f"{'─' * 72}")
    print("7. PACKED TRAINING — max pack_len (micro_batch=1)")
    print(f"{'─' * 72}")
    print("  Activation memory is O(S) with FlashAttention → analytically solvable.")
    print()
    for gc in (True, False):
        max_pl = solve_max_pack_len(cfg, fixed, args.gpu_memory_gb, d, gc)
        gc_str = "with gradient checkpointing" if gc else "without gradient checkpointing"
        coeff = activation_bytes_per_token(cfg, d, gc)
        act_at_max = coeff * max_pl
        print(f"  {gc_str}:")
        print(f"    Activation cost/token:  {coeff / 1024:.1f} KB/tok")
        print(f"    Max pack_len:           {max_pl:,} tokens")
        print(f"    Activation at max:      {fmt_gb(act_at_max)}")
        print(f"    Total peak at max:      {fmt_gb(fixed + act_at_max)}")
        print()

    print(f"{'=' * 72}")
    print("8. RECOMMENDATIONS")
    print(f"{'=' * 72}")

    if current_peak > gpu_bytes - cuda_overhead:
        print("\n  Current config exceeds GPU memory. Options:")
        # Check gradient checkpointing
        if not args.gradient_checkpointing:
            gc_act = estimate_activation_memory(cfg, args.train_micro_batch, args.max_seq_len, d, True)
            gc_peak = fixed + gc_act["peak_activations"]
            savings = current_peak - gc_peak
            print(f"\n  1. Enable gradient checkpointing (saves ~{fmt_gb(savings)})")
            if gc_peak < gpu_bytes - cuda_overhead:
                print(f"     → Fits! Peak would be {fmt_gb(gc_peak)}")
            else:
                print(f"     → Still OOM ({fmt_gb(gc_peak)}), combine with smaller micro-batch")

        mb1_act = estimate_activation_memory(cfg, 1, args.max_seq_len, d, args.gradient_checkpointing)
        mb1_peak = fixed + mb1_act["peak_activations"]
        print(f"\n  2. Reduce train_micro_batch to 1 (peak: {fmt_gb(mb1_peak)})")

        mb1_gc_act = estimate_activation_memory(cfg, 1, args.max_seq_len, d, True)
        mb1_gc_peak = fixed + mb1_gc_act["peak_activations"]
        print(f"\n  3. Both: micro_batch=1 + gradient checkpointing (peak: {fmt_gb(mb1_gc_peak)})")
    else:
        headroom = gpu_bytes - cuda_overhead - current_peak
        print(f"\n  Current config fits with {fmt_gb(headroom)} headroom.")

    print()


if __name__ == "__main__":
    main()
