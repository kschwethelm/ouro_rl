"""GRPO / CISPO loss computation: advantages, log-probs, surrogate objectives + KL.

References:
    GRPO: DeepSeekMath (arXiv:2402.03300) — clipped surrogate.
    CISPO: ScaleRL (arXiv:2510.13786) / MiniMax-M1 — truncated IS policy gradient.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_advantages(
    rewards: torch.Tensor,
    scale_rewards: str = "group",
    batch_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Group-relative advantage normalization.

    Args:
        rewards: (num_prompts, rollouts_per_prompt) binary rewards.
        scale_rewards: Normalization strategy:
            "group" — subtract group mean, divide by group std (original GRPO).
            "none"  — subtract group mean only; avoids question-level difficulty bias
                       (Understanding R1-Zero-Like Training, arXiv:2503.20783).
            "batch" — subtract group mean, divide by *batch* std; more robust shaping
                       (Tricks or Traps? arXiv:2508.08221).
        batch_std: Pre-computed batch std (e.g. synced across ranks for distributed).
            Only used when scale_rewards="batch". Falls back to local std if None.

    Returns:
        advantages: same shape, normalized within each prompt group.
            When scale_rewards="group", groups with zero std get zero advantage.
    """
    mean = rewards.mean(dim=1, keepdim=True)

    if scale_rewards == "none":
        return rewards - mean
    elif scale_rewards == "batch":
        if batch_std is None:
            batch_std = rewards.std()
        return torch.where(batch_std > 0, (rewards - mean) / (batch_std + 1e-8), torch.zeros_like(rewards))
    else:  # "group" (default, original behavior)
        std = rewards.std(dim=1, keepdim=True)
        return torch.where(std > 0, (rewards - mean) / (std + 1e-8), torch.zeros_like(rewards))


@torch.no_grad()
def compute_log_probs_batch(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_indices: torch.Tensor,
    micro_batch_size: int = 4,
    **model_kwargs,
) -> torch.Tensor:
    """Compute per-token log-probs for response tokens only.

    Processes in micro-batches to control memory.

    Args:
        model: HF causal LM (Ouro returns CausalLMOutputWithPast).
        input_ids: (batch, seq_len) full prompt+response token ids.
        attention_mask: (batch, seq_len).
        response_start_indices: (batch,) index where response tokens begin.
        micro_batch_size: Forward pass batch size.
        **model_kwargs: Extra kwargs forwarded to model.forward().

    Returns:
        token_log_probs: (batch, seq_len) with zeros for prompt positions.
    """
    batch_size, seq_len = input_ids.shape
    all_log_probs = torch.zeros(batch_size, seq_len, device=input_ids.device)

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        mb_ids = input_ids[start:end]
        mb_mask = attention_mask[start:end]

        outputs = model(input_ids=mb_ids, attention_mask=mb_mask, **model_kwargs)
        logits = outputs.logits  # (mb, seq_len, vocab)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = mb_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (mb, seq_len-1)

        # Mask out prompt tokens: only keep response positions.
        # response_start_indices[i] is the position of the first response token.
        # In the shifted view, response log-probs start at index response_start_indices[i] - 1.
        for j, idx in enumerate(response_start_indices[start:end]):
            resp_start = max(idx.item() - 1, 0)
            all_log_probs[start + j, resp_start : seq_len - 1] = token_lp[j, resp_start:]

    return all_log_probs


def compute_log_probs_with_grad(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_indices: torch.Tensor,
    **model_kwargs,
) -> torch.Tensor:
    """Same as compute_log_probs_batch but retains gradients for policy training.

    Processes entire batch at once (caller handles micro-batching).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Build response mask (same logic as above but batched).
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len - 1, device=input_ids.device).unsqueeze(0)
    # shifted: response_start - 1 is first response log-prob position
    resp_starts_shifted = (response_start_indices - 1).clamp(min=0).unsqueeze(1)
    response_mask = positions >= resp_starts_shifted  # (batch, seq_len-1)

    masked_lp = token_lp * response_mask
    # Pad to full seq_len (prepend a zero column) for consistent indexing.
    return F.pad(masked_lp, (1, 0), value=0.0)


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.0,
    ref_log_probs: torch.Tensor | None = None,
    token_level_loss: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO clipped surrogate loss with optional KL penalty.

    All log-prob tensors are (batch, seq_len) with zeros at non-response positions.
    advantages is (batch,) — one scalar per sequence.
    response_mask is (batch, seq_len) — 1.0 for response tokens.

    When old_log_probs is None (num_iterations == 1), the ratio is 1.0 everywhere
    and clipping is skipped — equivalent to vanilla policy gradient.

    When kl_coeff=0 (default), the KL term is skipped entirely and ref_log_probs
    is not required. This follows DAPO / Open-Reasoner-Zero / R1-Zero findings
    that KL regularization is unnecessary with verifiable rewards.

    Args:
        token_level_loss: If True (default), average over all response tokens across
            the batch (TRL default, avoids response-level length bias per
            arXiv:2503.20783). If False, average per-sequence then over batch
            (original GRPO formulation).

    Returns:
        loss: scalar, the GRPO objective to minimize.
        metrics: dict with surrogate_loss, mean_ratio, clip_ratio, and optionally kl.
    """
    adv = advantages.unsqueeze(1)  # (batch,) → (batch, 1) for broadcasting

    if old_log_probs is not None:  # noqa: SIM108
        # Per-token policy ratio in log-space.
        log_ratio = policy_log_probs - old_log_probs  # (batch, seq_len)
    else:
        # num_iterations=1 → old == policy, so ratio evaluates to 1.0 and clipping
        # is a no-op. We still need the gradient path through policy_log_probs,
        # so compute ratio as exp(log_π - log_π.detach()) which is 1.0 in value
        # but has ∂ratio/∂θ = ∂log_π/∂θ (REINFORCE gradient).
        log_ratio = policy_log_probs - policy_log_probs.detach()

    ratio = torch.exp(log_ratio.clamp(-10, 10))

    # Clipped surrogate.
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    token_surrogate = torch.min(surr1, surr2)

    total_response_tokens = response_mask.sum().clamp(min=1)

    if token_level_loss:
        # TRL default: flat average over all response tokens (avoids length bias).
        surrogate_loss = -(token_surrogate * response_mask).sum() / total_response_tokens
    else:
        # Original GRPO: per-sequence average then batch average.
        response_lengths = response_mask.sum(dim=1).clamp(min=1)
        seq_surrogate = (token_surrogate * response_mask).sum(dim=1) / response_lengths
        surrogate_loss = -seq_surrogate.mean()

    loss = surrogate_loss

    metrics: dict[str, float] = {"surrogate_loss": surrogate_loss.item()}

    is_clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)) & response_mask.bool()
    metrics["mean_ratio"] = ratio[response_mask.bool()].mean().item() if response_mask.any() else 1.0
    metrics["clip_ratio"] = is_clipped.sum().float().item() / total_response_tokens.item()

    # Optional KL penalty (policy vs reference).
    if kl_coeff > 0:
        assert ref_log_probs is not None, "ref_log_probs required when kl_coeff > 0"
        token_kl = policy_log_probs - ref_log_probs
        if token_level_loss:
            kl = (token_kl * response_mask).sum() / total_response_tokens
        else:
            response_lengths = response_mask.sum(dim=1).clamp(min=1)
            kl = ((token_kl * response_mask).sum(dim=1) / response_lengths).mean()
        loss = loss + kl_coeff * kl
        metrics["kl"] = kl.item()

    return loss, metrics


def cispo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    truncation_max: float = 5.0,
    kl_coeff: float = 0.0,
    ref_log_probs: torch.Tensor | None = None,
    token_level_loss: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute CISPO loss: truncated IS-weighted vanilla policy gradient.

    Key difference from GRPO: gradient flows ONLY through log_pi (the IS ratio
    is stop-gradiented). Uses one-sided truncation (cap at truncation_max)
    instead of symmetric clipping.

    Loss = -sg(min(ρ, truncation_max)) * advantage * log_π(y|x)

    When old_log_probs is None (num_iterations == 1), the IS ratio is 1.0
    everywhere, so CISPO reduces to vanilla REINFORCE — identical to GRPO.

    Args:
        policy_log_probs: (batch, seq_len) per-token log-probs with grad.
        old_log_probs: (batch, seq_len) frozen snapshot, or None for num_iterations=1.
        advantages: (batch,) per-sequence advantage.
        response_mask: (batch, seq_len) 1.0 for response tokens.
        truncation_max: IS ratio cap (default 5.0). Insensitive to choice in {4, 5, 8}.
        kl_coeff: KL penalty coefficient (0.0 to disable).
        ref_log_probs: (batch, seq_len) reference model log-probs (required if kl_coeff > 0).
        token_level_loss: If True, flat average over all response tokens.

    Returns:
        loss: scalar to minimize.
        metrics: dict with surrogate_loss, mean_ratio, truncation_ratio, and optionally kl.
    """
    adv = advantages.unsqueeze(1)  # (batch,) → (batch, 1)

    # IS ratio (fully stop-gradiented).
    if old_log_probs is not None:  # noqa: SIM108
        log_ratio = policy_log_probs.detach() - old_log_probs
    else:
        # num_iterations=1: ratio is 1.0 everywhere.
        log_ratio = torch.zeros_like(policy_log_probs)

    ratio = torch.exp(log_ratio.clamp(-10, 10))
    truncated_ratio = torch.clamp(ratio, max=truncation_max).detach()

    # CISPO objective: sg(min(ρ, ε_max)) * advantage * log_π
    token_objective = truncated_ratio * adv * policy_log_probs

    total_response_tokens = response_mask.sum().clamp(min=1)

    if token_level_loss:
        surrogate_loss = -(token_objective * response_mask).sum() / total_response_tokens
    else:
        response_lengths = response_mask.sum(dim=1).clamp(min=1)
        seq_objective = (token_objective * response_mask).sum(dim=1) / response_lengths
        surrogate_loss = -seq_objective.mean()

    loss = surrogate_loss

    metrics: dict[str, float] = {"surrogate_loss": surrogate_loss.item()}

    is_truncated = (ratio > truncation_max) & response_mask.bool()
    metrics["mean_ratio"] = ratio[response_mask.bool()].mean().item() if response_mask.any() else 1.0
    metrics["truncation_ratio"] = is_truncated.sum().float().item() / total_response_tokens.item()

    # Optional KL penalty (same as GRPO).
    if kl_coeff > 0:
        assert ref_log_probs is not None, "ref_log_probs required when kl_coeff > 0"
        token_kl = policy_log_probs - ref_log_probs
        if token_level_loss:
            kl = (token_kl * response_mask).sum() / total_response_tokens
        else:
            response_lengths = response_mask.sum(dim=1).clamp(min=1)
            kl = ((token_kl * response_mask).sum(dim=1) / response_lengths).mean()
        loss = loss + kl_coeff * kl
        metrics["kl"] = kl.item()

    return loss, metrics
