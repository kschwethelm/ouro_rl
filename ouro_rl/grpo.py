"""GRPO loss computation: advantages, log-probs, clipped surrogate + KL.

Reference: DeepSeekMath (arXiv:2402.03300) Group Relative Policy Optimization.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Group-relative advantage normalization.

    Args:
        rewards: (num_prompts, rollouts_per_prompt) binary rewards.

    Returns:
        advantages: same shape, zero-mean unit-variance within each prompt group.
            Groups with zero variance (all same reward) get zero advantage.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    advantages = torch.where(std > 0, (rewards - mean) / (std + 1e-8), torch.zeros_like(rewards))
    return advantages


@torch.no_grad()
def compute_log_probs_batch(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_indices: torch.Tensor,
    micro_batch_size: int = 4,
) -> torch.Tensor:
    """Compute per-token log-probs for response tokens only.

    Processes in micro-batches to control memory.

    Args:
        model: HF causal LM (Ouro returns CausalLMOutputWithPast).
        input_ids: (batch, seq_len) full prompt+response token ids.
        attention_mask: (batch, seq_len).
        response_start_indices: (batch,) index where response tokens begin.
        micro_batch_size: Forward pass batch size.

    Returns:
        token_log_probs: (batch, seq_len) with zeros for prompt positions.
    """
    batch_size, seq_len = input_ids.shape
    all_log_probs = torch.zeros(batch_size, seq_len, device=input_ids.device)

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        mb_ids = input_ids[start:end]
        mb_mask = attention_mask[start:end]

        outputs = model(input_ids=mb_ids, attention_mask=mb_mask)
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
) -> torch.Tensor:
    """Same as compute_log_probs_batch but retains gradients for policy training.

    Processes entire batch at once (caller handles micro-batching).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
    ref_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coeff: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO clipped surrogate loss with KL penalty.

    All log-prob tensors are (batch, seq_len) with zeros at non-response positions.
    advantages is (batch,) — one scalar per sequence.
    response_mask is (batch, seq_len) — 1.0 for response tokens.

    Returns:
        loss: scalar, the GRPO objective to minimize.
        metrics: dict with surrogate_loss, kl, mean_ratio for logging.
    """
    # Per-token policy ratio in log-space.
    log_ratio = policy_log_probs - old_log_probs  # (batch, seq_len)

    # Per-token ratio (clamped for numerical stability).
    ratio = torch.exp(log_ratio.clamp(-10, 10))

    # Per-token clipped surrogate.  advantages is (batch,) → (batch, 1) for broadcasting.
    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    token_surrogate = torch.min(surr1, surr2)

    # Average over response tokens per sequence, then over batch.
    response_lengths = response_mask.sum(dim=1).clamp(min=1)
    seq_surrogate = (token_surrogate * response_mask).sum(dim=1) / response_lengths
    surrogate_loss = -seq_surrogate.mean()

    # Per-token KL divergence (policy vs reference), averaged same way.
    token_kl = policy_log_probs - ref_log_probs
    seq_kl = (token_kl * response_mask).sum(dim=1) / response_lengths
    kl = seq_kl.mean()

    loss = surrogate_loss + kl_coeff * kl

    metrics = {
        "surrogate_loss": surrogate_loss.item(),
        "kl": kl.item(),
        "mean_ratio": ratio[response_mask.bool()].mean().item() if response_mask.any() else 1.0,
    }
    return loss, metrics
