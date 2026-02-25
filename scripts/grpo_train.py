"""GRPO training loop for Ouro-1.4B-Thinking.

Orchestrates: rollout generation → reward scoring → GRPO policy update.
Supports data-parallel training across multiple GPUs.

Usage:
    torchrun --standalone --nproc_per_node=1 scripts/grpo_train.py --smoke-test --no-wandb  # single GPU
    torchrun --standalone --nproc_per_node=2 scripts/grpo_train.py                          # multi-GPU

Architecture:
    - HF Transformers model.generate(): rollout generation using the policy model directly
    - HF Transformers: policy + reference model for log-prob computation + training
    - Multi-GPU: data-parallel generation + manual gradient sync for training
      (no DDP wrapper, compatible with memory-efficient workflows)
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from ouro_rl.data import INTERRUPTION_PHRASE, format_prompt, load_math_train
from ouro_rl.distributed import broadcast_object, get_rank, get_world_size, is_dist, shard_range, sync_gradients, sync_metrics
from ouro_rl.grpo import cispo_loss, compute_advantages, compute_log_probs_batch, compute_log_probs_with_grad, grpo_loss
from ouro_rl.modeling import BOS_TOKEN_ID, CHAT_TEMPLATE, EOS_TOKEN_ID, PAD_TOKEN_ID, OuroForCausalLM
from ouro_rl.reward import score_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Hyperparameters matching RLTT paper Table 6 (scaled for 1.4B on 2xA100)."""

    # Model
    model_name: str = "ByteDance/Ouro-1.4B-Thinking"
    dtype: str = "bfloat16"
    fp32_lm_head: bool = False  # Upcast lm_head matmul to fp32 (ScaleRL fix). Disable on low VRAM.
    max_model_len: int = 6144  # Max sequence length (prompt avg 89, response p75 = 2793)

    # Dataset
    dataset_name: str = "qwedsacf/competition_math"
    system_prompt: str | None = None  # None = no system prompt (see ouro_rl/data.py for alternatives)

    # Training
    num_steps: int = 140
    prompt_batch_size: int = 32
    rollouts_per_prompt: int = 8
    lr: float = 1e-6
    max_grad_norm: float = 0.1
    warmup_steps: int = 0
    weight_decay: float = 0.01

    # Loss
    loss_type: str = "cispo"  # "grpo" (clipped surrogate) or "cispo" (truncated IS policy gradient)
    clip_eps: float = 0.2  # GRPO: symmetric clip range [1-eps, 1+eps]
    truncation_max: float = 5.0  # CISPO: IS ratio cap (insensitive to choice in {4, 5, 8})
    kl_coeff: float = 0.0  # KL not needed with verifiable rewards (DAPO, Open-Reasoner-Zero)
    scale_rewards: str = "batch"  # "batch" (group mean, batch std), "group" (per-group std), "none" (no std)
    num_iterations: int = 2  # μ: number of policy updates per generation batch (ScaleRL uses μ=2)
    token_level_loss: bool = True  # Token-level average (avoids length bias) vs per-sequence average

    # Generation
    max_new_tokens: int = 4608
    temperature: float = 1.0
    top_p: float = 0.7
    enable_thinking: bool = True

    # Interruptions (ScaleRL: force truncated completions to produce an answer)
    enable_interruptions: bool = True
    thinking_budget_min: int = 3072  # min thinking tokens before interruption
    thinking_budget_max: int = 4096  # max thinking tokens (randomized per step)
    answer_budget: int = 512  # max tokens for answer after interruption

    # Memory
    generation_batch_size: int = 2  # prompts per model.generate() call (×rollouts_per_prompt sequences)
    log_prob_micro_batch: int = 4  # micro-batch for log-prob forward passes
    train_micro_batch: int = 2  # micro-batch for training forward/backward

    # Logging & checkpointing
    output_dir: str = "outputs/grpo"
    log_every: int = 1
    save_every: int = 10
    wandb_project: str = "ouro-rl"
    wandb_run_name: str | None = None
    wandb_enabled: bool = True

    # Reproducibility
    seed: int = 42

    # Smoke test overrides
    smoke_test: bool = False

    def __post_init__(self) -> None:
        if self.smoke_test:
            self.num_steps = 3
            self.prompt_batch_size = 4
            self.rollouts_per_prompt = 4
            self.save_every = 1
            self.max_new_tokens = 256
            self.max_model_len = 512
            self.log_prob_micro_batch = 1
            self.train_micro_batch = 1
            if self.enable_interruptions:
                self.thinking_budget_min = 128
                self.thinking_budget_max = 192
                self.answer_budget = 64

    # Derived
    @property
    def total_rollouts_per_step(self) -> int:
        return self.prompt_batch_size * self.rollouts_per_prompt


# ---------------------------------------------------------------------------
# Rollout generation (HF model.generate)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_rollouts(
    model: OuroForCausalLM,
    prompt_token_ids: list[list[int]],
    *,
    num_rollouts: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    return_logprobs: bool = False,
    batch_size: int = 0,
) -> tuple[list[list[list[int]]], list[list[list[float]]] | None]:
    """Generate rollouts using HF model.generate().

    Uses the policy model directly — no separate vLLM process, no weight sync,
    no memory leaks. ``num_rollouts`` controls how many completions per prompt
    (equivalent to vLLM's ``n`` parameter).

    Args:
        model: The policy model (already on GPU).
        prompt_token_ids: Pre-tokenized prompts as lists of token IDs.
        num_rollouts: Number of completions per prompt.
        max_new_tokens: Maximum response tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        return_logprobs: If True, capture per-token log-probs from generation
            (useful as old_log_probs when num_iterations > 1).
        batch_size: Number of prompts per model.generate() call. Each call
            produces batch_size × num_rollouts sequences. 0 = all at once.

    Returns:
        (response_ids, generation_logprobs) where:
        - response_ids[prompt_idx][rollout_idx] → list of token IDs
        - generation_logprobs[prompt_idx][rollout_idx] → list of per-token
          log-probs (same length as the corresponding response), or None
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    n_prompts = len(prompt_token_ids)
    if batch_size <= 0:
        batch_size = n_prompts

    all_response_ids: list[list[list[int]]] = []
    all_logprobs: list[list[list[float]]] | None = [] if return_logprobs else None

    for chunk_start in range(0, n_prompts, batch_size):
        chunk_end = min(chunk_start + batch_size, n_prompts)
        chunk_prompts = prompt_token_ids[chunk_start:chunk_end]
        chunk_size = len(chunk_prompts)

        # Left-pad this chunk for batched generation.
        max_prompt_len = max(len(ids) for ids in chunk_prompts)
        input_ids = torch.full((chunk_size, max_prompt_len), PAD_TOKEN_ID, dtype=torch.long, device=device)
        attention_mask = torch.zeros(chunk_size, max_prompt_len, dtype=torch.long, device=device)
        for i, ids in enumerate(chunk_prompts):
            pad_len = max_prompt_len - len(ids)
            input_ids[i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, pad_len:] = 1

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_rollouts,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            output_scores=return_logprobs,
            return_dict_in_generate=True,
        )

        # Extract response token IDs (strip prompt, strip trailing pad).
        full_ids = outputs.sequences  # (chunk_size * num_rollouts, max_prompt_len + max_gen_len)
        for pi in range(chunk_size):
            rollouts = []
            for ri in range(num_rollouts):
                seq_idx = pi * num_rollouts + ri
                resp = full_ids[seq_idx, max_prompt_len:].tolist()
                while resp and resp[-1] == PAD_TOKEN_ID:
                    resp.pop()
                rollouts.append(resp)
            all_response_ids.append(rollouts)

        # Extract per-token log-probs if requested.
        if return_logprobs and outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)  # (total_seqs, num_steps, vocab)
            log_probs = F.log_softmax(scores, dim=-1)
            generated_tokens = full_ids[:, max_prompt_len:]
            num_steps = scores.shape[1]
            generated_tokens = generated_tokens[:, :num_steps]
            token_log_probs = log_probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)

            for pi in range(chunk_size):
                rollouts_lp = []
                for ri in range(num_rollouts):
                    seq_idx = pi * num_rollouts + ri
                    resp_len = len(all_response_ids[chunk_start + pi][ri])
                    rollouts_lp.append(token_log_probs[seq_idx, :resp_len].tolist())
                all_logprobs.append(rollouts_lp)

    if was_training:
        model.train()

    return all_response_ids, all_logprobs if all_logprobs else None


def align_generation_logprobs(
    flat_gen_logprobs: list[list[float]],
    response_start_indices: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Align per-sequence generation log-probs into the padded training tensor format.

    Returns a (batch, seq_len) tensor matching the output format of
    ``compute_log_probs_batch``: log-probs at response positions, zeros elsewhere.
    """
    batch_size = len(flat_gen_logprobs)
    aligned = torch.zeros(batch_size, seq_len, device=device)
    for i, lps in enumerate(flat_gen_logprobs):
        start = response_start_indices[i].item()
        end = min(start + len(lps), seq_len)
        n = end - start
        aligned[i, start:end] = torch.tensor(lps[:n], device=device)
    return aligned


def find_truncated_completions(
    rollout_response_ids: list[list[list[int]]],
    eos_token_id: int,
    think_close_id: int,
) -> list[tuple[int, int]]:
    """Identify completions that were truncated mid-thinking.

    A completion is truncated if it has neither EOS nor </think>.
    These need an interruption phrase + second generation to produce an answer.

    Returns:
        List of (prompt_idx, rollout_idx) tuples for truncated completions.
    """
    needs_interruption: list[tuple[int, int]] = []
    for i, prompt_rollouts in enumerate(rollout_response_ids):
        for j, resp_ids in enumerate(prompt_rollouts):
            has_eos = resp_ids and resp_ids[-1] == eos_token_id
            has_think_close = think_close_id in resp_ids
            if not has_eos and not has_think_close:
                needs_interruption.append((i, j))
    return needs_interruption


def stitch_interruptions(
    rollout_response_ids: list[list[list[int]]],
    needs_interruption: list[tuple[int, int]],
    interruption_token_ids: list[int],
    phase2_responses: list[list[list[int]]],
) -> None:
    """Stitch interrupted completions: thinking + interruption phrase + answer.

    Modifies rollout_response_ids in-place.

    Args:
        rollout_response_ids: [prompt_idx][rollout_idx] → token ID list.
        needs_interruption: (prompt_idx, rollout_idx) pairs from find_truncated_completions.
        interruption_token_ids: Tokenized INTERRUPTION_PHRASE.
        phase2_responses: [idx][0] → answer token IDs (n=1 per interrupted completion).
    """
    for idx, (pi, ri) in enumerate(needs_interruption):
        rollout_response_ids[pi][ri] = rollout_response_ids[pi][ri] + interruption_token_ids + phase2_responses[idx][0]


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def pad_token_id_pairs(
    prompt_ids_list: list[list[int]],
    response_ids_list: list[list[int]],
    max_length: int,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Left-pad prompt+response token ID pairs into batched tensors.

    Returns input_ids, attention_mask, response_start_indices, response_mask.
    """
    all_input_ids = []
    all_response_starts = []

    for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list):
        # Truncate response if prompt+response exceeds max_length.
        max_resp_len = max_length - len(prompt_ids)
        if max_resp_len <= 0:
            # Prompt itself is too long — truncate prompt from the left.
            prompt_ids = prompt_ids[-(max_length - 1) :]
            response_ids = response_ids[:1]
        else:
            response_ids = response_ids[:max_resp_len]

        combined = prompt_ids + response_ids
        all_input_ids.append(combined)
        all_response_starts.append(len(prompt_ids))

    # Pad to longest sequence in batch (left-padding for causal LM).
    max_len = max(len(ids) for ids in all_input_ids)
    padded_ids = []
    padded_mask = []
    adjusted_starts = []

    for ids, resp_start in zip(all_input_ids, all_response_starts):
        pad_len = max_len - len(ids)
        padded_ids.append([pad_token_id] * pad_len + ids)
        padded_mask.append([0] * pad_len + [1] * len(ids))
        adjusted_starts.append(resp_start + pad_len)

    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    attention_mask = torch.tensor(padded_mask, dtype=torch.long)
    response_start_indices = torch.tensor(adjusted_starts, dtype=torch.long)

    # Response mask: 1 for response token positions, 0 for prompt/padding.
    seq_positions = torch.arange(max_len).unsqueeze(0)
    response_mask = (seq_positions >= response_start_indices.unsqueeze(1)).float()
    response_mask = response_mask * attention_mask.float()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_start_indices": response_start_indices,
        "response_mask": response_mask,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    policy_model: OuroForCausalLM,
    ref_model: OuroForCausalLM | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenized: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    config: GRPOConfig,
    old_log_probs: torch.Tensor | None = None,
) -> dict[str, float]:
    """Single GRPO gradient update with micro-batching.

    When config.kl_coeff == 0, ref_model can be None and ref log-probs are skipped.

    Args:
        old_log_probs: Pre-computed log-probs from generation time (frozen π_old).
            When provided and num_iterations > 1, skips the forward pass to compute
            them. When None and num_iterations > 1, computes from policy_model.

    Returns aggregated metrics dict.
    """
    device = next(policy_model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    response_start_indices = tokenized["response_start_indices"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    advantages = advantages.to(device)

    batch_size = input_ids.shape[0]

    # Compute reference log-probs only when KL is enabled.
    ref_log_probs = None
    if config.kl_coeff > 0:
        assert ref_model is not None, "ref_model required when kl_coeff > 0"
        ref_log_probs = compute_log_probs_batch(
            ref_model,
            input_ids,
            attention_mask,
            response_start_indices,
            micro_batch_size=config.log_prob_micro_batch,
            use_early_exit_gate=False,
        )

    # Compute old log-probs only when num_iterations > 1 (the frozen snapshot anchors the
    # clipping ratio across multiple optimization passes over the same rollouts).
    # With num_iterations == 1, old == policy so ratio is always 1.0 and clipping is a no-op.
    if old_log_probs is None and config.num_iterations > 1:
        old_log_probs = compute_log_probs_batch(
            policy_model,
            input_ids,
            attention_mask,
            response_start_indices,
            micro_batch_size=config.log_prob_micro_batch,
            use_early_exit_gate=False,
        )

    # Micro-batched forward/backward for policy gradient.
    optimizer.zero_grad()
    total_metrics: dict[str, float] = {}
    mb_size = config.train_micro_batch
    num_micro_batches = (batch_size + mb_size - 1) // mb_size

    for mb_start in range(0, batch_size, mb_size):
        mb_end = min(mb_start + mb_size, batch_size)
        mb_slice = slice(mb_start, mb_end)

        policy_lp = compute_log_probs_with_grad(
            policy_model,
            input_ids[mb_slice],
            attention_mask[mb_slice],
            response_start_indices[mb_slice],
            use_early_exit_gate=False,
        )

        loss_kwargs = {
            "policy_log_probs": policy_lp,
            "old_log_probs": old_log_probs[mb_slice] if old_log_probs is not None else None,
            "advantages": advantages[mb_slice],
            "response_mask": response_mask[mb_slice],
            "kl_coeff": config.kl_coeff,
            "ref_log_probs": ref_log_probs[mb_slice] if ref_log_probs is not None else None,
            "token_level_loss": config.token_level_loss,
        }
        if config.loss_type == "cispo":
            loss, metrics = cispo_loss(**loss_kwargs, truncation_max=config.truncation_max)
        else:
            loss, metrics = grpo_loss(**loss_kwargs, clip_eps=config.clip_eps)

        # Scale loss by number of micro-batches for correct gradient averaging.
        scaled_loss = loss / num_micro_batches
        scaled_loss.backward()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v / num_micro_batches

    # Sync gradients across ranks before clipping.
    sync_gradients(policy_model)

    grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
    optimizer.step()
    scheduler.step()

    total_metrics["grad_norm"] = grad_norm.item()
    total_metrics["lr"] = scheduler.get_last_lr()[0]
    return total_metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config: GRPOConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Distributed setup — always use torchrun (even single-GPU: torchrun --standalone --nproc_per_node=1).
    dist.init_process_group(backend="nccl")
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    assert config.prompt_batch_size % world_size == 0, (
        f"prompt_batch_size ({config.prompt_batch_size}) must be divisible by world_size ({world_size})"
    )

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config (rank 0 only).
    if rank == 0:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    # Wandb setup (rank 0 only).
    if config.wandb_enabled and rank == 0:
        import wandb

        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.__dict__)

    # Load tokenizer with fixed token IDs and chat template.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.bos_token_id = BOS_TOKEN_ID
    tokenizer.eos_token_id = EOS_TOKEN_ID
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.chat_template = CHAT_TEMPLATE

    # Load dataset.
    if rank == 0:
        logger.info("Loading MATH dataset: %s", config.dataset_name)
    dataset = load_math_train(config.dataset_name)
    problems = dataset["problem"]
    solutions = dataset["solution"]
    if rank == 0:
        logger.info("Loaded %d problems", len(problems))

    # Load policy + reference models for training.
    torch_dtype = getattr(torch, config.dtype)
    if rank == 0:
        logger.info("Loading policy model: %s", config.model_name)
    policy_model = OuroForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    if config.fp32_lm_head:
        policy_model.enable_fp32_lm_head()
    policy_model.train()

    ref_model = None
    if config.kl_coeff > 0:
        if rank == 0:
            logger.info("Loading reference model: %s", config.model_name)
        ref_model = OuroForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )
        if config.fp32_lm_head:
            ref_model.enable_fp32_lm_head()
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        if rank == 0:
            logger.info("KL disabled (kl_coeff=0) — skipping reference model")

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    if rank == 0:
        logger.info("Starting GRPO training: %d steps, %d GPU(s)", config.num_steps, world_size)

    for step in range(1, config.num_steps + 1):
        step_start = time.time()

        # --- 1. Sample prompts (rank 0 → broadcast) ---
        indices = random.sample(range(len(problems)), config.prompt_batch_size) if rank == 0 else None
        indices = broadcast_object(indices)
        batch_problems = [problems[i] for i in indices]
        batch_solutions = [solutions[i] for i in indices]

        prompts = [
            format_prompt(p, tokenizer, system_prompt=config.system_prompt, enable_thinking=config.enable_thinking)
            for p in batch_problems
        ]
        prompt_token_ids = [tokenizer.encode(p) for p in prompts]

        # --- 2. Generate rollouts (data-parallel, HF model.generate) ---
        if rank == 0:
            logger.info("[Step %d/%d] Generating %d rollouts...", step, config.num_steps, config.total_rollouts_per_step)

        gen_start = time.time()
        n_interrupted = 0

        # Split prompts across ranks — each rank generates its subset independently.
        my_start, my_end = shard_range(len(prompt_token_ids), rank, world_size)
        my_prompt_token_ids = prompt_token_ids[my_start:my_end]
        my_num_prompts = len(my_prompt_token_ids)

        # Capture generation logprobs when num_iterations > 1 (reused as old_log_probs
        # in train_step, saving a full forward pass per training step).
        capture_logprobs = config.num_iterations > 1
        my_gen_logprobs: list[list[list[float]]] | None = None

        if config.enable_interruptions:
            # Two-phase generation: thinking → interruption → answer.
            # Sync thinking budget so all ranks use the same value.
            thinking_budget = random.randint(config.thinking_budget_min, config.thinking_budget_max) if rank == 0 else 0
            thinking_budget = broadcast_object(thinking_budget)

            # Phase 1: generate thinking tokens (can't capture logprobs — stitching invalidates them).
            my_rollout_ids, _ = generate_rollouts(
                policy_model,
                my_prompt_token_ids,
                num_rollouts=config.rollouts_per_prompt,
                max_new_tokens=thinking_budget,
                temperature=config.temperature,
                top_p=config.top_p,
                batch_size=config.generation_batch_size,
            )

            # Identify truncated completions on this rank's subset.
            think_close_id = tokenizer.convert_tokens_to_ids("</think>")
            interruption_token_ids = tokenizer.encode(INTERRUPTION_PHRASE, add_special_tokens=False)

            needs_interruption = find_truncated_completions(my_rollout_ids, EOS_TOKEN_ID, think_close_id)
            my_n_interrupted = len(needs_interruption)

            if my_n_interrupted > 0:
                # Build phase 2 prompts: original prompt + thinking + interruption phrase.
                phase2_prompt_ids = [
                    my_prompt_token_ids[pi] + my_rollout_ids[pi][ri] + interruption_token_ids for pi, ri in needs_interruption
                ]

                phase2_responses, _ = generate_rollouts(
                    policy_model,
                    phase2_prompt_ids,
                    num_rollouts=1,
                    max_new_tokens=config.answer_budget,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    batch_size=config.generation_batch_size,
                )

                stitch_interruptions(my_rollout_ids, needs_interruption, interruption_token_ids, phase2_responses)

            # Sum interrupted counts across ranks for logging.
            n_interrupted_t = torch.tensor([my_n_interrupted], device=device, dtype=torch.long)
            if is_dist():
                dist.all_reduce(n_interrupted_t, op=dist.ReduceOp.SUM)
            n_interrupted = int(n_interrupted_t.item())

            if rank == 0:
                logger.info(
                    "[Step %d/%d] Generation complete: %d/%d interrupted",
                    step,
                    config.num_steps,
                    n_interrupted,
                    config.total_rollouts_per_step,
                )
        else:
            my_rollout_ids, my_gen_logprobs = generate_rollouts(
                policy_model,
                my_prompt_token_ids,
                num_rollouts=config.rollouts_per_prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                return_logprobs=capture_logprobs,
                batch_size=config.generation_batch_size,
            )

        gen_time = time.time() - gen_start

        # Decode rollout token IDs to text.
        rollout_texts = [[tokenizer.decode(ids) for ids in prompt_rollouts] for prompt_rollouts in my_rollout_ids]

        # Completion length stats (local, synced across ranks for logging).
        all_response_lengths = [len(ids) for prompt_rollouts in my_rollout_ids for ids in prompt_rollouts]
        mean_completion_len = sum(all_response_lengths) / len(all_response_lengths)
        max_completion_len = max(all_response_lengths)
        min_completion_len = min(all_response_lengths)
        clipped_ratio = sum(
            1 for prompt_rollouts in my_rollout_ids for ids in prompt_rollouts if not ids or ids[-1] != EOS_TOKEN_ID
        ) / len(all_response_lengths)
        completion_log_data: dict[str, float] = {
            "completions/mean_length": mean_completion_len,
            "completions/clipped_ratio": clipped_ratio,
        }
        if config.enable_interruptions:
            completion_log_data["completions/interrupted_ratio"] = n_interrupted / config.total_rollouts_per_step
            completion_log_data["completions/thinking_budget"] = thinking_budget
        completion_log_data = sync_metrics(completion_log_data, device)

        # Max/min need dedicated reduce ops (AVG is wrong for extrema).
        if is_dist():
            max_t = torch.tensor(max_completion_len, dtype=torch.float32, device=device)
            min_t = torch.tensor(min_completion_len, dtype=torch.float32, device=device)
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
            max_completion_len = int(max_t.item())
            min_completion_len = int(min_t.item())
        completion_log_data["completions/max_length"] = max_completion_len
        completion_log_data["completions/min_length"] = min_completion_len

        # --- 3. Score rollouts (each rank scores its own subset) ---
        rewards_list: list[list[float]] = []
        for prompt_idx in range(my_num_prompts):
            group_rewards = []
            for rollout_idx in range(config.rollouts_per_prompt):
                r = score_answer(rollout_texts[prompt_idx][rollout_idx], batch_solutions[my_start + prompt_idx])
                group_rewards.append(r)
            rewards_list.append(group_rewards)

        rewards = torch.tensor(rewards_list, dtype=torch.float32)  # (my_num_prompts, rollouts)
        mean_reward = rewards.mean().item()
        fraction_correct = (rewards > 0).float().mean().item()
        frac_reward_zero_std = (rewards.std(dim=1) == 0).float().mean().item()

        # Sync reward stats across ranks for consistent logging.
        reward_stats = sync_metrics(
            {"mean_reward": mean_reward, "fraction_correct": fraction_correct, "frac_reward_zero_std": frac_reward_zero_std},
            device,
        )
        mean_reward = reward_stats["mean_reward"]
        fraction_correct = reward_stats["fraction_correct"]
        frac_reward_zero_std = reward_stats["frac_reward_zero_std"]

        # --- 4. Compute advantages (group mean is local, batch std is global) ---
        batch_std = None
        if config.scale_rewards == "batch" and is_dist():
            stats = torch.stack([rewards.sum(), (rewards**2).sum(), torch.tensor(float(rewards.numel()))])
            stats = stats.to(device)
            dist.all_reduce(stats)
            global_mean = stats[0] / stats[2]
            batch_std = ((stats[1] / stats[2]) - global_mean**2).clamp(min=0).sqrt().cpu()
        advantages = compute_advantages(rewards, scale_rewards=config.scale_rewards, batch_std=batch_std)

        # Flatten local data for training.
        flat_advantages = advantages.reshape(-1)
        flat_prompt_ids = [my_prompt_token_ids[i] for i in range(my_num_prompts) for _ in range(config.rollouts_per_prompt)]
        flat_response_ids = [my_rollout_ids[i][j] for i in range(my_num_prompts) for j in range(config.rollouts_per_prompt)]

        # --- 5. Skip step if no learning signal (coordinated across all ranks) ---
        local_has_signal = float((flat_advantages != 0).any())
        signal_t = torch.tensor([local_has_signal], device=device)
        if is_dist():
            dist.all_reduce(signal_t, op=dist.ReduceOp.MAX)
        if signal_t.item() == 0:
            step_time = time.time() - step_start
            if rank == 0:
                logger.info(
                    "[Step %d/%d] All-zero advantages (reward=%.3f), skipping update.",
                    step,
                    config.num_steps,
                    mean_reward,
                )
            if config.wandb_enabled and rank == 0:
                import wandb

                wandb.log(
                    {
                        "step": step,
                        "reward/mean": mean_reward,
                        "reward/fraction_correct": fraction_correct,
                        "reward/frac_zero_std": frac_reward_zero_std,
                        **completion_log_data,
                        "time/step_total": step_time,
                        "time/generation": gen_time,
                        "time/training": 0.0,
                        "train/surrogate_loss": 0.0,
                        "train/mean_ratio": 1.0,
                        "train/clip_ratio": 0.0,
                        "train/grad_norm": 0.0,
                        "train/lr": scheduler.get_last_lr()[0],
                        "skipped": True,
                    }
                )
            continue

        # Pad local batch into tensors (each rank already has its own shard from generation).
        max_length = config.max_model_len
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        tokenized = pad_token_id_pairs(flat_prompt_ids, flat_response_ids, max_length, pad_token_id=pad_id)

        # Build old_log_probs from generation if available (saves a forward pass per train_step).
        old_log_probs = None
        if my_gen_logprobs is not None:
            flat_gen_logprobs = [
                my_gen_logprobs[i][j] for i in range(my_num_prompts) for j in range(config.rollouts_per_prompt)
            ]
            old_log_probs = align_generation_logprobs(
                flat_gen_logprobs,
                tokenized["response_start_indices"],
                tokenized["input_ids"].shape[1],
                device,
            )

        # --- 6. GRPO update (μ iterations per generation batch) ---
        train_start = time.time()
        for _iteration in range(config.num_iterations):
            metrics = train_step(
                policy_model,
                ref_model,
                optimizer,
                scheduler,
                tokenized,
                flat_advantages,
                config,
                old_log_probs=old_log_probs,
            )
        train_time = time.time() - train_start

        step_time = time.time() - step_start

        # --- 7. Aggregate metrics across ranks ---
        metrics = sync_metrics(metrics, device)

        # --- 8. Logging (rank 0 only) ---
        log_data = {
            "step": step,
            "reward/mean": mean_reward,
            "reward/fraction_correct": fraction_correct,
            "reward/frac_zero_std": frac_reward_zero_std,
            **completion_log_data,
            "time/step_total": step_time,
            "time/generation": gen_time,
            "time/training": train_time,
            **{f"train/{k}": v for k, v in metrics.items()},
        }

        if step % config.log_every == 0 and rank == 0:
            kl_str = f" kl={metrics['kl']:.4f}" if "kl" in metrics else ""
            logger.info(
                "[Step %d/%d] reward=%.3f correct=%.1f%% len=%.0f grad=%.4f surr=%.4f%s"
                " ratio=%.3f clip=%.3f trunc=%.1f%% gen=%.1fs train=%.1fs",
                step,
                config.num_steps,
                mean_reward,
                fraction_correct * 100,
                completion_log_data["completions/mean_length"],
                metrics.get("grad_norm", 0),
                metrics.get("surrogate_loss", 0),
                kl_str,
                metrics.get("mean_ratio", 1),
                metrics.get("clip_ratio", metrics.get("truncation_ratio", 0)),
                completion_log_data["completions/clipped_ratio"] * 100,
                gen_time,
                train_time,
            )

        if config.wandb_enabled and rank == 0:
            import wandb

            wandb.log(log_data)

        # --- 9. Checkpoint (rank 0 saves, all ranks wait) ---
        if step % config.save_every == 0 or step == config.num_steps:
            ckpt_dir = output_dir / f"step_{step:04d}"
            if rank == 0:
                logger.info("Saving checkpoint to %s", ckpt_dir)
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
            if is_dist():
                dist.barrier()  # Ensure checkpoint is written before any rank reads it.

    # Final save.
    final_dir = output_dir / "final"
    if rank == 0:
        policy_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info("Training complete. Final model saved to %s", final_dir)

    if config.wandb_enabled and rank == 0:
        import wandb

        wandb.finish()

    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser(description="GRPO training for Ouro")
    p.add_argument("--model", dest="model_name")
    p.add_argument("--dataset", dest="dataset_name")
    p.add_argument("--num-steps", type=int)
    p.add_argument("--prompt-batch-size", type=int)
    p.add_argument("--rollouts-per-prompt", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--kl-coeff", type=float)
    p.add_argument("--scale-rewards", choices=["batch", "group", "none"])
    p.add_argument("--num-iterations", type=int)
    p.add_argument("--loss-type", choices=["grpo", "cispo"])
    p.add_argument("--clip-eps", type=float)
    p.add_argument("--truncation-max", type=float)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--fp32-lm-head", dest="fp32_lm_head", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--temperature", type=float)
    p.add_argument("--enable-interruptions", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--thinking-budget-min", type=int)
    p.add_argument("--thinking-budget-max", type=int)
    p.add_argument("--answer-budget", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--save-every", type=int)
    p.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb", dest="wandb_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb-run-name", dest="wandb_run_name")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    return GRPOConfig(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main(parse_args())
