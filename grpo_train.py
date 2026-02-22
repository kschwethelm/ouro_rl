"""GRPO training loop for Ouro-1.4B-Thinking.

Orchestrates: vLLM rollout generation → reward scoring → GRPO policy update.

Usage:
    uv run python grpo_train.py
    uv run python grpo_train.py --smoke-test --no-wandb # 3 steps, small batch
    uv run python grpo_train.py --num-steps 140         # full training run

Architecture:
    - vLLM: generates rollouts (created/destroyed each step to free GPU memory)
    - HF Transformers: policy + reference model for log-prob computation + training
    - Weight sync: save checkpoint to disk → vLLM reloads from checkpoint next step
"""

import argparse
import gc
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from ouro_rl.data import CHAT_TEMPLATE, format_prompt, load_math_train
from ouro_rl.grpo import compute_advantages, compute_log_probs_batch, compute_log_probs_with_grad, grpo_loss
from ouro_rl.patches import CORRECT_EOS_TOKEN_ID, patch_ouro, patch_ouro_post_load
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
    max_model_len: int = 4096  # vLLM context window (prompt avg 89, response p75 = 2793)

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

    # GRPO
    clip_eps: float = 0.2
    kl_coeff: float = 0.0  # KL not needed with verifiable rewards (DAPO, Open-Reasoner-Zero)
    scale_rewards: str = "batch"  # "batch" (group mean, batch std), "group" (per-group std), "none" (no std)
    num_iterations: int = 1  # μ: number of policy updates per generation batch
    mask_truncated_completions: bool = False  # Include truncated completions (reward=0 → negative advantage teaches brevity)
    token_level_loss: bool = True  # Token-level average (avoids length bias) vs per-sequence average

    # Generation
    max_new_tokens: int = 3072
    temperature: float = 1.0
    top_p: float = 0.95
    enable_thinking: bool = True

    # Memory
    log_prob_micro_batch: int = 4  # micro-batch for log-prob forward passes
    train_micro_batch: int = 2  # micro-batch for training forward/backward

    # Logging & checkpointing
    output_dir: str = "outputs/grpo"
    log_every: int = 1
    save_every: int = 10
    keep_last_n_checkpoints: int = 2  # Delete older checkpoints to save disk (0 = keep all)
    wandb_project: str = "ouro-rl"
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
            self.max_model_len = 1024

    # Derived
    @property
    def total_rollouts_per_step(self) -> int:
        return self.prompt_batch_size * self.rollouts_per_prompt


# ---------------------------------------------------------------------------
# vLLM rollout generation
# ---------------------------------------------------------------------------


def generate_rollouts(
    model_path: str,
    prompt_token_ids: list[list[int]],
    sampling_params: SamplingParams,
    *,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
) -> list[list[list[int]]]:
    """Generate multiple rollouts per prompt using vLLM.

    Creates a vLLM LLM instance, generates, then destroys it to free GPU memory.
    Number of rollouts per prompt is controlled by sampling_params.n.

    Accepts pre-tokenized prompt_token_ids and skips vLLM's tokenizer entirely
    (skip_tokenizer_init=True). The caller controls tokenization (correct
    bos/eos/pad) and must decode response token IDs externally.

    Whether a response completed naturally (vs hitting max_tokens) can be
    determined by checking if the last token is EOS — the stop token is
    included in token_ids when skip_special_tokens=False.

    Returns:
        response_token_ids: [prompt_idx][rollout_idx] list of token ID lists.
    """
    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=True,  # Simpler, avoids CUDA graph issues with model swapping.
        skip_tokenizer_init=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_token_ids],
        sampling_params,
    )

    # Clean up vLLM completely.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return [[list(out.token_ids) for out in output.outputs] for output in outputs]


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
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenized: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    config: GRPOConfig,
) -> dict[str, float]:
    """Single GRPO gradient update with micro-batching.

    When config.kl_coeff == 0, ref_model can be None and ref log-probs are skipped.

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
        )

    # Compute old log-probs only when num_iterations > 1 (the frozen snapshot anchors the
    # clipping ratio across multiple optimization passes over the same rollouts).
    # With num_iterations == 1, old == policy so ratio is always 1.0 and clipping is a no-op.
    old_log_probs = None
    if config.num_iterations > 1:
        old_log_probs = compute_log_probs_batch(
            policy_model,
            input_ids,
            attention_mask,
            response_start_indices,
            micro_batch_size=config.log_prob_micro_batch,
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
        )

        loss, metrics = grpo_loss(
            policy_log_probs=policy_lp,
            old_log_probs=old_log_probs[mb_slice] if old_log_probs is not None else None,
            advantages=advantages[mb_slice],
            response_mask=response_mask[mb_slice],
            clip_eps=config.clip_eps,
            kl_coeff=config.kl_coeff,
            ref_log_probs=ref_log_probs[mb_slice] if ref_log_probs is not None else None,
            token_level_loss=config.token_level_loss,
        )

        # Scale loss by number of micro-batches for correct gradient averaging.
        scaled_loss = loss / num_micro_batches
        scaled_loss.backward()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v / num_micro_batches

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

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config.
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Wandb setup.
    if config.wandb_enabled:
        import wandb

        wandb.init(project=config.wandb_project, config=config.__dict__)

    # Apply pre-load patches.
    patch_ouro()

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    # Load chat template.
    tokenizer.chat_template = CHAT_TEMPLATE

    # Load dataset.
    logger.info("Loading MATH dataset: %s", config.dataset_name)
    dataset = load_math_train(config.dataset_name)
    problems = dataset["problem"]
    solutions = dataset["solution"]
    logger.info("Loaded %d problems", len(problems))

    # vLLM sampling params.
    # NOTE: stop_token_ids is required because the upstream Ouro model ships
    # with eos_token_id=0 (<|endoftext|>) in both tokenizer and model config.
    # vLLM reads that and never stops on <|im_end|> (id=2), causing every
    # completion to hit max_tokens and be flagged as truncated.
    sampling_params = SamplingParams(
        n=config.rollouts_per_prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        stop_token_ids=[CORRECT_EOS_TOKEN_ID],
        skip_special_tokens=False,  # Keep <think>...</think> for reward parsing.
    )

    # The current model path — starts as the base model, updated after checkpoints.
    current_model_path = config.model_name

    # Load policy + reference models for training.
    torch_dtype = getattr(torch, config.dtype)
    logger.info("Loading policy model: %s", config.model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    patch_ouro_post_load(policy_model, tokenizer)
    policy_model.train()

    ref_model = None
    if config.kl_coeff > 0:
        logger.info("Loading reference model: %s", config.model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="cuda",
        )
        patch_ouro_post_load(ref_model, tokenizer)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
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
    logger.info("Starting GRPO training: %d steps", config.num_steps)

    for step in range(1, config.num_steps + 1):
        step_start = time.time()

        # --- 1. Sample prompts ---
        indices = random.sample(range(len(problems)), config.prompt_batch_size)
        batch_problems = [problems[i] for i in indices]
        batch_solutions = [solutions[i] for i in indices]

        prompts = [
            format_prompt(p, tokenizer, system_prompt=config.system_prompt, enable_thinking=config.enable_thinking)
            for p in batch_problems
        ]
        # Tokenize prompts once with correctly-configured HF tokenizer.
        # vLLM receives token IDs directly, bypassing its internal tokenizer
        # (which has wrong bos/eos/pad from upstream Ouro config).
        prompt_token_ids = [tokenizer.encode(p) for p in prompts]

        # --- 2. Generate rollouts with vLLM ---
        logger.info("[Step %d/%d] Generating %d rollouts...", step, config.num_steps, config.total_rollouts_per_step)

        # Move training models + optimizer state to CPU to free GPU memory for vLLM.
        policy_model.cpu()
        if ref_model is not None:
            ref_model.cpu()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

        gen_start = time.time()
        rollout_response_ids = generate_rollouts(
            current_model_path,
            prompt_token_ids,
            sampling_params,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
        )
        gen_time = time.time() - gen_start

        # Decode response token IDs to text (vLLM skips tokenizer, so we do it here).
        rollout_texts = [[tokenizer.decode(ids) for ids in prompt_rollouts] for prompt_rollouts in rollout_response_ids]

        # Completion length stats (computed early so skipped steps can still log them).
        all_response_lengths = [len(ids) for prompt_rollouts in rollout_response_ids for ids in prompt_rollouts]
        mean_completion_len = sum(all_response_lengths) / len(all_response_lengths)
        max_completion_len = max(all_response_lengths)
        min_completion_len = min(all_response_lengths)
        clipped_ratio = sum(
            1
            for prompt_rollouts in rollout_response_ids
            for ids in prompt_rollouts
            if not ids or ids[-1] != CORRECT_EOS_TOKEN_ID
        ) / len(all_response_lengths)
        completion_log_data = {
            "completions/mean_length": mean_completion_len,
            "completions/max_length": max_completion_len,
            "completions/min_length": min_completion_len,
            "completions/clipped_ratio": clipped_ratio,
        }

        # Move training models + optimizer state back to GPU.
        device = torch.device("cuda")
        policy_model.to(device)
        if ref_model is not None:
            ref_model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # --- 3. Score rollouts ---
        rewards_list: list[list[float]] = []
        for prompt_idx in range(config.prompt_batch_size):
            group_rewards = []
            for rollout_idx in range(config.rollouts_per_prompt):
                r = score_answer(rollout_texts[prompt_idx][rollout_idx], batch_solutions[prompt_idx])
                group_rewards.append(r)
            rewards_list.append(group_rewards)

        rewards = torch.tensor(rewards_list, dtype=torch.float32)  # (num_prompts, rollouts)
        mean_reward = rewards.mean().item()
        fraction_correct = (rewards > 0).float().mean().item()
        frac_reward_zero_std = (rewards.std(dim=1) == 0).float().mean().item()

        # --- 4. Compute advantages ---
        advantages = compute_advantages(rewards, scale_rewards=config.scale_rewards)  # (num_prompts, rollouts)

        # Skip step if all advantages are zero (no learning signal).
        if (advantages == 0).all():
            step_time = time.time() - step_start
            logger.info(
                "[Step %d/%d] All-zero advantages (reward=%.3f), skipping update.",
                step,
                config.num_steps,
                mean_reward,
            )
            if config.wandb_enabled:
                import wandb

                wandb.log(
                    {
                        "step": step,
                        "reward/mean": mean_reward,
                        "reward/fraction_correct": fraction_correct,
                        "reward/kept_mean": mean_reward,
                        "reward/kept_fraction_correct": fraction_correct,
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

        # Flatten: (num_prompts, rollouts) → (total_rollouts,)
        flat_advantages = advantages.reshape(-1)

        # Build flat lists for tokenization and reward filtering.
        flat_prompt_ids = [
            prompt_token_ids[i] for i in range(config.prompt_batch_size) for _ in range(config.rollouts_per_prompt)
        ]
        flat_response_ids = [
            rollout_response_ids[i][j] for i in range(config.prompt_batch_size) for j in range(config.rollouts_per_prompt)
        ]
        # --- 5. Filter truncated completions (no EOS → no learning signal) ---
        # Drop them before tokenization to avoid wasting compute on zero-gradient sequences.
        # A response that completed naturally ends with EOS (stop token is included
        # in token_ids). Truncated responses (hit max_tokens) won't have it.
        kept_reward_mean = mean_reward
        kept_fraction_correct = fraction_correct
        if config.mask_truncated_completions:
            keep = [r[-1] == CORRECT_EOS_TOKEN_ID if r else False for r in flat_response_ids]
            n_total = len(flat_response_ids)
            n_truncated = n_total - sum(keep)

            if n_truncated > 0 and n_truncated < n_total:
                flat_prompt_ids = [p for p, k in zip(flat_prompt_ids, keep) if k]
                flat_response_ids = [r for r, k in zip(flat_response_ids, keep) if k]
                flat_advantages = flat_advantages[torch.tensor(keep)]
                # Reward stats for non-truncated samples only.
                kept_rewards = rewards.reshape(-1)[torch.tensor(keep)]
                kept_reward_mean = kept_rewards.mean().item()
                kept_fraction_correct = (kept_rewards > 0).float().mean().item()
            elif n_truncated == n_total:
                step_time = time.time() - step_start
                logger.info(
                    "[Step %d/%d] All completions truncated, skipping update.",
                    step,
                    config.num_steps,
                )
                if config.wandb_enabled:
                    import wandb

                    wandb.log(
                        {
                            "step": step,
                            "reward/mean": mean_reward,
                            "reward/fraction_correct": fraction_correct,
                            "reward/kept_mean": 0.0,
                            "reward/kept_fraction_correct": 0.0,
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

        # --- 5b. Pad into batched tensors ---
        max_length = config.max_model_len
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        tokenized = pad_token_id_pairs(flat_prompt_ids, flat_response_ids, max_length, pad_token_id=pad_id)

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
            )
        train_time = time.time() - train_start

        step_time = time.time() - step_start

        # --- 7. Logging ---
        log_data = {
            "step": step,
            "reward/mean": mean_reward,
            "reward/fraction_correct": fraction_correct,
            "reward/kept_mean": kept_reward_mean,
            "reward/kept_fraction_correct": kept_fraction_correct,
            "reward/frac_zero_std": frac_reward_zero_std,
            **completion_log_data,
            "time/step_total": step_time,
            "time/generation": gen_time,
            "time/training": train_time,
            **{f"train/{k}": v for k, v in metrics.items()},
        }

        if step % config.log_every == 0:
            kl_str = f" kl={metrics['kl']:.4f}" if "kl" in metrics else ""
            kept_str = (
                f" kept={kept_fraction_correct * 100:.1f}%" if completion_log_data["completions/clipped_ratio"] > 0 else ""
            )
            logger.info(
                "[Step %d/%d] reward=%.3f correct=%.1f%%%s len=%.0f grad=%.4f surr=%.4f%s"
                " ratio=%.3f clip=%.3f trunc=%.1f%% gen=%.1fs train=%.1fs",
                step,
                config.num_steps,
                mean_reward,
                fraction_correct * 100,
                kept_str,
                completion_log_data["completions/mean_length"],
                metrics.get("grad_norm", 0),
                metrics.get("surrogate_loss", 0),
                kl_str,
                metrics.get("mean_ratio", 1),
                metrics.get("clip_ratio", 0),
                completion_log_data["completions/clipped_ratio"] * 100,
                gen_time,
                train_time,
            )

        if config.wandb_enabled:
            import wandb

            wandb.log(log_data)

        # --- 8. Checkpoint ---
        if step % config.save_every == 0 or step == config.num_steps:
            ckpt_dir = output_dir / f"step_{step:04d}"
            logger.info("Saving checkpoint to %s", ckpt_dir)
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # Update model path for vLLM to load from checkpoint next step.
            current_model_path = str(ckpt_dir)

            # Delete old checkpoints to save disk space.
            if config.keep_last_n_checkpoints > 0:
                existing = sorted(output_dir.glob("step_*"))
                to_delete = existing[: -config.keep_last_n_checkpoints]
                for old_ckpt in to_delete:
                    logger.info("Deleting old checkpoint: %s", old_ckpt)
                    shutil.rmtree(old_ckpt)

    # Final save.
    final_dir = output_dir / "final"
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete. Final model saved to %s", final_dir)

    if config.wandb_enabled:
        import wandb

        wandb.finish()


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
    p.add_argument("--mask-truncated", dest="mask_truncated_completions", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--output-dir")
    p.add_argument("--save-every", type=int)
    p.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb", dest="wandb_enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    return GRPOConfig(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main(parse_args())
