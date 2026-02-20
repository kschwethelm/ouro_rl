"""GRPO training loop for Ouro-1.4B-Thinking.

Orchestrates: vLLM rollout generation → reward scoring → GRPO policy update.

Usage:
    uv run python grpo_train.py
    uv run python grpo_train.py --smoke-test          # 3 steps, small batch
    uv run python grpo_train.py --num-steps 140       # full training run

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
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from ouro_rl.data import CHAT_TEMPLATE, format_prompt, load_math_train
from ouro_rl.grpo import compute_advantages, compute_log_probs_batch, compute_log_probs_with_grad, grpo_loss
from ouro_rl.patches import patch_ouro, patch_ouro_post_load
from ouro_rl.reward import score_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Hyperparameters matching RLTT paper Table 6 (scaled for 1.4B on 2×A100)."""

    # Model
    model_name: str = "ByteDance/Ouro-1.4B-Thinking"
    dtype: str = "bfloat16"
    max_model_len: int = 4096  # vLLM context window

    # Dataset
    dataset_name: str = "qwedsacf/competition_math"
    system_prompt: str = (
        "You are a helpful assistant. Solve the following math problem step by step. Put your final answer within \\boxed{}."
    )

    # Training
    num_steps: int = 140
    prompt_batch_size: int = 16
    rollouts_per_prompt: int = 8
    grad_accumulation_steps: int = 4
    lr: float = 1e-6
    max_grad_norm: float = 0.1
    warmup_steps: int = 14  # 10% of 140
    weight_decay: float = 0.01

    # GRPO
    clip_eps: float = 0.2
    kl_coeff: float = 1e-3

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    enable_thinking: bool = True

    # Memory
    log_prob_micro_batch: int = 4  # micro-batch for log-prob forward passes
    train_micro_batch: int = 2  # micro-batch for training forward/backward

    # Logging & checkpointing
    output_dir: str = "outputs/grpo"
    log_every: int = 1
    save_every: int = 10
    wandb_project: str = "ouro-rl"
    wandb_enabled: bool = True

    # Smoke test overrides
    smoke_test: bool = False

    def __post_init__(self) -> None:
        if self.smoke_test:
            self.num_steps = 3
            self.prompt_batch_size = 4
            self.rollouts_per_prompt = 4
            self.grad_accumulation_steps = 1
            self.save_every = 1
            self.wandb_enabled = False
            self.max_new_tokens = 256
            self.max_model_len = 2048

    # Derived
    @property
    def total_rollouts_per_step(self) -> int:
        return self.prompt_batch_size * self.rollouts_per_prompt

    @property
    def effective_batch_size(self) -> int:
        return self.total_rollouts_per_step * self.grad_accumulation_steps


# ---------------------------------------------------------------------------
# vLLM rollout generation
# ---------------------------------------------------------------------------


def generate_rollouts(
    model_path: str,
    prompts: list[str],
    rollouts_per_prompt: int,
    sampling_params: SamplingParams,
    *,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
) -> list[list[str]]:
    """Generate multiple rollouts per prompt using vLLM.

    Creates a vLLM LLM instance, generates, then destroys it to free GPU memory.

    Returns:
        List of lists: outer=prompts, inner=rollouts per prompt.
    """
    # Expand prompts: each prompt repeated rollouts_per_prompt times.
    expanded_prompts = [p for p in prompts for _ in range(rollouts_per_prompt)]

    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,  # Leave room for training models later.
        enforce_eager=True,  # Simpler, avoids CUDA graph issues with model swapping.
    )
    outputs = llm.generate(expanded_prompts, sampling_params)

    # Clean up vLLM completely.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Reshape: (num_prompts * rollouts_per_prompt,) → (num_prompts, rollouts_per_prompt)
    results: list[list[str]] = []
    for i in range(0, len(outputs), rollouts_per_prompt):
        group = [outputs[i + j].outputs[0].text for j in range(rollouts_per_prompt)]
        results.append(group)
    return results


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_prompt_response_pairs(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt+response pairs, returning input_ids, attention_mask, response_start_indices, response_mask.

    Left-pads sequences to max_length for batch processing.
    """
    all_input_ids = []
    all_response_starts = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        # Truncate response if prompt+response exceeds max_length.
        max_resp_len = max_length - len(prompt_ids)
        if max_resp_len <= 0:
            # Prompt itself is too long — truncate prompt from the left.
            prompt_ids = prompt_ids[-(max_length - 1) :]
            response_ids = response_ids[:1]
            max_resp_len = 1
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

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for ids, resp_start in zip(all_input_ids, all_response_starts):
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
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
    ref_model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenized: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    config: GRPOConfig,
) -> dict[str, float]:
    """Single GRPO gradient update with micro-batching.

    Returns aggregated metrics dict.
    """
    device = next(policy_model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    response_start_indices = tokenized["response_start_indices"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    advantages = advantages.to(device)

    batch_size = input_ids.shape[0]

    # Compute reference log-probs (no grad, micro-batched).
    ref_log_probs = compute_log_probs_batch(
        ref_model,
        input_ids,
        attention_mask,
        response_start_indices,
        micro_batch_size=config.log_prob_micro_batch,
    )

    # Compute old log-probs (from current policy, no grad — these are the "old" policy for this step).
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
            ref_log_probs=ref_log_probs[mb_slice],
            old_log_probs=old_log_probs[mb_slice],
            advantages=advantages[mb_slice],
            response_mask=response_mask[mb_slice],
            clip_eps=config.clip_eps,
            kl_coeff=config.kl_coeff,
        )

        # Scale loss by number of micro-batches for correct gradient averaging.
        scaled_loss = loss / num_micro_batches
        scaled_loss.backward()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v / num_micro_batches

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
    optimizer.step()
    scheduler.step()

    total_metrics["lr"] = scheduler.get_last_lr()[0]
    return total_metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(config: GRPOConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

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
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
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

        # --- 2. Generate rollouts with vLLM ---
        logger.info("[Step %d/%d] Generating %d rollouts...", step, config.num_steps, config.total_rollouts_per_step)

        # Move training models to CPU to free GPU memory for vLLM.
        policy_model.cpu()
        ref_model.cpu()
        torch.cuda.empty_cache()

        gen_start = time.time()
        rollout_texts = generate_rollouts(
            current_model_path,
            prompts,
            config.rollouts_per_prompt,
            sampling_params,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
        )
        gen_time = time.time() - gen_start

        # Move training models back to GPU.
        device = torch.device("cuda")
        policy_model.to(device)
        ref_model.to(device)

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

        # --- 4. Compute advantages ---
        advantages = compute_advantages(rewards)  # (num_prompts, rollouts)

        # Skip step if all advantages are zero (no learning signal).
        if (advantages == 0).all():
            logger.info(
                "[Step %d/%d] All-zero advantages (reward=%.3f), skipping update.",
                step,
                config.num_steps,
                mean_reward,
            )
            if config.wandb_enabled:
                import wandb

                wandb.log(
                    {"step": step, "reward/mean": mean_reward, "reward/fraction_correct": fraction_correct, "skipped": True}
                )
            continue

        # Flatten: (num_prompts, rollouts) → (total_rollouts,)
        flat_advantages = advantages.reshape(-1)

        # Build flat lists of prompts and responses for tokenization.
        flat_prompts = [prompts[i] for i in range(config.prompt_batch_size) for _ in range(config.rollouts_per_prompt)]
        flat_responses = [
            rollout_texts[i][j] for i in range(config.prompt_batch_size) for j in range(config.rollouts_per_prompt)
        ]

        # --- 5. Tokenize ---
        max_length = config.max_model_len
        tokenized = tokenize_prompt_response_pairs(tokenizer, flat_prompts, flat_responses, max_length)

        # --- 6. GRPO update ---
        train_start = time.time()
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
            "time/step_total": step_time,
            "time/generation": gen_time,
            "time/training": train_time,
            **{f"train/{k}": v for k, v in metrics.items()},
        }

        if step % config.log_every == 0:
            logger.info(
                "[Step %d/%d] reward=%.3f correct=%.1f%% surr=%.4f kl=%.4f ratio=%.3f gen=%.1fs train=%.1fs",
                step,
                config.num_steps,
                mean_reward,
                fraction_correct * 100,
                metrics.get("surrogate_loss", 0),
                metrics.get("kl", 0),
                metrics.get("mean_ratio", 1),
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
    p.add_argument("--model", dest="model_name", default="ByteDance/Ouro-1.4B-Thinking")
    p.add_argument("--dataset", dest="dataset_name", default="qwedsacf/competition_math")
    p.add_argument("--num-steps", type=int, default=140)
    p.add_argument("--prompt-batch-size", type=int, default=16)
    p.add_argument("--rollouts-per-prompt", type=int, default=8)
    p.add_argument("--grad-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coeff", type=float, default=1e-3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", default="outputs/grpo")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    args = p.parse_args()
    return GRPOConfig(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main(parse_args())
