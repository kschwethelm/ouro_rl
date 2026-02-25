"""Integration tests for grpo_train.py — real vLLM + Ouro model.

Run with:
    uv run pytest tests/integration/test_grpo_train_integration.py -m integration -v

Requires GPU with enough VRAM for the 1.4B model.
Each test spawns a vLLM subprocess that exits after generation, so GPU memory
is fully reclaimed between tests.
"""

import pytest
import torch
from transformers import AutoTokenizer

from ouro_rl.data import CHAT_TEMPLATE, INTERRUPTION_PHRASE, format_prompt
from ouro_rl.modeling import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
from scripts.grpo_train import (
    pad_token_id_pairs,
    run_vllm_generation,
)

MODEL_NAME = "ByteDance/Ouro-1.4B-Thinking"
MAX_MODEL_LEN = 512
MAX_NEW_TOKENS = 32

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.chat_template = CHAT_TEMPLATE
    tok.bos_token_id = BOS_TOKEN_ID
    tok.eos_token_id = EOS_TOKEN_ID
    tok.pad_token_id = PAD_TOKEN_ID
    return tok


@pytest.fixture(scope="module")
def rollout_results(tokenizer):
    """Generate rollouts once and share across tests in this module.

    Uses format_prompt (with chat template + enable_thinking) to match the
    real training pipeline, so prompt_ids will contain all special tokens.
    """
    raw_prompts = ["What is 2 + 2?", "What is the capital of France?"]
    formatted_prompts = [format_prompt(p, tokenizer, enable_thinking=True) for p in raw_prompts]
    prompt_token_ids = [tokenizer.encode(p) for p in formatted_prompts]

    response_ids, _ = run_vllm_generation(
        model_path=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        seed=42,
        prompt_token_ids=prompt_token_ids,
        n=2,
        temperature=0.7,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[EOS_TOKEN_ID],
        gpu_id=0,
    )
    return prompt_token_ids, response_ids, formatted_prompts


@requires_cuda
@pytest.mark.integration
class TestGenerateRolloutsIntegration:
    def test_output_structure(self, tokenizer, rollout_results):
        """2 prompts × 2 rollouts: correct nesting, non-empty, decodable."""
        prompt_ids, response_ids, _ = rollout_results

        assert len(prompt_ids) == 2
        assert len(response_ids) == 2

        for i in range(2):
            assert len(response_ids[i]) == 2
            assert isinstance(prompt_ids[i], list)
            assert all(isinstance(t, int) for t in prompt_ids[i])
            assert len(tokenizer.decode(prompt_ids[i])) > 0

            for j in range(2):
                assert len(response_ids[i][j]) >= 1
                assert len(tokenizer.decode(response_ids[i][j])) > 0

    def test_special_tokens_in_prompt_ids(self, tokenizer, rollout_results):
        """Prompt token IDs contain ChatML structural tokens from the template.

        The formatted prompt looks like:
            <|im_start|>user\\n{problem}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n
        """
        prompt_ids, _, _ = rollout_results
        think_token_id = tokenizer.convert_tokens_to_ids("<think>")

        for i, ids in enumerate(prompt_ids):
            assert BOS_TOKEN_ID in ids, f"Prompt {i}: missing <|im_start|>"
            assert EOS_TOKEN_ID in ids, f"Prompt {i}: missing <|im_end|>"
            assert think_token_id in ids, f"Prompt {i}: missing <think>"
            # <think> should be near the end — template appends "<think>\n" after "assistant\n".
            think_pos = ids.index(think_token_id)
            assert think_pos >= len(ids) - 3, f"Prompt {i}: <think> at position {think_pos}/{len(ids)}, expected near end"

    def test_prompt_response_boundary(self, tokenizer, rollout_results):
        """No duplicate special tokens at the prompt/response join point."""
        prompt_ids, response_ids, _ = rollout_results
        think_token_id = tokenizer.convert_tokens_to_ids("<think>")

        for prompt_idx in range(len(prompt_ids)):
            for rollout_idx in range(len(response_ids[prompt_idx])):
                r_ids = response_ids[prompt_idx][rollout_idx]
                # Response should not duplicate the prompt's trailing <think>.
                if len(r_ids) > 0:
                    assert r_ids[0] != think_token_id, (
                        f"[{prompt_idx}][{rollout_idx}]: response starts with <think>, "
                        "duplicating the prompt's trailing <think>"
                    )


@requires_cuda
@pytest.mark.integration
class TestPadAndRecoverIntegration:
    def test_pad_preserves_content(self, rollout_results):
        """After padding, non-pad positions contain original prompt+response IDs."""
        prompt_ids, response_ids, _ = rollout_results

        # Flatten: repeat prompt_ids for each rollout
        flat_prompt_ids = [prompt_ids[i] for i in range(2) for _ in range(2)]
        flat_response_ids = [response_ids[i][j] for i in range(2) for j in range(2)]

        result = pad_token_id_pairs(flat_prompt_ids, flat_response_ids, max_length=MAX_MODEL_LEN, pad_token_id=PAD_TOKEN_ID)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        for seq_idx in range(4):
            actual_ids = input_ids[seq_idx][attention_mask[seq_idx].bool()].tolist()
            expected = flat_prompt_ids[seq_idx] + flat_response_ids[seq_idx]
            # May be truncated if over max_length, so compare up to actual length.
            assert actual_ids == expected[: len(actual_ids)]


# ---------------------------------------------------------------------------
# Two-phase interruption flow
# ---------------------------------------------------------------------------

THINKING_BUDGET = 16  # Tiny budget to force truncation.
ANSWER_BUDGET = 32


@requires_cuda
@pytest.mark.integration
class TestInterruptionFlowIntegration:
    """End-to-end test of the two-phase interruption pipeline with real vLLM."""

    def test_two_phase_generation(self, tokenizer):
        """Phase 1 (tiny budget) → detect truncated → Phase 2 → stitch.

        With a 16-token thinking budget, the model won't produce </think> or EOS,
        so all completions should be identified as truncated and interrupted.
        """
        prompts = ["What is 2 + 2?", "Solve x + 1 = 3."]
        formatted = [format_prompt(p, tokenizer, enable_thinking=True) for p in prompts]
        prompt_token_ids = [tokenizer.encode(p) for p in formatted]

        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        interruption_token_ids = tokenizer.encode(INTERRUPTION_PHRASE, add_special_tokens=False)

        n_prompts = len(prompts)
        n_rollouts = 2

        rollout_response_ids, n_interrupted = run_vllm_generation(
            model_path=MODEL_NAME,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            seed=42,
            prompt_token_ids=prompt_token_ids,
            n=n_rollouts,
            temperature=0.7,
            top_p=1.0,
            max_tokens=THINKING_BUDGET,
            stop_token_ids=[EOS_TOKEN_ID],
            interruptions={
                "think_close_id": think_close_id,
                "interruption_token_ids": interruption_token_ids,
                "phase2_temperature": 0.7,
                "phase2_top_p": 1.0,
                "phase2_max_tokens": ANSWER_BUDGET,
                "phase2_stop_token_ids": [EOS_TOKEN_ID],
            },
            gpu_id=0,
        )

        # Verify structure.
        assert len(rollout_response_ids) == n_prompts
        for prompt_rollouts in rollout_response_ids:
            assert len(prompt_rollouts) == n_rollouts

        # With 16-token budget, expect most/all to be truncated.
        assert n_interrupted > 0, "Expected at least some truncated completions with tiny budget"

        # Verify stitching: interrupted completions should contain the
        # interruption phrase and be longer than the thinking budget.
        for prompt_rollouts in rollout_response_ids:
            for resp_ids in prompt_rollouts:
                if len(resp_ids) > THINKING_BUDGET:
                    decoded = tokenizer.decode(resp_ids)
                    assert "time is up" in decoded.lower(), (
                        f"Interrupted response should contain interruption phrase: {decoded[:200]}"
                    )

    def test_interrupted_completions_are_decodable(self, tokenizer):
        """Stitched responses decode to valid text containing thinking + interruption + answer."""
        prompt = "What is 7 * 8?"
        formatted = format_prompt(prompt, tokenizer, enable_thinking=True)
        prompt_token_ids = [tokenizer.encode(formatted)]

        think_close_id = tokenizer.convert_tokens_to_ids("</think>")
        interruption_token_ids = tokenizer.encode(INTERRUPTION_PHRASE, add_special_tokens=False)

        rollout_response_ids, n_interrupted = run_vllm_generation(
            model_path=MODEL_NAME,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            seed=42,
            prompt_token_ids=prompt_token_ids,
            n=1,
            temperature=0.7,
            top_p=1.0,
            max_tokens=THINKING_BUDGET,
            stop_token_ids=[EOS_TOKEN_ID],
            interruptions={
                "think_close_id": think_close_id,
                "interruption_token_ids": interruption_token_ids,
                "phase2_temperature": 0.7,
                "phase2_top_p": 1.0,
                "phase2_max_tokens": ANSWER_BUDGET,
                "phase2_stop_token_ids": [EOS_TOKEN_ID],
            },
            gpu_id=0,
        )

        # Every response should decode cleanly.
        for prompt_rollouts in rollout_response_ids:
            for resp_ids in prompt_rollouts:
                decoded = tokenizer.decode(resp_ids)
                assert len(decoded) > 0, "Decoded response should not be empty"
                # If it was interrupted, it should contain </think> from the interruption phrase.
                if n_interrupted > 0:
                    assert "</think>" in decoded, f"Interrupted response should contain </think>: {decoded[:200]}"
