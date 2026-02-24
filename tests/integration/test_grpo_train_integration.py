"""Integration tests for grpo_train.py — real vLLM + Ouro model.

Run with: uv run pytest tests/integration/ -m integration -v
Requires GPU with enough VRAM for the 1.4B model.
"""

import pytest
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from ouro_rl.data import CHAT_TEMPLATE, format_prompt
from ouro_rl.patches import CORRECT_BOS_TOKEN_ID, CORRECT_EOS_TOKEN_ID, PAD_TOKEN_ID, patch_ouro
from scripts.grpo_train import generate_rollouts, pad_token_id_pairs

MODEL_NAME = "ByteDance/Ouro-1.4B-Thinking"
MAX_MODEL_LEN = 512
MAX_NEW_TOKENS = 32

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    patch_ouro()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.chat_template = CHAT_TEMPLATE
    tok.bos_token_id = CORRECT_BOS_TOKEN_ID
    tok.eos_token_id = CORRECT_EOS_TOKEN_ID
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
    sampling_params = SamplingParams(
        n=2,
        temperature=0.7,
        max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[CORRECT_EOS_TOKEN_ID],
        skip_special_tokens=False,
    )
    response_ids = generate_rollouts(
        MODEL_NAME,
        prompt_token_ids,
        sampling_params,
        max_model_len=MAX_MODEL_LEN,
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
            assert CORRECT_BOS_TOKEN_ID in ids, f"Prompt {i}: missing <|im_start|>"
            assert CORRECT_EOS_TOKEN_ID in ids, f"Prompt {i}: missing <|im_end|>"
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
