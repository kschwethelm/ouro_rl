"""Unit tests for grpo_train.py — pad_token_id_pairs and generate_rollouts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from ouro_rl.modeling import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
from scripts.grpo_train import align_generation_logprobs, generate_rollouts, pad_token_id_pairs

# Realistic ChatML token IDs for Ouro-Thinking.
# <|im_start|>=1, <|im_end|>=2, <|endoftext|>=0, <think>=151648, </think>=151649
IM_START = BOS_TOKEN_ID  # 1
IM_END = EOS_TOKEN_ID  # 2
PAD = PAD_TOKEN_ID  # 0
THINK_OPEN = 151648
THINK_CLOSE = 151649
NEWLINE = 198  # "\n" token in Qwen/Ouro tokenizer

# Realistic ChatML token sequences for Ouro-Thinking.
# prompt: "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
CHATML_PROMPT_IDS = [
    IM_START,
    882,
    NEWLINE,
    3838,
    374,
    220,
    17,
    10,
    17,
    30,
    IM_END,
    NEWLINE,
    IM_START,
    78191,
    NEWLINE,
    THINK_OPEN,
    NEWLINE,
]
# response: "4\n</think>\nThe answer is 4.<|im_end|>"
CHATML_RESPONSE_IDS = [19, NEWLINE, THINK_CLOSE, NEWLINE, 791, 4226, 374, 220, 19, 13, IM_END]


# ---------------------------------------------------------------------------
# TestPadTokenIdPairs
# ---------------------------------------------------------------------------


class TestPadTokenIdPairs:
    def test_left_padding(self):
        """Shorter sequence is left-padded; attention mask marks real tokens."""
        prompt_ids = [[1, 2], [10, 20, 30]]
        response_ids = [[3], [40, 50]]

        result = pad_token_id_pairs(prompt_ids, response_ids, max_length=20)

        # Longer: [10,20,30,40,50] = 5, shorter: [1,2,3] = 3 → padded to 5
        assert result["input_ids"].shape == (2, 5)
        assert torch.equal(result["input_ids"][0], torch.tensor([0, 0, 1, 2, 3]))
        assert torch.equal(result["input_ids"][1], torch.tensor([10, 20, 30, 40, 50]))
        assert torch.equal(result["attention_mask"][0], torch.tensor([0, 0, 1, 1, 1]))
        assert torch.equal(result["attention_mask"][1], torch.tensor([1, 1, 1, 1, 1]))
        # response_start accounts for left-padding offset.
        assert result["response_start_indices"][0].item() == 4  # 2 pad + 2 prompt
        assert result["response_start_indices"][1].item() == 3  # 0 pad + 3 prompt

    def test_response_truncation(self):
        """Response truncated when prompt+response exceeds max_length."""
        prompt_ids = [[1, 2, 3]]
        response_ids = [[4, 5, 6, 7, 8]]

        result = pad_token_id_pairs(prompt_ids, response_ids, max_length=5)

        # max_length=5, prompt=3 → only 2 response tokens fit
        assert result["input_ids"].shape == (1, 5)
        assert torch.equal(result["input_ids"][0], torch.tensor([1, 2, 3, 4, 5]))

    def test_prompt_truncation(self):
        """Prompt exceeds max_length → truncated from left, response clipped to 1 token."""
        prompt_ids = [[1, 2, 3, 4, 5, 6]]
        response_ids = [[7, 8, 9]]

        result = pad_token_id_pairs(prompt_ids, response_ids, max_length=4)

        # max_length=4, prompt too long → keep last 3 prompt tokens + 1 response token
        assert result["input_ids"].shape == (1, 4)
        assert torch.equal(result["input_ids"][0], torch.tensor([4, 5, 6, 7]))
        assert result["response_start_indices"][0].item() == 3

    def test_response_mask(self):
        """Response mask: 1 at response positions, 0 at prompt and padding."""
        prompt_ids = [[1, 2], [10, 20, 30]]
        response_ids = [[3, 4, 5], [40]]

        result = pad_token_id_pairs(prompt_ids, response_ids, max_length=20)

        # Seq 0: prompt=[1,2], response=[3,4,5] → combined len=5
        # Seq 1: prompt=[10,20,30], response=[40] → combined len=4, padded to 5
        response_mask = result["response_mask"]
        assert response_mask[0].tolist() == [0, 0, 1, 1, 1]  # no pad, 2 prompt, 3 response
        assert response_mask[1].tolist() == [0, 0, 0, 0, 1]  # 1 pad, 3 prompt, 1 response

    def test_truncation_with_padding(self):
        """One sequence truncated by max_length, another padded — both correct."""
        prompt_ids = [[1, 2], [10, 20, 30, 40, 50]]
        response_ids = [[3, 4], [60, 70, 80]]

        result = pad_token_id_pairs(prompt_ids, response_ids, max_length=6)

        # Seq 0: [1,2,3,4] = 4 tokens, fits, padded to 6
        # Seq 1: prompt=5 + response=3 = 8 > max_length=6, response truncated to 1 → [10,20,30,40,50,60]
        assert result["input_ids"].shape == (2, 6)
        assert torch.equal(result["input_ids"][0], torch.tensor([0, 0, 1, 2, 3, 4]))
        assert torch.equal(result["input_ids"][1], torch.tensor([10, 20, 30, 40, 50, 60]))
        assert result["response_mask"][0].tolist() == [0, 0, 0, 0, 1, 1]  # 2 pad, 2 prompt, 2 response
        assert result["response_mask"][1].tolist() == [0, 0, 0, 0, 0, 1]  # 5 prompt, 1 response

    def test_chatml_tokens_preserved(self):
        """Padding doesn't corrupt ChatML special tokens."""
        short_prompt = [IM_START, 882, IM_END, NEWLINE, IM_START, 78191, NEWLINE, THINK_OPEN, NEWLINE]
        short_response = [19, THINK_CLOSE, NEWLINE, IM_END]

        result = pad_token_id_pairs(
            [CHATML_PROMPT_IDS, short_prompt],
            [CHATML_RESPONSE_IDS, short_response],
            max_length=100,
            pad_token_id=PAD,
        )

        for seq_idx in range(2):
            real_ids = result["input_ids"][seq_idx][result["attention_mask"][seq_idx].bool()].tolist()

            assert IM_START in real_ids, f"Seq {seq_idx}: missing <|im_start|>"
            assert IM_END in real_ids, f"Seq {seq_idx}: missing <|im_end|>"
            assert THINK_OPEN in real_ids, f"Seq {seq_idx}: missing <think>"
            assert THINK_CLOSE in real_ids, f"Seq {seq_idx}: missing </think>"
            assert real_ids[-1] == IM_END, f"Seq {seq_idx}: last token should be <|im_end|>"


# ---------------------------------------------------------------------------
# TestGenerateRollouts
# ---------------------------------------------------------------------------


def _mock_model_generate(prompt_token_ids: list[list[int]], response_map: dict[int, list[list[int]]]):
    """Build a mock model whose .generate() returns known responses.

    Args:
        prompt_token_ids: List of prompt token ID sequences.
        response_map: {prompt_idx: [[rollout0_ids], [rollout1_ids], ...]}
    """
    model = MagicMock()
    model.training = False
    model.parameters.return_value = iter([torch.zeros(1)])  # device = cpu

    n_prompts = len(prompt_token_ids)
    max_prompt_len = max(len(ids) for ids in prompt_token_ids)

    def fake_generate(input_ids, attention_mask, **kwargs):
        num_rollouts = kwargs.get("num_return_sequences", 1)
        # Determine max response length across all rollouts.
        max_resp_len = 0
        for pi in range(n_prompts):
            for ri in range(num_rollouts):
                resp = response_map[pi][ri]
                max_resp_len = max(max_resp_len, len(resp))

        total_len = max_prompt_len + max_resp_len
        total_seqs = n_prompts * num_rollouts
        sequences = torch.full((total_seqs, total_len), PAD_TOKEN_ID, dtype=torch.long)

        for pi in range(n_prompts):
            for ri in range(num_rollouts):
                seq_idx = pi * num_rollouts + ri
                # Copy prompt (left-padded to max_prompt_len).
                ids = prompt_token_ids[pi]
                pad_len = max_prompt_len - len(ids)
                sequences[seq_idx, pad_len:max_prompt_len] = torch.tensor(ids, dtype=torch.long)
                # Copy response.
                resp = response_map[pi][ri]
                sequences[seq_idx, max_prompt_len : max_prompt_len + len(resp)] = torch.tensor(resp, dtype=torch.long)

        return SimpleNamespace(sequences=sequences, scores=None)

    model.generate.side_effect = fake_generate
    return model


class TestGenerateRollouts:
    def test_output_nesting(self):
        """2 prompts × 3 rollouts → correct nesting structure."""
        prompts = [[1, 2], [3, 4, 5]]
        responses = {
            0: [[10], [11], [12]],
            1: [[20, 21], [22, 23], [24, 25]],
        }
        model = _mock_model_generate(prompts, responses)

        result_ids, result_lp = generate_rollouts(model, prompts, num_rollouts=3)

        assert len(result_ids) == 2
        assert len(result_ids[0]) == 3
        assert len(result_ids[1]) == 3
        assert result_ids[0][0] == [10]
        assert result_ids[0][1] == [11]
        assert result_ids[0][2] == [12]
        assert result_ids[1][0] == [20, 21]
        assert result_ids[1][2] == [24, 25]
        assert result_lp is None  # return_logprobs=False by default

    def test_single_rollout(self):
        """1 prompt × 1 rollout."""
        prompts = [[1, 2, 3]]
        responses = {0: [[10, 20, 30]]}
        model = _mock_model_generate(prompts, responses)

        result_ids, _ = generate_rollouts(model, prompts, num_rollouts=1)

        assert len(result_ids) == 1
        assert len(result_ids[0]) == 1
        assert result_ids[0][0] == [10, 20, 30]

    def test_strips_trailing_pad(self):
        """Trailing PAD tokens from generation are stripped."""
        # Two prompts with different response lengths — shorter one gets padded in output.
        prompts = [[1, 2], [3, 4]]
        responses = {0: [[10]], 1: [[20, 30, 40]]}
        model = _mock_model_generate(prompts, responses)

        result_ids, _ = generate_rollouts(model, prompts, num_rollouts=1)

        assert result_ids[0][0] == [10]  # Trailing pads stripped
        assert result_ids[1][0] == [20, 30, 40]

    def test_restores_training_mode(self):
        """Model's training mode is restored after generate_rollouts."""
        prompts = [[1, 2]]
        responses = {0: [[10]]}
        model = _mock_model_generate(prompts, responses)
        model.training = True

        generate_rollouts(model, prompts, num_rollouts=1)

        model.train.assert_called_once()

    def test_eval_mode_stays_eval(self):
        """Model already in eval mode stays eval."""
        prompts = [[1, 2]]
        responses = {0: [[10]]}
        model = _mock_model_generate(prompts, responses)
        model.training = False

        generate_rollouts(model, prompts, num_rollouts=1)

        model.train.assert_not_called()


# ---------------------------------------------------------------------------
# TestAlignGenerationLogprobs
# ---------------------------------------------------------------------------


class TestAlignGenerationLogprobs:
    def test_basic_alignment(self):
        """Logprobs placed at correct response positions, zeros elsewhere."""
        flat_gen_logprobs = [[-0.5, -1.0, -0.3], [-0.2, -0.8]]
        response_start_indices = torch.tensor([3, 2])
        seq_len = 6

        result = align_generation_logprobs(flat_gen_logprobs, response_start_indices, seq_len, torch.device("cpu"))

        assert result.shape == (2, 6)
        # Seq 0: logprobs at positions 3, 4, 5
        expected_0 = torch.tensor([0.0, 0.0, 0.0, -0.5, -1.0, -0.3])
        assert torch.allclose(result[0], expected_0)
        # Seq 1: logprobs at positions 2, 3
        expected_1 = torch.tensor([0.0, 0.0, -0.2, -0.8, 0.0, 0.0])
        assert torch.allclose(result[1], expected_1)

    def test_truncation_when_exceeds_seq_len(self):
        """Logprobs truncated if they would exceed seq_len."""
        flat_gen_logprobs = [[-0.1, -0.2, -0.3, -0.4]]
        response_start_indices = torch.tensor([3])
        seq_len = 5  # Only positions 3, 4 available (not 5, 6)

        result = align_generation_logprobs(flat_gen_logprobs, response_start_indices, seq_len, torch.device("cpu"))

        assert result.shape == (1, 5)
        expected = torch.tensor([0.0, 0.0, 0.0, -0.1, -0.2])
        assert torch.allclose(result[0], expected)

    def test_empty_logprobs(self):
        """Empty logprob list produces all zeros."""
        flat_gen_logprobs = [[]]
        response_start_indices = torch.tensor([2])
        seq_len = 4

        result = align_generation_logprobs(flat_gen_logprobs, response_start_indices, seq_len, torch.device("cpu"))

        assert result.shape == (1, 4)
        assert result[0].tolist() == [0.0, 0.0, 0.0, 0.0]
