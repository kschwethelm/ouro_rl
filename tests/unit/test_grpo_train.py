"""Unit tests for grpo_train.py — pad_token_id_pairs."""

import torch

from ouro_rl.modeling import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID
from scripts.grpo_train import pad_token_id_pairs

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
