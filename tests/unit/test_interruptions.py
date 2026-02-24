"""Unit tests for interruption detection and stitching logic in grpo_train.py."""

import copy

# These are pure-Python helpers — no GPU, no vLLM, no model required.
from scripts.grpo_train import find_truncated_completions, stitch_interruptions

# Fake token IDs for tests.
EOS = 2  # <|im_end|>
THINK_CLOSE = 99  # </think>
INTERRUPTION_IDS = [50, 51, 52]  # Tokenized interruption phrase.


# ---------------------------------------------------------------------------
# find_truncated_completions
# ---------------------------------------------------------------------------


class TestFindTruncatedCompletions:
    def test_all_complete_with_eos(self):
        """Completions ending with EOS are not truncated."""
        rollouts = [
            [[10, 20, 30, EOS], [10, 20, EOS]],
            [[40, 50, EOS], [60, EOS]],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == []

    def test_all_complete_with_think_close(self):
        """Completions containing </think> are not truncated (model finished thinking)."""
        rollouts = [
            [[10, 20, THINK_CLOSE, 30, 40]],
            [[THINK_CLOSE, 10]],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == []

    def test_mixed_truncated_and_complete(self):
        """Correctly identifies only the truncated completions."""
        rollouts = [
            [
                [10, 20, 30, EOS],  # (0,0) complete — EOS
                [10, 20, 30, 40],  # (0,1) truncated — no EOS, no </think>
            ],
            [
                [10, THINK_CLOSE, 30],  # (1,0) complete — has </think>
                [10, 20],  # (1,1) truncated
            ],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == [(0, 1), (1, 1)]

    def test_all_truncated(self):
        """When all completions are truncated, all indices are returned."""
        rollouts = [
            [[10, 20], [30, 40]],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == [(0, 0), (0, 1)]

    def test_empty_response_is_truncated(self):
        """Empty response has neither EOS nor </think> — counts as truncated."""
        rollouts = [
            [[], [10, EOS]],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == [(0, 0)]

    def test_eos_not_at_end_is_truncated(self):
        """EOS in the middle but not at end: completion was truncated after EOS appeared in content."""
        rollouts = [
            [[10, EOS, 20, 30]],  # EOS not at end
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        # has_eos checks resp[-1] == EOS, so this IS truncated
        assert result == [(0, 0)]

    def test_both_eos_and_think_close(self):
        """Completion with both EOS and </think> is not truncated."""
        rollouts = [
            [[10, THINK_CLOSE, 20, EOS]],
        ]
        result = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert result == []


# ---------------------------------------------------------------------------
# stitch_interruptions
# ---------------------------------------------------------------------------


class TestStitchInterruptions:
    def test_basic_stitching(self):
        """Stitches thinking + interruption + answer correctly."""
        rollouts = [
            [[10, 20, 30], [40, 50, 60]],  # prompt 0: 2 rollouts
        ]
        needs_interruption = [(0, 0)]
        phase2_responses = [[[70, 80, EOS]]]  # answer for (0,0)

        stitch_interruptions(rollouts, needs_interruption, INTERRUPTION_IDS, phase2_responses)

        assert rollouts[0][0] == [10, 20, 30] + INTERRUPTION_IDS + [70, 80, EOS]
        # Untouched rollout stays the same.
        assert rollouts[0][1] == [40, 50, 60]

    def test_multiple_interruptions(self):
        """Stitches multiple truncated completions across different prompts."""
        rollouts = [
            [[10, 20], [30, 40, EOS]],  # (0,0) truncated, (0,1) complete
            [[50, 60], [70, 80]],  # (1,0) and (1,1) both truncated
        ]
        original = copy.deepcopy(rollouts)
        needs_interruption = [(0, 0), (1, 0), (1, 1)]
        phase2_responses = [
            [[90, EOS]],  # answer for (0,0)
            [[91, 92, EOS]],  # answer for (1,0)
            [[93, EOS]],  # answer for (1,1)
        ]

        stitch_interruptions(rollouts, needs_interruption, INTERRUPTION_IDS, phase2_responses)

        assert rollouts[0][0] == original[0][0] + INTERRUPTION_IDS + [90, EOS]
        assert rollouts[0][1] == original[0][1]  # unchanged
        assert rollouts[1][0] == original[1][0] + INTERRUPTION_IDS + [91, 92, EOS]
        assert rollouts[1][1] == original[1][1] + INTERRUPTION_IDS + [93, EOS]

    def test_empty_answer(self):
        """Phase 2 returns empty answer — still stitches interruption phrase."""
        rollouts = [[[10, 20]]]
        needs_interruption = [(0, 0)]
        phase2_responses = [[[EOS]]]

        stitch_interruptions(rollouts, needs_interruption, INTERRUPTION_IDS, phase2_responses)

        assert rollouts[0][0] == [10, 20] + INTERRUPTION_IDS + [EOS]

    def test_no_interruptions_is_noop(self):
        """Empty needs_interruption list doesn't modify anything."""
        rollouts = [[[10, 20, EOS]]]
        original = copy.deepcopy(rollouts)

        stitch_interruptions(rollouts, [], INTERRUPTION_IDS, [])

        assert rollouts == original


# ---------------------------------------------------------------------------
# Integration: find + stitch together
# ---------------------------------------------------------------------------


class TestInterruptionFlow:
    """End-to-end test of the detection → stitching pipeline (no vLLM)."""

    def test_detect_then_stitch(self):
        """find_truncated_completions + stitch_interruptions compose correctly."""
        rollouts = [
            [
                [10, 20, THINK_CLOSE, 30, EOS],  # complete (both)
                [10, 20, 30],  # truncated (no EOS, no </think>)
                [10, THINK_CLOSE, 40],  # complete (has </think>)
                [10, 20],  # truncated
            ],
        ]
        original = copy.deepcopy(rollouts)

        needs = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert needs == [(0, 1), (0, 3)]

        # Simulate phase 2 answers.
        phase2_responses = [
            [[100, EOS]],  # answer for (0,1)
            [[200, 201, EOS]],  # answer for (0,3)
        ]
        stitch_interruptions(rollouts, needs, INTERRUPTION_IDS, phase2_responses)

        # Complete rollouts unchanged.
        assert rollouts[0][0] == original[0][0]
        assert rollouts[0][2] == original[0][2]
        # Truncated rollouts stitched.
        assert rollouts[0][1] == original[0][1] + INTERRUPTION_IDS + [100, EOS]
        assert rollouts[0][3] == original[0][3] + INTERRUPTION_IDS + [200, 201, EOS]

    def test_all_complete_no_stitching(self):
        """When nothing is truncated, nothing changes."""
        rollouts = [
            [[10, EOS], [20, THINK_CLOSE, 30, EOS]],
        ]
        original = copy.deepcopy(rollouts)

        needs = find_truncated_completions(rollouts, EOS, THINK_CLOSE)
        assert needs == []
        # No stitch needed.
        assert rollouts == original
