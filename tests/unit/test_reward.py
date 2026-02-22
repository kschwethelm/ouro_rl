"""Unit tests for ouro_rl/reward.py — score_answer with math-verify."""

from ouro_rl.reward import score_answer


class TestScoreAnswerBasic:
    """Core exact-match reward logic."""

    def test_correct_boxed_answer(self):
        """Matching \\boxed answers → 1.0."""
        assert score_answer("The answer is \\boxed{42}", "\\boxed{42}") == 1.0

    def test_incorrect_boxed_answer(self):
        """Non-matching \\boxed answers → 0.0."""
        assert score_answer("\\boxed{99}", "\\boxed{42}") == 0.0

    def test_equivalent_fraction(self):
        """math-verify recognizes equivalent expressions: 1/2 == 0.5."""
        assert score_answer("\\boxed{\\frac{1}{2}}", "\\boxed{0.5}") == 1.0


class TestScoreAnswerThinkingStrip:
    """Verify <think>...</think> tags are stripped before scoring."""

    def test_strips_think_tags(self):
        """Answer after </think> is used, thinking content is ignored."""
        model_output = "<think>Let me reason... the answer is 99</think>\\boxed{42}"
        assert score_answer(model_output, "\\boxed{42}") == 1.0

    def test_right_answer_only_in_think(self):
        """Correct answer only inside <think> → 0.0 (must be in final answer)."""
        model_output = "<think>\\boxed{42}</think>I'm not sure."
        assert score_answer(model_output, "\\boxed{42}") == 0.0


class TestScoreAnswerEdgeCases:
    """Edge cases and failure modes."""

    def test_empty_model_output(self):
        assert score_answer("", "\\boxed{42}") == 0.0

    def test_no_boxed_in_prediction_gibberish(self):
        """Model produced gibberish with no parseable answer → 0.0."""
        assert score_answer("I don't know the answer sorry", "\\boxed{42}") == 0.0

    def test_plain_number_match(self):
        """Plain numbers (no \\boxed) can still be parsed by ExprExtractionConfig."""
        assert score_answer("42", "\\boxed{42}") == 1.0
