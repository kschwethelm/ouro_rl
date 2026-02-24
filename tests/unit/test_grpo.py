"""Unit tests for ouro_rl/grpo.py — advantages, log-probs, GRPO loss, and CISPO loss."""

import torch

from ouro_rl.grpo import cispo_loss, compute_advantages, compute_log_probs_with_grad, grpo_loss

# ---------------------------------------------------------------------------
# compute_advantages
# ---------------------------------------------------------------------------


class TestComputeAdvantages:
    def test_group_normalization_basic(self):
        """Group normalization: mean-centered and std-scaled per row."""
        rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # 1 prompt, 4 rollouts
        adv = compute_advantages(rewards, scale_rewards="group")

        assert adv.shape == (1, 4)
        assert abs(adv.sum().item()) < 1e-5

    def test_group_all_same_reward(self):
        """All rollouts got same reward → zero advantage (std=0 guard)."""
        rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        adv = compute_advantages(rewards, scale_rewards="group")
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_none_normalization(self):
        """scale_rewards='none': subtract mean only, no std division."""
        rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        adv = compute_advantages(rewards, scale_rewards="none")

        mean = rewards.mean(dim=1, keepdim=True)
        expected = rewards - mean
        assert torch.allclose(adv, expected)

    def test_batch_normalization(self):
        """scale_rewards='batch': divide by batch-wide std instead of group std."""
        rewards = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        adv = compute_advantages(rewards, scale_rewards="batch")

        mean = rewards.mean(dim=1, keepdim=True)
        batch_std = rewards.std()
        expected = (rewards - mean) / (batch_std + 1e-8)
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_multiple_groups_independent(self):
        """Each prompt group is normalized independently."""
        rewards = torch.tensor(
            [
                [1.0, 0.0],  # group 0: mean=0.5
                [0.0, 0.0],  # group 1: all same → zero advantage
            ]
        )
        adv = compute_advantages(rewards, scale_rewards="group")

        assert torch.allclose(adv[1], torch.zeros(2))
        assert adv[0, 0].item() > 0  # reward=1 > mean=0.5
        assert adv[0, 1].item() < 0  # reward=0 < mean=0.5


# ---------------------------------------------------------------------------
# compute_log_probs_with_grad
# ---------------------------------------------------------------------------


class TestComputeLogProbsWithGrad:
    """Tests for the differentiable log-prob computation."""

    def _make_dummy_model(self, vocab_size: int = 10):
        """Create a minimal model that returns deterministic logits."""

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, 16)
                self.head = torch.nn.Linear(16, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                h = self.embed(input_ids)
                logits = self.head(h)

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        return DummyModel()

    def test_output_shape(self):
        """Log-probs tensor has same shape as input_ids."""
        model = self._make_dummy_model()
        batch, seq_len = 2, 8
        input_ids = torch.randint(0, 10, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        response_start = torch.tensor([3, 5])

        lp = compute_log_probs_with_grad(model, input_ids, attention_mask, response_start)
        assert lp.shape == (batch, seq_len)

    def test_prompt_positions_are_zero(self):
        """Log-probs at prompt positions (before response_start) should be 0."""
        model = self._make_dummy_model()
        input_ids = torch.randint(0, 10, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        response_start = torch.tensor([5])

        lp = compute_log_probs_with_grad(model, input_ids, attention_mask, response_start)

        for i in range(5):
            assert lp[0, i].item() == 0.0, f"Position {i} should be zero (prompt)"

    def test_gradient_flows(self):
        """Gradients must flow back through compute_log_probs_with_grad."""
        model = self._make_dummy_model()
        input_ids = torch.randint(0, 10, (2, 6))
        attention_mask = torch.ones(2, 6, dtype=torch.long)
        response_start = torch.tensor([2, 3])

        lp = compute_log_probs_with_grad(model, input_ids, attention_mask, response_start)
        loss = lp.sum()
        loss.backward()

        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# grpo_loss
# ---------------------------------------------------------------------------


class TestGRPOLoss:
    """Tests for the clipped surrogate GRPO loss."""

    def test_no_old_log_probs_vanilla_pg(self):
        """When old_log_probs=None (num_iterations=1), ratio is 1.0 in forward."""
        policy_lp = torch.randn(4, 10, requires_grad=True) * 0.1
        adv = torch.randn(4)
        mask = torch.zeros(4, 10)
        mask[:, 3:] = 1.0

        loss, metrics = grpo_loss(policy_lp, None, adv, mask)
        assert loss.dim() == 0
        assert abs(metrics["mean_ratio"] - 1.0) < 1e-5

    def test_clipping_triggers_on_large_ratio(self):
        """Large policy/old divergence → some tokens get clipped."""
        policy_lp = torch.zeros(2, 10)
        old_lp = torch.full((2, 10), -5.0)  # large gap → ratio >> 1
        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 10)

        _, metrics = grpo_loss(policy_lp, old_lp, adv, mask, clip_eps=0.2)
        assert metrics["clip_ratio"] > 0, "Expected some clipping with large ratio"

    def test_kl_penalty_added(self):
        """With kl_coeff > 0, KL term appears in metrics and affects loss."""
        policy_lp = torch.randn(4, 10) * 0.1
        old_lp = policy_lp.detach().clone()
        ref_lp = policy_lp.detach().clone() - 0.5  # reference differs
        adv = torch.randn(4)
        mask = torch.zeros(4, 10)
        mask[:, 3:] = 1.0

        loss_no_kl, m1 = grpo_loss(policy_lp, old_lp, adv, mask, kl_coeff=0.0)
        loss_kl, m2 = grpo_loss(policy_lp, old_lp, adv, mask, kl_coeff=0.1, ref_log_probs=ref_lp)

        assert "kl" not in m1
        assert "kl" in m2
        assert m2["kl"] > 0  # policy > ref → positive KL approx
        assert not torch.isclose(loss_no_kl, loss_kl)

    def test_zero_advantages_zero_loss(self):
        """If all advantages are zero, surrogate loss is zero."""
        policy_lp = torch.randn(4, 10) * 0.1
        old_lp = policy_lp.detach().clone()
        adv = torch.zeros(4)
        mask = torch.ones(4, 10)

        loss, metrics = grpo_loss(policy_lp, old_lp, adv, mask)
        assert abs(loss.item()) < 1e-6
        assert abs(metrics["surrogate_loss"]) < 1e-6

    def test_token_level_vs_sequence_level(self):
        """Both token_level_loss modes execute without error on variable-length inputs."""
        policy_lp = torch.randn(4, 10) * 0.1
        old_lp = policy_lp.detach().clone() + torch.randn(4, 10) * 0.02
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(4, 10)
        mask[0, 2:] = 1.0  # 8 response tokens
        mask[1, 5:] = 1.0  # 5 response tokens
        mask[2, 3:] = 1.0  # 7 response tokens
        mask[3, 8:] = 1.0  # 2 response tokens

        loss_tok, _ = grpo_loss(policy_lp, old_lp, adv, mask, token_level_loss=True)
        loss_seq, _ = grpo_loss(policy_lp, old_lp, adv, mask, token_level_loss=False)

        assert loss_tok.dim() == 0
        assert loss_seq.dim() == 0

    def test_gradient_flows_through_loss(self):
        """GRPO loss must be differentiable w.r.t. policy_log_probs."""
        policy_lp = torch.randn(4, 10).requires_grad_(True)
        old_lp = policy_lp.detach().clone()
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(4, 10)

        loss, _ = grpo_loss(policy_lp, old_lp, adv, mask)
        loss.backward()

        assert policy_lp.grad is not None
        assert policy_lp.grad.abs().sum().item() > 0

    def test_gradient_flows_vanilla_pg_no_old(self):
        """old_log_probs=None: gradient must still flow through the policy_lp - policy_lp.detach() trick.

        This is the num_iterations=1 path. The ratio is 1.0 in forward but
        ∂ratio/∂θ = ∂log_π/∂θ (REINFORCE). If someone replaced the trick with
        zeros or a constant, this test catches it.
        """
        policy_lp = torch.randn(4, 10).requires_grad_(True)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(4, 10)

        loss, _ = grpo_loss(policy_lp, None, adv, mask)
        loss.backward()

        assert policy_lp.grad is not None
        assert policy_lp.grad.abs().sum().item() > 0, "Vanilla PG path must produce nonzero gradients"

    def test_vanilla_pg_gradient_direction(self):
        """Positive advantage → gradient should push log-probs up (decrease loss).

        For a single token with positive advantage and ratio=1 (no old_log_probs),
        ∂loss/∂log_π = -advantage. So the gradient on policy_lp should be negative
        (minimize loss = increase log_π via gradient descent).
        """
        policy_lp = torch.tensor([[0.0, 0.0, -1.0]]).requires_grad_(True)
        adv = torch.tensor([1.0])  # positive advantage
        mask = torch.tensor([[0.0, 0.0, 1.0]])  # only last position is response

        loss, _ = grpo_loss(policy_lp, None, adv, mask)
        loss.backward()

        # Gradient at response position should be negative (loss decreases when log_π increases).
        resp_grad = policy_lp.grad[0, 2].item()
        assert resp_grad < 0, f"Expected negative gradient for positive advantage, got {resp_grad}"

        # Gradient at prompt positions should be zero (masked out).
        assert policy_lp.grad[0, 0].item() == 0.0
        assert policy_lp.grad[0, 1].item() == 0.0

    def test_vanilla_pg_gradient_matches_standard_form(self):
        """old_log_probs=None gradient must equal the standard PPO formula with old=policy.

        When old_log_probs is explicitly set to a detached copy, the gradient
        should be identical to the None path (both compute ratio as
        exp(log_π - log_π.detach())).
        """
        torch.manual_seed(42)
        policy_lp_1 = torch.randn(3, 8).requires_grad_(True)
        policy_lp_2 = policy_lp_1.detach().clone().requires_grad_(True)
        adv = torch.tensor([1.0, -0.5, 0.3])
        mask = torch.zeros(3, 8)
        mask[:, 3:] = 1.0

        # Path 1: old_log_probs=None (vanilla PG trick)
        loss1, _ = grpo_loss(policy_lp_1, None, adv, mask)
        loss1.backward()

        # Path 2: old_log_probs=detached copy (explicit equivalent)
        old_lp = policy_lp_2.detach().clone()
        loss2, _ = grpo_loss(policy_lp_2, old_lp, adv, mask)
        loss2.backward()

        assert torch.allclose(policy_lp_1.grad, policy_lp_2.grad, atol=1e-6), (
            f"Vanilla PG gradient should match explicit old=policy gradient.\n"
            f"Max diff: {(policy_lp_1.grad - policy_lp_2.grad).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# cispo_loss
# ---------------------------------------------------------------------------


class TestCISPOLoss:
    """Tests for the CISPO (truncated IS-weighted policy gradient) loss."""

    def test_no_old_log_probs_reduces_to_reinforce(self):
        """When old_log_probs=None, ratio=1.0 everywhere → vanilla REINFORCE."""
        policy_lp = torch.randn(4, 10, requires_grad=True) * 0.1
        adv = torch.randn(4)
        mask = torch.zeros(4, 10)
        mask[:, 3:] = 1.0

        loss, metrics = cispo_loss(policy_lp, None, adv, mask)
        assert loss.dim() == 0
        assert abs(metrics["mean_ratio"] - 1.0) < 1e-5
        assert metrics["truncation_ratio"] == 0.0

    def test_gradient_flows_only_through_log_pi(self):
        """Gradient must flow through log_pi and be nonzero."""
        torch.manual_seed(42)
        policy_lp = torch.randn(4, 10, requires_grad=True)
        old_lp = policy_lp.detach().clone() + torch.randn(4, 10) * 0.3
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(4, 10)

        loss, _ = cispo_loss(policy_lp, old_lp, adv, mask)
        loss.backward()

        assert policy_lp.grad is not None
        assert policy_lp.grad.abs().sum().item() > 0

    def test_stop_gradient_on_ratio(self):
        """old_lp=policy.detach() (ratio=1.0) must produce same gradient as old_lp=None."""
        torch.manual_seed(42)
        policy_lp_1 = torch.randn(4, 10, requires_grad=True)
        policy_lp_2 = policy_lp_1.detach().clone().requires_grad_(True)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(4, 10)

        # Path 1: old_lp = None (explicit REINFORCE)
        loss1, _ = cispo_loss(policy_lp_1, None, adv, mask)
        loss1.backward()

        # Path 2: old_lp = detached policy (ratio=1.0, should match)
        old_lp = policy_lp_2.detach().clone()
        loss2, _ = cispo_loss(policy_lp_2, old_lp, adv, mask)
        loss2.backward()

        assert torch.allclose(policy_lp_1.grad, policy_lp_2.grad, atol=1e-6)

    def test_truncation_triggers_on_large_ratio(self):
        """Large policy/old divergence → some ratios get truncated."""
        policy_lp = torch.zeros(2, 10)
        old_lp = torch.full((2, 10), -5.0)  # large gap → ratio >> 1
        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 10)

        _, metrics = cispo_loss(policy_lp, old_lp, adv, mask, truncation_max=5.0)
        assert metrics["truncation_ratio"] > 0, "Expected some truncation with large ratio"

    def test_zero_advantages_zero_loss(self):
        """If all advantages are zero, surrogate loss is zero."""
        policy_lp = torch.randn(4, 10) * 0.1
        old_lp = policy_lp.detach().clone()
        adv = torch.zeros(4)
        mask = torch.ones(4, 10)

        loss, _ = cispo_loss(policy_lp, old_lp, adv, mask)
        assert abs(loss.item()) < 1e-6

    def test_token_level_vs_sequence_level(self):
        """Both token_level_loss modes execute without error."""
        policy_lp = torch.randn(4, 10) * 0.1
        old_lp = policy_lp.detach().clone() + torch.randn(4, 10) * 0.02
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.zeros(4, 10)
        mask[0, 2:] = 1.0
        mask[1, 5:] = 1.0
        mask[2, 3:] = 1.0
        mask[3, 8:] = 1.0

        loss_tok, _ = cispo_loss(policy_lp, old_lp, adv, mask, token_level_loss=True)
        loss_seq, _ = cispo_loss(policy_lp, old_lp, adv, mask, token_level_loss=False)

        assert loss_tok.dim() == 0
        assert loss_seq.dim() == 0

    def test_kl_penalty_added(self):
        """With kl_coeff > 0, KL term appears in metrics."""
        policy_lp = torch.randn(4, 10) * 0.1
        ref_lp = policy_lp.detach().clone() - 0.5
        adv = torch.randn(4)
        mask = torch.zeros(4, 10)
        mask[:, 3:] = 1.0

        _, m1 = cispo_loss(policy_lp, None, adv, mask, kl_coeff=0.0)
        _, m2 = cispo_loss(policy_lp, None, adv, mask, kl_coeff=0.1, ref_log_probs=ref_lp)

        assert "kl" not in m1
        assert "kl" in m2

    def test_cispo_matches_grpo_when_num_iterations_1(self):
        """With old_log_probs=None, both losses produce identical gradients."""
        torch.manual_seed(42)
        policy_lp_grpo = torch.randn(4, 10, requires_grad=True)
        policy_lp_cispo = policy_lp_grpo.detach().clone().requires_grad_(True)
        adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
        mask = torch.ones(4, 10)

        loss_g, _ = grpo_loss(policy_lp_grpo, None, adv, mask)
        loss_g.backward()

        loss_c, _ = cispo_loss(policy_lp_cispo, None, adv, mask)
        loss_c.backward()

        assert torch.allclose(policy_lp_grpo.grad, policy_lp_cispo.grad, atol=1e-5), (
            "With num_iterations=1, CISPO and GRPO should produce identical gradients"
        )
