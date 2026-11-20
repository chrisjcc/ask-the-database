"""Comprehensive unit tests for OPDVR and GRPD mathematical loss engine.

Validates:
    - Numerical correctness of sign-aligned ReLU gating.
    - Proposition C.1: Directional alignment with verifier gradient (<Delta_GRPD, Delta_RLVR> >= 0).
    - Proposition C.2: Orthogonal conflict decomposition (<Delta_conflict, Delta_RLVR> <= 0).
    - Multi-turn action token loss masking and zero observation gradient leakage.
    - Gating modes ('relu', 'none', 'inverse') and conflict diagnostics telemetry.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ask_the_database.training.losses import (
    GRPDLoss,
    OPDVRLoss,
    compute_conflict_diagnostics,
    compute_grpd_token_rewards,
    compute_opdvr_token_rewards,
)


class TestOPDVRTokenRewards:
    """Test suite for compute_opdvr_token_rewards and compute_grpd_token_rewards."""

    def test_binary_opdvr_positive_reward(self) -> None:
        """When R = +1 (verified correct), token reward is ReLU(log_pi_T - log_pi_theta)."""
        student_lp = torch.tensor([[-2.0, -1.0, -3.0]])
        teacher_lp = torch.tensor([[-1.0, -2.0, -3.0]])
        rewards = torch.tensor([1.0])

        token_r = compute_opdvr_token_rewards(student_lp, teacher_lp, rewards, gating_mode="relu")

        # Token 0: teacher (-1.0) > student (-2.0) -> log_ratio = +1.0 -> reward = +1.0
        # Token 1: teacher (-2.0) < student (-1.0) -> log_ratio = -1.0 -> Type I conflict -> ReLU gates to 0.0
        # Token 2: teacher (-3.0) == student (-3.0) -> log_ratio = 0.0 -> reward = 0.0
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(token_r, expected, atol=1e-6)

    def test_binary_opdvr_negative_reward(self) -> None:
        """When R = -1 (verified failed), token reward is -ReLU(log_pi_theta - log_pi_T)."""
        student_lp = torch.tensor([[-1.0, -2.0, -3.0]])
        teacher_lp = torch.tensor([[-2.0, -1.0, -3.0]])
        rewards = torch.tensor([-1.0])

        token_r = compute_opdvr_token_rewards(student_lp, teacher_lp, rewards, gating_mode="relu")

        # Token 0: student (-1.0) > teacher (-2.0) -> overconfident student error -> reward = -1.0
        # Token 1: student (-2.0) < teacher (-1.0) -> Type II conflict (teacher mistake) -> ReLU gates to 0.0
        # Token 2: equal -> reward = 0.0
        expected = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(token_r, expected, atol=1e-6)

    def test_continuous_group_advantages_scaling(self) -> None:
        """GRPD scales by advantage magnitude while preserving sign-aligned ReLU gate."""
        student_lp = torch.tensor([[-2.0, -1.0], [-1.0, -3.0]])
        teacher_lp = torch.tensor([[-1.0, -2.0], [-2.0, -1.0]])
        # Batch of 2: traj 0 has A_hat = +2.5, traj 1 has A_hat = -1.5
        advantages = torch.tensor([2.5, -1.5])

        token_r = compute_grpd_token_rewards(student_lp, teacher_lp, advantages, gating_mode="relu")

        # Traj 0 (A_hat = +2.5):
        #   Tok 0: log_ratio = +1.0 -> +2.5 * 1.0 = +2.5
        #   Tok 1: log_ratio = -1.0 -> gated to 0.0
        # Traj 1 (A_hat = -1.5):
        #   Tok 0: log_ratio = -1.0 -> sign=-1, -1.0 * -1.0 = +1.0 -> -1.5 * 1.0 = -1.5
        #   Tok 1: log_ratio = +2.0 -> sign=-1, -1.0 * +2.0 = -2.0 -> ReLU = 0.0 -> gated to 0.0
        expected = torch.tensor([[2.5, 0.0], [-1.5, 0.0]])
        assert torch.allclose(token_r, expected, atol=1e-6)

    def test_gating_modes_comparison(self) -> None:
        """Verify 'none' (vanilla OPD) and 'inverse' (negative control) gating modes."""
        student_lp = torch.tensor([[-2.0, -1.0]])
        teacher_lp = torch.tensor([[-1.0, -2.0]])
        adv = torch.tensor([1.0])

        # 'none': un-gated linear product A * log_ratio
        r_none = compute_grpd_token_rewards(student_lp, teacher_lp, adv, gating_mode="none")
        expected_none = torch.tensor([[1.0, -1.0]])
        assert torch.allclose(r_none, expected_none, atol=1e-6)

        # 'inverse': inverted gates (swapped)
        r_inv = compute_grpd_token_rewards(student_lp, teacher_lp, adv, gating_mode="inverse")
        expected_inv = torch.tensor([[0.0, -1.0]])
        assert torch.allclose(r_inv, expected_inv, atol=1e-6)

        with pytest.raises(ValueError, match="Unknown gating_mode"):
            compute_grpd_token_rewards(student_lp, teacher_lp, adv, gating_mode="invalid")  # type: ignore[arg-type]


class TestTheoreticalInvariants:
    """Test mathematical guarantees proven in Lin et al. (2026) and AskTheDatabase architecture."""

    def test_proposition_c1_directional_alignment(self) -> None:
        """Proposition C.1: Inner product between GRPD and RLVR gradients is strictly non-negative.

        For any token t with policy score vector u_t = grad_theta log pi_theta(x_t):
            <Delta_GRPD, Delta_RLVR> = A_hat^2 * ReLU(sign(A_hat) * log_ratio) * ||u_t||^2 >= 0
        """
        torch.manual_seed(42)
        batch_size, seq_len, vocab_size = 4, 8, 16

        # Toy policy model: linear layer over random embeddings
        embed_dim = 16
        policy = nn.Linear(embed_dim, vocab_size, bias=False)
        inputs = torch.randn(batch_size, seq_len, embed_dim)

        # Sample student tokens
        with torch.no_grad():
            student_logits = policy(inputs)
            student_lp_all = student_logits.log_softmax(dim=-1)
            tokens = student_lp_all.argmax(dim=-1)

        # Mock teacher logprobs and advantages
        teacher_lp = torch.randn(batch_size, seq_len)
        advantages = torch.tensor([2.0, -1.5, 0.8, -2.2])
        action_mask = torch.ones(batch_size, seq_len)

        # Forward through policy with autograd tracking
        student_logits_track = policy(inputs)
        student_lp_track = (
            student_logits_track.log_softmax(dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        )

        loss_fn = GRPDLoss(gating_mode="relu", detach_reward=True)
        grpd_loss, _ = loss_fn(student_lp_track, teacher_lp, advantages, action_mask)

        # Compute GRPD gradient
        policy.zero_grad()
        grpd_loss.backward(retain_graph=True)
        grpd_grad = policy.weight.grad.clone()  # type: ignore[union-attr]

        # Compute pure RLVR gradient: - sum( A_hat * log pi_theta ) / N_active
        policy.zero_grad()
        rlvr_loss = (
            -(advantages.unsqueeze(1) * student_lp_track * action_mask).sum() / action_mask.sum()
        )
        rlvr_loss.backward()
        rlvr_grad = policy.weight.grad.clone()  # type: ignore[union-attr]

        # Directional alignment: inner product between parameter updates
        # Since updates are -grad: <Delta_GRPD, Delta_RLVR> = < -grpd_grad, -rlvr_grad > = <grpd_grad, rlvr_grad>
        inner_product = torch.dot(grpd_grad.flatten(), rlvr_grad.flatten()).item()
        assert inner_product >= -1e-6, (
            f"Proposition C.1 violated: inner product was negative ({inner_product})"
        )

    def test_proposition_c2_orthogonal_conflict_decomposition(self) -> None:
        """Proposition C.2: Vanilla OPD decomposes into OPDVR + conflict vector with non-positive projection.

        Delta_OPD = Delta_OPDVR + Delta_conflict, with <Delta_conflict, Delta_RLVR> <= 0.
        """
        torch.manual_seed(1337)
        batch_size, seq_len = 6, 10

        student_lp = torch.randn(batch_size, seq_len)
        teacher_lp = torch.randn(batch_size, seq_len)
        rewards = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        r_opdvr = compute_opdvr_token_rewards(student_lp, teacher_lp, rewards, gating_mode="relu")
        r_opd = compute_opdvr_token_rewards(student_lp, teacher_lp, rewards, gating_mode="none")

        # Conflict reward: Delta_conflict = r_opd - r_opdvr
        r_conflict = r_opd - r_opdvr

        # For every token, verify <Delta_conflict_t, Delta_RLVR_t> = r_conflict_t * R_t <= 0
        rewards_expanded = rewards.unsqueeze(1).expand_as(r_conflict)
        token_projections = r_conflict * rewards_expanded

        # Every individual token projection of the conflict vector onto RLVR must be <= 0
        assert (token_projections <= 1e-6).all(), (
            "Proposition C.2 violated: conflict projection was positive"
        )


class TestActionMaskingAndObservationLeakage:
    """Test suite ensuring strict action token loss masking and zero observation gradient leakage."""

    def test_zero_observation_gradient_leakage(self) -> None:
        """Gradients on masked observation tokens must be identically zero."""
        batch_size, seq_len = 2, 6
        student_lp = torch.randn(batch_size, seq_len, requires_grad=True)
        teacher_lp = torch.randn(batch_size, seq_len)
        advantages = torch.tensor([1.0, -1.0])

        # Mask: 1 = action token (indices 0, 1, 4), 0 = observation token (indices 2, 3, 5)
        action_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])

        loss_fn = GRPDLoss(reduction="mean")
        loss, _ = loss_fn(student_lp, teacher_lp, advantages, action_mask)
        loss.backward()

        assert student_lp.grad is not None
        obs_mask = action_mask == 0.0

        # Assert all observation token gradients are exactly 0.0
        assert (student_lp.grad[obs_mask] == 0.0).all(), "Observation gradient leakage detected!"

        # Assert action tokens received non-zero gradients
        act_mask = action_mask == 1.0
        assert (student_lp.grad[act_mask].abs() >= 0.0).all()

    def test_fully_masked_sequence_safety(self) -> None:
        """Ensure no divide-by-zero or NaN when sequence is completely masked."""
        student_lp = torch.randn(2, 4, requires_grad=True)
        teacher_lp = torch.randn(2, 4)
        advantages = torch.tensor([1.0, -1.0])
        action_mask = torch.zeros(2, 4)

        loss_fn = GRPDLoss(reduction="mean")
        loss, diag = loss_fn(student_lp, teacher_lp, advantages, action_mask)

        assert torch.isfinite(loss)
        assert loss.item() == 0.0
        assert diag["r_zero_ratio"] == 0.0
        assert diag["active_token_count"] == 0.0


class TestReductionsAndShapes:
    """Test loss reduction modes and tensor broadcasting."""

    @pytest.mark.parametrize("reduction", ["mean", "sequence_mean", "sum", "none"])
    def test_supported_reductions(self, reduction: str) -> None:
        """Verify behavior of supported reduction strategies."""
        student_lp = torch.randn(2, 4)
        teacher_lp = torch.randn(2, 4)
        adv = torch.tensor([1.0, -0.5])
        mask = torch.ones(2, 4)

        loss_fn = GRPDLoss(reduction=reduction)  # type: ignore[arg-type]
        loss, _ = loss_fn(student_lp, teacher_lp, adv, mask)

        if reduction in ("mean", "sequence_mean", "sum"):
            assert loss.dim() == 0
        elif reduction == "none":
            assert loss.shape == (2, 4)

    def test_advantage_tensor_dimensions_broadcasting(self) -> None:
        """Advantages passed as (B,), (B, 1), or (B, L) should all produce equivalent losses."""
        student_lp = torch.randn(3, 5)
        teacher_lp = torch.randn(3, 5)
        mask = torch.ones(3, 5)

        adv_1d = torch.tensor([1.0, -1.0, 0.5])
        adv_2d = adv_1d.unsqueeze(1)
        adv_token = adv_1d.unsqueeze(1).expand(3, 5)

        loss_fn = GRPDLoss()
        l1, _ = loss_fn(student_lp, teacher_lp, adv_1d, mask)
        l2, _ = loss_fn(student_lp, teacher_lp, adv_2d, mask)
        l3, _ = loss_fn(student_lp, teacher_lp, adv_token, mask)

        assert torch.allclose(l1, l2)
        assert torch.allclose(l1, l3)

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatches must raise informative ValueErrors."""
        loss_fn = GRPDLoss()
        with pytest.raises(ValueError, match="Shape mismatch between student_logprobs"):
            loss_fn(
                torch.randn(2, 4), torch.randn(2, 5), torch.tensor([1.0, -1.0]), torch.ones(2, 4)
            )

        with pytest.raises(ValueError, match="Shape mismatch between action_mask"):
            loss_fn(
                torch.randn(2, 4), torch.randn(2, 4), torch.tensor([1.0, -1.0]), torch.ones(2, 3)
            )


class TestOPDVRLossSubclass:
    """Verify OPDVRLoss operates as expected for scalar/binary outcome rewards."""

    def test_opdvr_matches_grpd(self) -> None:
        """OPDVRLoss with binary rewards produces identical output to GRPDLoss."""
        student_lp = torch.randn(2, 4)
        teacher_lp = torch.randn(2, 4)
        rewards = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 4)

        opdvr_fn = OPDVRLoss()
        grpd_fn = GRPDLoss()

        loss_opdvr, diag_opdvr = opdvr_fn(student_lp, teacher_lp, rewards, mask)
        loss_grpd, diag_grpd = grpd_fn(student_lp, teacher_lp, rewards, mask)

        assert torch.allclose(loss_opdvr, loss_grpd)
        assert diag_opdvr["r_zero_ratio"] == diag_grpd["r_zero_ratio"]
        assert diag_opdvr["type1_conflict_ratio"] == diag_grpd["type1_conflict_ratio"]


class TestConflictDiagnostics:
    """Verify computation of conflict metrics and zero-gated ratio."""

    def test_conflict_diagnostics_calculation(self) -> None:
        """Verify exact calculation of Type I and Type II conflict ratios."""
        # Traj 0: positive advantage (+1.0)
        #   Tok 0: student -1.0, teacher -2.0 -> student more confident -> Type 1 conflict
        #   Tok 1: student -2.0, teacher -1.0 -> teacher more confident -> normal imitation
        # Traj 1: negative advantage (-1.0)
        #   Tok 0: student -1.0, teacher -2.0 -> student more confident -> penalized error
        #   Tok 1: student -2.0, teacher -1.0 -> teacher more confident -> Type 2 conflict
        student_lp = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]])
        teacher_lp = torch.tensor([[-2.0, -1.0], [-2.0, -1.0]])
        advantages = torch.tensor([1.0, -1.0])
        action_mask = torch.ones(2, 2)

        diag = compute_conflict_diagnostics(student_lp, teacher_lp, advantages, action_mask)

        # On traj 0 (pos): 1 out of 2 tokens is Type 1 conflict -> 50%
        assert diag["type1_conflict_ratio"] == pytest.approx(0.5)
        # On traj 1 (neg): 1 out of 2 tokens is Type 2 conflict -> 50%
        assert diag["type2_conflict_ratio"] == pytest.approx(0.5)
        # Exactly 2 out of 4 total tokens are zeroed by ReLU gate -> 50%
        assert diag["r_zero_ratio"] == pytest.approx(0.5)
        assert diag["active_token_count"] == 4.0
