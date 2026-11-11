"""OPDVR and GRPD mathematical loss engine for AskTheDatabase.

Implements On-Policy Distillation with Verifiable Reward (OPDVR) and Group Relative
Policy Distillation (GRPD) based on Lin et al. (arXiv:2608.24696, August 2026).

Key Mathematical Properties:
    1. Sign-Aligned ReLU Gating:
       R_GRPD(x_t) = A_hat * ReLU(sign(A_hat) * (log pi_T(x_t) - log pi_theta(x_t)))
    2. Proposition C.1 (Directional Alignment):
       <Delta_GRPD, Delta_RLVR> >= 0 for all active tokens.
    3. Proposition C.2 (Orthogonal Conflict Decomposition):
       Delta_OPD = Delta_OPDVR + Delta_conflict, with <Delta_conflict, Delta_RLVR> <= 0.
    4. Action Token Loss Masking:
       Loss updates apply strictly to student-generated tokens (thoughts and tool calls),
       guaranteeing zero observation leakage from SQLite sandbox executions.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

GatingMode = Literal["relu", "none", "inverse"]


def compute_opdvr_token_rewards(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    gating_mode: GatingMode = "relu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute token-level rewards for binary/scalar outcome OPDVR.

    Args:
        student_logprobs: Log-probabilities from student policy, shape (B, L).
        teacher_logprobs: Log-probabilities from teacher policy, shape (B, L).
        rewards: Outcome rewards, shape (B,) or (B, 1) or (B, L).
        gating_mode: Gating strategy:
            - 'relu': OPDVR parameter-free sign-aligned ReLU gate.
            - 'none': Vanilla un-gated on-policy distillation baseline.
            - 'inverse': Inverted ReLU gate (negative control ablation).
        eps: Small epsilon for numerical stability.

    Returns:
        Token-level reward tensor of shape (B, L).
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)  # (B, 1)

    # Token log-probability ratio: r_t = log pi_T(x_t) - log pi_theta(x_t)
    log_ratio = teacher_logprobs - student_logprobs

    if gating_mode == "relu":
        # R_OPDVR = R * ReLU(sign(R) * log_ratio)
        sign_r = torch.sign(rewards)
        token_rewards = rewards * F.relu(sign_r * log_ratio)
    elif gating_mode == "none":
        # Vanilla OPD: un-gated teacher log-ratio (no reward filtering or advantage weighting)
        token_rewards = log_ratio
    elif gating_mode == "inverse":
        # Inverse-gated negative control (ablation C5): keeps only conflict tokens
        sign_r = torch.sign(rewards)
        token_rewards = -rewards * F.relu(-sign_r * log_ratio)
    else:
        raise ValueError(
            f"Unknown gating_mode: {gating_mode}. Must be 'relu', 'none', or 'inverse'."
        )

    return token_rewards


def compute_grpd_token_rewards(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    gating_mode: GatingMode = "relu",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute token-level rewards for Group Relative Policy Distillation (GRPD).

    In GRPD, the group advantage A_hat scales the sign-aligned ReLU gate:
        R_GRPD(x_t) = A_hat * ReLU(sign(A_hat) * (log pi_T(x_t) - log pi_theta(x_t)))

    Args:
        student_logprobs: Log-probabilities from student policy, shape (B, L).
        teacher_logprobs: Log-probabilities from teacher policy, shape (B, L).
        advantages: Group-relative advantages A_hat, shape (B,) or (B, 1) or (B, L).
        gating_mode: Gating mode ('relu', 'none', or 'inverse').
        eps: Numerical stability constant.

    Returns:
        Token-level GRPD reward tensor of shape (B, L).
    """
    return compute_opdvr_token_rewards(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        rewards=advantages,
        gating_mode=gating_mode,
        eps=eps,
    )


def compute_conflict_diagnostics(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    token_rewards: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute diagnostic metrics for token conflicts and gating statistics.

    Diagnostics include:
        - r_zero_ratio: Fraction of active tokens where token reward is 0.
        - type1_conflict_ratio: Fraction of active tokens on positive-advantage trajectories
          where student is more confident than teacher (pi_theta > pi_T), protected by ReLU.
        - type2_conflict_ratio: Fraction of active tokens on negative-advantage trajectories
          where teacher is more confident than student (pi_T > pi_theta), suppressed by ReLU.
        - mean_log_ratio: Mean (log pi_T - log pi_theta) on active tokens.
        - active_token_count: Number of active action tokens in batch.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    # Broadcast advantages to (B, L)
    adv_expanded = advantages.expand_as(student_logprobs)
    mask = action_mask.bool()
    active_tokens = mask.sum().item()

    if active_tokens == 0:
        return {
            "r_zero_ratio": 0.0,
            "type1_conflict_ratio": 0.0,
            "type2_conflict_ratio": 0.0,
            "mean_log_ratio": 0.0,
            "active_token_count": 0.0,
            "mean_advantage": float(advantages.mean().item()) if advantages.numel() > 0 else 0.0,
        }

    log_ratio = teacher_logprobs - student_logprobs  # (B, L)

    # Type I Conflict: A > 0 and log_ratio < 0 (student more confident than teacher on positive trajectory)
    pos_adv_mask = (adv_expanded > eps) & mask
    pos_tokens = pos_adv_mask.sum().item()
    type1_mask = pos_adv_mask & (log_ratio < 0)
    type1_ratio = float((type1_mask.sum().item() / max(pos_tokens, 1)) if pos_tokens > 0 else 0.0)

    # Type II Conflict: A < -eps and log_ratio > 0 (teacher more confident than student on negative trajectory)
    neg_adv_mask = (adv_expanded < -eps) & mask
    neg_tokens = neg_adv_mask.sum().item()
    type2_mask = neg_adv_mask & (log_ratio > 0)
    type2_ratio = float((type2_mask.sum().item() / max(neg_tokens, 1)) if neg_tokens > 0 else 0.0)

    # Zero-gated ratio
    if token_rewards is not None:
        zero_reward_mask = (token_rewards.abs() < eps) & mask
        r_zero_ratio = float(zero_reward_mask.sum().item() / active_tokens)
    else:
        # Theoretical zero-gated condition for ReLU gate:
        # Gated when (sign(A) * log_ratio <= 0)
        sign_adv = torch.sign(adv_expanded)
        gated_mask = ((sign_adv * log_ratio) <= 0) & mask
        r_zero_ratio = float(gated_mask.sum().item() / active_tokens)

    mean_log_ratio = float(log_ratio[mask].mean().item())
    mean_adv = float(advantages.mean().item())

    return {
        "r_zero_ratio": r_zero_ratio,
        "type1_conflict_ratio": type1_ratio,
        "type2_conflict_ratio": type2_ratio,
        "mean_log_ratio": mean_log_ratio,
        "active_token_count": float(active_tokens),
        "mean_advantage": mean_adv,
    }


class GRPDLoss(nn.Module):
    """Group Relative Policy Distillation (GRPD) Loss.

    Computes policy gradient distillation loss:
        L_GRPD(theta) = - (1 / N_active) * sum_{t in active} R_GRPD(x_t) * log pi_theta(x_t)

    Where R_GRPD(x_t) is computed via sign-aligned ReLU gating.
    Gradients flow through student_logprobs, while R_GRPD is detached as a token reward weight.
    """

    def __init__(
        self,
        gating_mode: GatingMode = "relu",
        reduction: Literal["mean", "sequence_mean", "sum", "none"] = "mean",
        detach_reward: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """Initialize GRPDLoss.

        Args:
            gating_mode: 'relu' (standard GRPD), 'none' (vanilla OPD), or 'inverse' (ablation).
            reduction: Loss reduction strategy across batch and sequence tokens.
            detach_reward: Whether to detach token rewards from the autograd graph (default: True).
            eps: Epsilon constant for numerical stability.
        """
        super().__init__()
        self.gating_mode = gating_mode
        self.reduction = reduction
        self.detach_reward = detach_reward
        self.eps = eps

    def forward(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the GRPD distillation loss and diagnostic telemetry.

        Args:
            student_logprobs: Log-probabilities from student policy, shape (B, L).
            teacher_logprobs: Log-probabilities from teacher policy, shape (B, L).
            advantages: Group-relative advantages A_hat, shape (B,) or (B, 1) or (B, L).
            action_mask: Binary mask (1 for student actions/thoughts, 0 for prompts/observations), shape (B, L).

        Returns:
            Tuple of:
                loss: Scalar or per-element distillation loss tensor.
                diagnostics: Dictionary of metrics (r_zero_ratio, conflict ratios, etc.).
        """
        if student_logprobs.shape != teacher_logprobs.shape:
            raise ValueError(
                f"Shape mismatch between student_logprobs {student_logprobs.shape} "
                f"and teacher_logprobs {teacher_logprobs.shape}"
            )
        if action_mask.shape != student_logprobs.shape:
            raise ValueError(
                f"Shape mismatch between action_mask {action_mask.shape} "
                f"and student_logprobs {student_logprobs.shape}"
            )

        # Compute token-level GRPD rewards
        token_rewards = compute_grpd_token_rewards(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            gating_mode=self.gating_mode,
            eps=self.eps,
        )

        reward_weights = token_rewards.detach() if self.detach_reward else token_rewards

        # Per-token policy gradient objective: - R_GRPD(x_t) * log pi_theta(x_t)
        per_token_loss = -reward_weights * student_logprobs

        # Apply action token mask: zero out prompt tokens and environment observations
        mask = action_mask.to(per_token_loss.dtype)
        masked_loss = per_token_loss * mask

        # Loss reduction
        total_active_tokens = mask.sum().clamp(min=1.0)
        if self.reduction == "mean":
            loss = masked_loss.sum() / total_active_tokens
        elif self.reduction == "sequence_mean":
            seq_active = mask.sum(dim=-1).clamp(min=1.0)  # (B,)
            seq_loss = masked_loss.sum(dim=-1) / seq_active  # (B,)
            loss = seq_loss.mean()
        elif self.reduction == "sum":
            loss = masked_loss.sum()
        elif self.reduction == "none":
            loss = masked_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Compute telemetry diagnostics
        diagnostics = compute_conflict_diagnostics(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=advantages,
            action_mask=action_mask,
            token_rewards=token_rewards,
            eps=self.eps,
        )
        diagnostics["mean_loss"] = float(loss.detach().mean().item()) if loss.numel() > 0 else 0.0

        return loss, diagnostics


class OPDVRLoss(GRPDLoss):
    """On-Policy Distillation with Verifiable Reward (OPDVR) Loss.

    Specialized implementation of GRPDLoss for binary or continuous outcome-level
    verifier rewards R in {-1, +1} or [0, 1].
    """

    def __init__(
        self,
        gating_mode: GatingMode = "relu",
        reduction: Literal["mean", "sequence_mean", "sum", "none"] = "mean",
        detach_reward: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            gating_mode=gating_mode,
            reduction=reduction,
            detach_reward=detach_reward,
            eps=eps,
        )

    def forward(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute OPDVR loss with outcome verifier rewards."""
        return super().forward(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            advantages=rewards,
            action_mask=action_mask,
        )


__all__ = [
    "GatingMode",
    "compute_opdvr_token_rewards",
    "compute_grpd_token_rewards",
    "compute_conflict_diagnostics",
    "GRPDLoss",
    "OPDVRLoss",
]
