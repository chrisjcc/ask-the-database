"""Unit and regression tests for GRPO, DAPO, and GSPO policy optimization objectives.

Verifies:
1. Dynamic Sampling Masking (zero-variance groups yield zero gradient).
2. DAPO Asymmetric Clipping (expansion up to 1+clip_high, lower bound 1-clip_low).
3. GSPO Sequence Invariance (all tokens in trajectory share identical importance weight).
4. Backward pass gradient stability across GRPO, DAPO, and GSPO.
5. GRPOTrainingConfig instantiation and Hydra compatibility.
"""

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip(
    "torch",
    reason="PyTorch is required for policy optimization tests.",
)

from ask_the_database.training.adapters.trl_grpo import (
    GRPOTrainingConfig,
    TRLGRPOAdapter,
    compute_policy_loss,
)


def test_grpo_training_config_defaults() -> None:
    """Verify default hyperparameters of GRPOTrainingConfig."""
    cfg = GRPOTrainingConfig()
    assert cfg.algorithm == "dapo"
    assert cfg.enable_dynamic_sampling is True
    assert cfg.min_reward_std == pytest.approx(1e-4)
    assert cfg.dapo_clip_low == pytest.approx(0.20)
    assert cfg.dapo_clip_high == pytest.approx(0.28)
    assert cfg.gspo_seq_epsilon == pytest.approx(3e-4)
    assert cfg.epsilon == pytest.approx(0.20)


def test_grpo_training_config_from_dict_config() -> None:
    """Verify loading GRPOTrainingConfig from Hydra DictConfig and dicts."""
    hydra_dict = {
        "training": {
            "algorithm": "gspo",
            "enable_dynamic_sampling": False,
            "min_reward_std": 1e-3,
            "dapo_clip_low": 0.15,
            "dapo_clip_high": 0.35,
            "gspo_seq_epsilon": 5e-4,
            "learning_rate": 2e-5,
            "num_generations": 4,
        }
    }
    cfg = GRPOTrainingConfig.from_dict_config(OmegaConf.create(hydra_dict))
    assert cfg.algorithm == "gspo"
    assert cfg.enable_dynamic_sampling is False
    assert cfg.min_reward_std == pytest.approx(1e-3)
    assert cfg.dapo_clip_low == pytest.approx(0.15)
    assert cfg.dapo_clip_high == pytest.approx(0.35)
    assert cfg.gspo_seq_epsilon == pytest.approx(5e-4)
    assert cfg.learning_rate == pytest.approx(2e-5)
    assert cfg.num_generations == 4


def test_dynamic_sampling_masking() -> None:
    """Confirm prompt groups with zero variance yield strictly zero gradients."""
    config = GRPOTrainingConfig(algorithm="dapo", enable_dynamic_sampling=True, min_reward_std=1e-4)

    # 2 groups of size 4: Group 0 is zero-variance [0, 0, 0, 0], Group 1 is non-zero [0, 1, 0, 1]
    b, t = 8, 5
    per_token_logps = torch.zeros((b, t), requires_grad=True)
    old_logps = torch.zeros((b, t))
    mask = torch.ones((b, t))
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    telemetry = compute_policy_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_logps,
        completion_mask=mask,
        rewards=rewards,
        group_size=4,
        config=config,
    )

    assert telemetry.total_groups == 2
    assert telemetry.active_groups == 1
    assert telemetry.filtered_groups_ratio == pytest.approx(0.5)

    telemetry.loss.backward()
    grad = per_token_logps.grad
    assert grad is not None

    # Group 0 (indices 0..3) MUST have strictly zero gradients
    group_0_grad = grad[:4, :]
    assert torch.all(group_0_grad == 0.0), "Zero-variance group produced non-zero gradient!"

    # Group 1 (indices 4..7) MUST have non-zero gradients
    group_1_grad = grad[4:, :]
    assert torch.any(group_1_grad != 0.0), "Active group produced zero gradient!"


def test_dapo_asymmetric_clipping() -> None:
    """Verify DAPO asymmetric clipping bounds [1 - clip_low, 1 + clip_high]."""
    clip_low = 0.20
    clip_high = 0.28
    config = GRPOTrainingConfig(
        algorithm="dapo",
        dapo_clip_low=clip_low,
        dapo_clip_high=clip_high,
        enable_dynamic_sampling=False,
    )

    b, t = 2, 4
    # Advantage > 0: reward = [1.0, 0.0] -> advantages[0] > 0, advantages[1] < 0
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)
    mask = torch.ones((b, t))

    # Test positive advantage: token_ratio exceeds 1 + clip_high (e.g. ratio = 1.5 > 1.28)
    log_ratio_high = torch.full((b, t), torch.log(torch.tensor(1.50)).item())
    old_logps = torch.zeros((b, t))
    per_token_logps = old_logps + log_ratio_high

    telemetry = compute_policy_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_logps,
        completion_mask=mask,
        rewards=rewards,
        group_size=2,
        config=config,
    )

    # Positive advantage rollout (index 0) should be clipped at upper bound (1.28)
    # Negative advantage rollout (index 1) has ratio = 1.5 > 1.0, advantage < 0 -> surr1 < surr2 -> not low clipped
    assert telemetry.clip_fraction_high > 0.0

    # Test negative advantage: token_ratio drops below 1 - clip_low (e.g. ratio = 0.60 < 0.80)
    log_ratio_low = torch.full((b, t), torch.log(torch.tensor(0.60)).item())
    per_token_logps_low = old_logps + log_ratio_low

    telemetry_low = compute_policy_loss(
        per_token_logps=per_token_logps_low,
        old_per_token_logps=old_logps,
        completion_mask=mask,
        rewards=rewards,
        group_size=2,
        config=config,
    )

    # Negative advantage rollout (index 1) should be clipped at lower bound (0.80)
    assert telemetry_low.clip_fraction_low > 0.0


def test_gspo_sequence_invariance() -> None:
    """Verify that all tokens within a trajectory share identical gradients under GSPO."""
    config = GRPOTrainingConfig(
        algorithm="gspo",
        gspo_seq_epsilon=0.10,
        enable_dynamic_sampling=False,
    )

    b, t = 2, 6
    per_token_logps = torch.zeros((b, t), requires_grad=True)
    old_logps = torch.zeros((b, t))
    mask = torch.ones((b, t))
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)

    telemetry = compute_policy_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_logps,
        completion_mask=mask,
        rewards=rewards,
        group_size=2,
        config=config,
    )

    telemetry.loss.backward()
    grad = per_token_logps.grad
    assert grad is not None

    # Check trajectory 0: all tokens must receive the EXACT same gradient
    traj_0_tokens = grad[0, :]
    first_token_grad = traj_0_tokens[0].item()
    for tok_idx in range(1, t):
        assert traj_0_tokens[tok_idx].item() == pytest.approx(first_token_grad, rel=1e-5), (
            f"Token {tok_idx} gradient diverged from token 0 under GSPO!"
        )

    # Check trajectory 1: all tokens must also receive the exact same gradient
    traj_1_tokens = grad[1, :]
    first_tok_1 = traj_1_tokens[0].item()
    for tok_idx in range(1, t):
        assert traj_1_tokens[tok_idx].item() == pytest.approx(first_tok_1, rel=1e-5)


@pytest.mark.parametrize("algo", ["grpo", "dapo", "gspo"])
def test_smoke_rl_step_all_algorithms(algo: str) -> None:
    """Verify numerical stability and gradient propagation across all three algorithms."""
    config = GRPOTrainingConfig(algorithm=algo, enable_dynamic_sampling=True)  # type: ignore[arg-type]

    b, t = 4, 10
    per_token_logps = torch.randn((b, t), requires_grad=True)
    old_logps = torch.randn((b, t))
    mask = torch.ones((b, t))
    # 1 group of 4 with varying rewards
    rewards = torch.tensor([0.2, 0.8, 0.5, 0.9], dtype=torch.float32)

    telemetry = compute_policy_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_logps,
        completion_mask=mask,
        rewards=rewards,
        group_size=4,
        config=config,
    )

    assert not torch.isnan(telemetry.loss)
    assert not torch.isinf(telemetry.loss)
    assert telemetry.effective_tokens > 0

    telemetry.loss.backward()
    assert per_token_logps.grad is not None
    assert not torch.isnan(per_token_logps.grad).any()
    assert not torch.isinf(per_token_logps.grad).any()


def test_trl_grpo_adapter_train_step_telemetry() -> None:
    """Verify that TRLGRPOAdapter computes and returns DAPO/GSPO telemetry in train_step."""
    from ask_the_database.rewards.rubric import SQLRewardRubric
    from ask_the_database.training.adapters.base import RolloutBatch

    config = GRPOTrainingConfig(algorithm="dapo", num_generations=2)
    rubric = SQLRewardRubric({"use_test_suite": False})
    adapter = TRLGRPOAdapter(config, rubric)

    batch = RolloutBatch(
        prompts=["Count users", "Count users"],
        completions=[
            "```sql\nSELECT COUNT(*) FROM users\n```",
            "```sql\nSELECT * FROM users\n```",
        ],
        gold_sqls=[
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM users",
        ],
        db_paths=["dummy.sqlite", "dummy.sqlite"],
    )

    metrics = adapter.train_step(batch)
    assert "train/filtered_groups_ratio" in metrics
    assert "train/effective_tokens" in metrics
    assert "train/clip_fraction_low" in metrics
    assert "train/clip_fraction_high" in metrics
