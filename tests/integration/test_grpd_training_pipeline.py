"""Integration tests for OPDVR / GRPD RLVR Trainer, CLI, and DVC Pipeline."""

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.action_masking import assert_zero_observation_leakage
from ask_the_database.training.adapters.trl_grpo import (
    TRLGRPOAdapter,
    compute_group_normalized_rewards,
    compute_policy_loss,
)
from ask_the_database.training.losses import compute_grpd_token_rewards
from ask_the_database.training.privileged_teacher import PrivilegedSchemaTeacher
from scripts.train_rlvr import run_rlvr_training


def test_grpd_dry_run_pipeline(tmp_path: Path) -> None:
    """Validate full GRPD dry-run execution and artifact persistence."""
    out_dir = tmp_path / "grpd_dry_run_artifacts"
    cfg = OmegaConf.create(
        {
            "training": {
                "algorithm": "grpd",
                "output_dir": str(out_dir),
                "num_rollouts_per_prompt": 4,
                "max_steps": 3,
                "dry_run": True,
                "distillation": {
                    "enabled": True,
                    "teacher_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "share_base_model": True,
                    "privileged_schema": True,
                    "gating_mode": "relu",
                    "eps": 1e-8,
                    "max_ddl_tokens": 2500,
                },
            },
            "reward": {
                "float_tolerance": 0.0001,
                "use_test_suite": True,
                "num_mutations": 4,
            },
            "dry_run": True,
        }
    )

    telemetry = run_rlvr_training(cfg)

    assert telemetry["status"] == "dry_run_completed"
    assert telemetry["episodes_completed"] == 3
    assert len(telemetry["mean_rewards"]) == 3
    assert len(telemetry["group_advantages"]) == 3

    # Verify persisted artifacts
    assert (out_dir / "training_state.json").exists()
    assert (out_dir / "grpo_adapter_config.json").exists()

    state = json.loads((out_dir / "training_state.json").read_text(encoding="utf-8"))
    assert state["dry_run"] is True
    assert state["episodes"] == 3
    assert "last_step_metrics" in state
    metrics = state["last_step_metrics"]
    assert "train/r_zero_ratio" in metrics
    assert "train/type1_conflict_ratio" in metrics
    assert "train/type2_conflict_ratio" in metrics

    adapter_config = json.loads((out_dir / "grpo_adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["algorithm"] == "grpd"
    assert adapter_config["distillation_enabled"] is True
    assert adapter_config["gating_mode"] == "relu"
    assert adapter_config["share_base_model"] is True


def test_grpd_optimization_step_with_action_masking_and_telemetry(tmp_path: Path) -> None:
    """Verify optimization step with action loss masking, teacher logps, and conflict telemetry."""
    cfg = OmegaConf.create(
        {
            "training": {
                "algorithm": "grpd",
                "output_dir": str(tmp_path / "grpd_opt"),
                "distillation": {
                    "enabled": True,
                    "gating_mode": "relu",
                    "share_base_model": True,
                    "eps": 1e-8,
                },
            },
            "reward": {
                "float_tolerance": 0.0001,
                "use_test_suite": False,
            },
        }
    )
    rubric = SQLRewardRubric()
    adapter = TRLGRPOAdapter(cfg, rubric)

    # 4 completions with variable lengths
    batch_size = 4
    seq_len = 12

    # Construct action mask: tokens 0..5 are observations (mask=0.0), 6..11 are agent actions (mask=1.0)
    action_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    action_mask[:, 6:] = 1.0

    # Ensure no leakage on observation indices
    obs_indices = [0, 1, 2, 3, 4, 5]
    for row in range(batch_size):
        assert_zero_observation_leakage(action_mask[row], obs_indices)

    # Mock student & teacher logps
    student_logps = torch.randn((batch_size, seq_len), requires_grad=True)
    teacher_logps = student_logps.detach() + 0.2

    # Rewards: two positive, two zero/negative
    rewards = torch.tensor([1.0, 0.5, 0.0, -0.5], dtype=torch.float32)

    telemetry = compute_policy_loss(
        per_token_logps=student_logps,
        old_per_token_logps=teacher_logps,
        completion_mask=torch.ones_like(student_logps),
        rewards=rewards,
        group_size=4,
        config=adapter.grpo_config,
        action_mask=action_mask,
    )

    assert telemetry.algorithm == "grpd"
    assert telemetry.policy_loss != 0.0
    assert 0.0 <= telemetry.r_zero_ratio <= 1.0
    assert 0.0 <= telemetry.type1_conflict_ratio <= 1.0
    assert 0.0 <= telemetry.type2_conflict_ratio <= 1.0

    # Verify backpropagation through student logps
    telemetry.loss.backward()
    assert student_logps.grad is not None
    # Observation positions (0..5) must receive strictly ZERO gradient
    obs_grads = student_logps.grad[:, :6]
    assert torch.all(obs_grads == 0.0), "Observation tokens received non-zero gradients!"
    # Action positions (6..11) must receive non-zero gradients
    act_grads = student_logps.grad[:, 6:]
    assert torch.any(act_grads != 0.0), "Action tokens received zero gradients!"


def test_grpd_mutation_reward_gating_behavior() -> None:
    """Verify that verifiable mutation test suite rewards properly scale GRPD token rewards.

    When reward > 0 (candidate passes test suite mutations), advantage > 0, ReLU gating passes teacher signal.
    When reward <= 0 (candidate fails mutations / incorrect SQL), advantage <= 0, ReLU gating clamps distillation signal to zero.
    """
    student_logps = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    teacher_logps = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])

    # Candidate 0: passed mutation suite (reward=1.0)
    # Candidate 1: failed mutation suite (reward=0.0)
    raw_rewards = [1.0, 0.0]
    advantages = compute_group_normalized_rewards(raw_rewards, group_size=2)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)  # (2, 1)

    assert adv_tensor[0].item() > 0.0, "Passing candidate must have positive advantage"
    assert adv_tensor[1].item() < 0.0, "Failing candidate must have negative advantage"

    # Compute GRPD token rewards with ReLU gating
    token_rewards = compute_grpd_token_rewards(
        student_logprobs=student_logps,
        teacher_logprobs=teacher_logps,
        advantages=adv_tensor,
        gating_mode="relu",
    )

    # Passing candidate receives positive distillation signal
    assert torch.all(token_rewards[0] > 0.0)
    # Failing candidate is clamped to zero by ReLU gating
    assert torch.all(token_rewards[1] == 0.0)


def test_privileged_schema_teacher_integration_with_adapter() -> None:
    """Verify PrivilegedSchemaTeacher integration and adapter weight sharing."""
    cfg = OmegaConf.create(
        {
            "training": {
                "algorithm": "grpd",
                "distillation": {
                    "enabled": True,
                    "share_base_model": True,
                    "privileged_schema": True,
                },
            },
            "reward": {"use_test_suite": False},
        }
    )
    adapter = TRLGRPOAdapter(cfg)

    # Initialize mock neural network
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 16)

        def forward(self, input_ids, **kwargs):
            b, seq_len = input_ids.shape
            return type("Out", (), {"logits": torch.zeros((b, seq_len, 32))})()

    mock_model = MockModel()
    adapter.initialize_teacher(model=mock_model)

    assert hasattr(adapter, "teacher")
    assert isinstance(adapter.teacher, PrivilegedSchemaTeacher)
    assert adapter.teacher.is_shared_base is True
    footprint = adapter.teacher.get_parameter_memory_footprint(mock_model)
    assert footprint["weight_sharing_verified"] is True
    assert footprint["extra_base_parameters_allocated"] == 0
