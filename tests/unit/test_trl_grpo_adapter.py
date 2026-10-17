"""Unit tests for TRLGRPOAdapter and group reward normalization."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip(
    "torch",
    reason="PyTorch is required for training and adapter tests. Install via 'make install' or 'pip install -e .[training]'.",
)

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import RolloutBatch
from ask_the_database.training.adapters.trl_grpo import (
    TRLGRPOAdapter,
    compute_group_normalized_rewards,
)


def test_compute_group_normalized_rewards_standard() -> None:
    """Verify standard zero-mean unit-variance group normalization."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = compute_group_normalized_rewards(rewards, group_size=4)
    assert len(advantages) == 4
    # Mean should be very close to 0.0
    mean_adv = sum(advantages) / 4
    assert abs(mean_adv) < 1e-6
    # Relative ordering preserved: higher reward has higher advantage
    assert advantages[3] > advantages[2] > advantages[1] > advantages[0]


def test_compute_group_normalized_rewards_zero_std() -> None:
    """Verify that constant rewards return 0.0 advantages without division by zero."""
    rewards = [1.0, 1.0, 1.0, 1.0]
    advantages = compute_group_normalized_rewards(rewards, group_size=4)
    assert advantages == [0.0, 0.0, 0.0, 0.0]


def test_compute_group_normalized_rewards_multiple_groups() -> None:
    """Verify that multiple groups are normalized independently."""
    # Group 1: [0.0, 1.0], Group 2: [10.0, 20.0]
    rewards = [0.0, 1.0, 10.0, 20.0]
    advantages = compute_group_normalized_rewards(rewards, group_size=2)
    assert len(advantages) == 4
    # In both groups, second element is higher than first
    assert advantages[1] > advantages[0]
    assert advantages[3] > advantages[2]
    # Symmetrical advantage in size-2 groups
    assert abs(advantages[0] + advantages[1]) < 1e-6
    assert abs(advantages[2] + advantages[3]) < 1e-6


def test_compute_group_normalized_rewards_empty() -> None:
    """Verify empty input returns empty list."""
    assert compute_group_normalized_rewards([], group_size=4) == []


def test_trl_grpo_adapter_compute_rewards(sample_sqlite_db: Path) -> None:
    """Verify TRLGRPOAdapter computes scalar rewards correctly using rubric."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": "/tmp/grpo_out",
                "learning_rate": 1e-5,
                "num_generations": 2,
                "max_completion_length": 256,
                "beta": 0.04,
            }
        }
    )
    rubric = SQLRewardRubric({"use_test_suite": False})
    adapter = TRLGRPOAdapter(cfg, rubric)

    db_str = str(sample_sqlite_db)
    batch = RolloutBatch(
        prompts=["Count users", "Count users"],
        completions=[
            "```sql\nSELECT COUNT(*) FROM users\n```",
            "This is not SQL",
        ],
        gold_sqls=[
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM users",
        ],
        db_paths=[db_str, db_str],
    )

    rewards = adapter.compute_rewards(batch)
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] < 0.0  # syntax penalty

    advantages = adapter.compute_advantages(batch, group_size=2)
    assert len(advantages) == 2
    assert advantages[0] > 0.0
    assert advantages[1] < 0.0


def test_trl_grpo_adapter_make_reward_function(sample_sqlite_db: Path) -> None:
    """Verify reward function callable conforms to TRL GRPOTrainer requirements."""
    cfg = OmegaConf.create({"training": {"output_dir": "/tmp/grpo"}})
    rubric = SQLRewardRubric({"use_test_suite": False})
    adapter = TRLGRPOAdapter(cfg, rubric)
    reward_fn = adapter.make_reward_function()

    db_str = str(sample_sqlite_db)
    # Broadcast single prompt gold & db to 2 completions
    completions = [
        "```sql\nSELECT name FROM users WHERE age > 25\n```",
        "```sql\nSELECT name FROM users WHERE age > 25\n```",
    ]
    rewards = reward_fn(
        completions=completions,
        gold_sqls=["SELECT name FROM users WHERE age > 25"],
        db_paths=[db_str],
    )
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(1.0)


def test_trl_grpo_adapter_train_step_telemetry(sample_sqlite_db: Path) -> None:
    """Verify train_step produces telemetry metrics."""
    cfg = OmegaConf.create({"training": {"output_dir": "/tmp/grpo", "num_generations": 2}})
    rubric = SQLRewardRubric({"use_test_suite": False})
    adapter = TRLGRPOAdapter(cfg, rubric)

    db_str = str(sample_sqlite_db)
    batch = RolloutBatch(
        prompts=["Q1", "Q2"],
        completions=["SELECT 1", "SELECT 1"],
        gold_sqls=["SELECT 1", "SELECT 1"],
        db_paths=[db_str, db_str],
    )
    metrics = adapter.train_step(batch)
    assert metrics["batch_size"] == 2.0
    assert "raw_mean_reward" in metrics
    assert "mean_advantage" in metrics


def test_train_baseline_grpo_dry_run(tmp_path: Path) -> None:
    """Verify scripts/train_baseline_grpo.py runs in dry-run mode."""
    from scripts.train_baseline_grpo import main

    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "models" / "grpo" / "final"),
                "learning_rate": 1e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_generations": 4,
                "max_completion_length": 128,
                "beta": 0.04,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
                "bf16": False,
                "max_steps": 1,
            },
            "reward": {
                "use_test_suite": False,
            },
            "dry_run": True,
        }
    )

    main(cfg)

    out_file = tmp_path / "models" / "grpo" / "final" / "grpo_adapter_config.json"
    assert out_file.is_file()
