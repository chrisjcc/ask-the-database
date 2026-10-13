"""Unit tests for Multi-Turn RLVR training entrypoint and pipeline execution."""

from pathlib import Path

from omegaconf import OmegaConf

from scripts.train_rlvr import run_rlvr_training


def test_rlvr_dry_run_pipeline_execution(tmp_path: Path) -> None:
    """Test run_rlvr_training executes multi-turn rollouts and saves artifacts in dry-run mode."""
    out_dir = tmp_path / "rlvr_output"
    cfg = OmegaConf.create(
        {
            "training": {
                "model_path": "models/sft/final",
                "output_dir": str(out_dir),
                "num_rollouts_per_prompt": 4,
                "max_turns": 4,
                "step_penalty": -0.05,
                "recovery_bonus": 0.3,
                "max_steps": 2,
                "dry_run": True,
            },
            "reward": {
                "float_tolerance": 0.0001,
                "use_test_suite": False,
            },
        }
    )

    telemetry = run_rlvr_training(cfg)

    assert telemetry["status"] == "dry_run_completed"
    assert telemetry["episodes_completed"] == 2
    assert len(telemetry["mean_rewards"]) == 2
    assert len(telemetry["group_advantages"]) == 2

    # Verify persisted state file
    state_file = out_dir / "training_state.json"
    assert state_file.exists()
