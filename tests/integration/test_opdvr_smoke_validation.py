"""Integration tests for OPDVR / GRPD Smoke Validation and Telemetry Logging (Phase 4).

Validates:
1. Full end-to-end execution of run_opdvr_smoke_validation across multiple steps.
2. Adjudication of all 4 quantitative decision gates:
   - Gate 1: Zero runtime errors.
   - Gate 2: r_zero in theoretical bounds [0.40, 0.50].
   - Gate 3: Gradient norm > 0.01 (non-vanishing).
   - Gate 4: Memory leak-free (< 25 MB delta).
3. Telemetry dictionary serialization and key completeness for W&B / Weave.
4. Mathematical verification of zero observation leakage in policy gradients.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ask_the_database.training.adapters.trl_grpo import (
    GRPOTrainingConfig,
    PolicyLossTelemetry,
    compute_policy_loss,
)
from scripts.run_opdvr_smoke_validation import (
    load_representative_queries,
    run_opdvr_smoke_validation,
)


def test_load_representative_queries() -> None:
    """Verify representative queries load cleanly with both gold and flawed SQL."""
    queries = load_representative_queries(count=50)
    assert len(queries) == 50
    for q in queries:
        assert "question" in q and len(q["question"]) > 0
        assert "gold_sql" in q and len(q["gold_sql"]) > 0
        assert "flawed_sql" in q and len(q["flawed_sql"]) > 0
        assert q["db_id"] == "seed_fixtures"


def test_opdvr_smoke_validation_execution(tmp_path: Path) -> None:
    """Execute smoke validation across 5 steps and verify all decision gates pass."""
    output_report = tmp_path / "opdvr_smoke_report.json"
    report = run_opdvr_smoke_validation(
        num_steps=5,
        num_queries=10,
        group_size=4,
        output_path=output_report,
        wandb_enabled=False,
        weave_enabled=False,
    )

    assert report["status"] == "PASSED"
    assert report["num_steps"] == 5
    assert report["total_rollouts"] == 20

    telemetry = report["telemetry_summary"]
    assert 0.40 <= telemetry["mean_r_zero_ratio"] <= 0.50
    assert telemetry["r_zero_in_theoretical_bounds"] is True
    assert telemetry["mean_grad_norm"] > 0.01
    assert telemetry["non_vanishing_gradient_verified"] is True
    assert telemetry["memory_growth_mb"] < 25.0
    assert telemetry["numerical_stability"] == "STABLE"

    gates = report["decision_gates"]
    assert gates["gate1_zero_errors"] is True
    assert gates["gate2_r_zero_bounded"] is True
    assert gates["gate3_gradient_non_vanishing"] is True
    assert gates["gate4_memory_leak_free"] is True

    assert output_report.exists()
    loaded_data = json.loads(output_report.read_text(encoding="utf-8"))
    assert loaded_data["status"] == "PASSED"
    assert len(loaded_data["step_telemetry"]) == 5


def test_telemetry_dictionary_keys() -> None:
    """Verify PolicyLossTelemetry produces all required W&B and Weave telemetry keys."""
    telemetry = PolicyLossTelemetry(
        loss=torch.tensor(0.042),
        policy_loss=0.042,
        filtered_groups_ratio=0.0,
        effective_tokens=80,
        active_groups=1,
        total_groups=1,
        algorithm="grpd",
        r_zero_ratio=0.45,
        type1_conflict_ratio=0.45,
        type2_conflict_ratio=0.45,
        grad_norm=0.085,
        tool_interaction_efficiency=0.833,
    )

    data = telemetry.to_dict()

    expected_keys = [
        "loss",
        "policy_loss",
        "train/loss",
        "train/policy_loss",
        "train/r_zero_ratio",
        "train/type1_conflict_ratio",
        "train/type2_conflict_ratio",
        "train/grad_norm",
        "telemetry/r_zero_ratio",
        "telemetry/type1_conflict_ratio",
        "telemetry/type2_conflict_ratio",
        "telemetry/grad_norm",
        "telemetry/tool_interaction_efficiency",
    ]

    for k in expected_keys:
        assert k in data, f"Missing telemetry key: {k}"

    assert data["telemetry/r_zero_ratio"] == 0.45
    assert data["telemetry/type1_conflict_ratio"] == 0.45
    assert data["telemetry/type2_conflict_ratio"] == 0.45
    assert data["telemetry/grad_norm"] == 0.085
    assert data["telemetry/tool_interaction_efficiency"] == 0.833


def test_zero_observation_gradient_leakage() -> None:
    """Verify observation tokens receive strictly zero gradient during GRPD backward pass."""
    group_size = 4
    seq_len = 20
    obs_len = 8

    action_mask = torch.zeros((group_size, seq_len), dtype=torch.float32)
    action_mask[:, obs_len:] = 1.0

    student_logps = torch.randn((group_size, seq_len), requires_grad=True, dtype=torch.float32)
    teacher_logps = student_logps.detach() + 0.2
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)

    cfg = GRPOTrainingConfig(algorithm="grpd", gating_mode="relu")
    telemetry = compute_policy_loss(
        per_token_logps=student_logps,
        old_per_token_logps=teacher_logps,
        completion_mask=torch.ones_like(student_logps),
        rewards=rewards,
        group_size=group_size,
        config=cfg,
        action_mask=action_mask,
    )

    telemetry.loss.backward()

    assert student_logps.grad is not None
    obs_grad = student_logps.grad[:, :obs_len]
    act_grad = student_logps.grad[:, obs_len:]

    # Zero observation leakage
    assert torch.all(obs_grad == 0.0)
    # Non-vanishing action gradient
    assert torch.norm(act_grad) > 0.001
