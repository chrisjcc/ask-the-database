"""Unit tests for scripts/compare_rlvr_efficiency.py."""

import json
from pathlib import Path

import pytest

from scripts.compare_rlvr_efficiency import (
    calculate_sample_efficiency_metrics,
    compare_runs,
    generate_markdown_report,
)


def test_calculate_sample_efficiency_metrics_empty() -> None:
    res = calculate_sample_efficiency_metrics([])
    assert res["total_steps"] == 0
    assert res["steps_to_milestone"] is None
    assert res["final_reward"] == 0.0
    assert res["area_under_curve"] == 0.0


def test_calculate_sample_efficiency_metrics_positive() -> None:
    rewards = [-0.5, -0.2, 0.1, 0.4, 0.6]
    res = calculate_sample_efficiency_metrics(rewards, milestone_threshold=0.0)
    assert res["total_steps"] == 5
    assert res["steps_to_milestone"] == 3  # index 2 -> step 3
    assert res["final_reward"] == 0.6
    assert res["area_under_curve"] == 0.4


def test_compare_runs_dry_run() -> None:
    res = compare_runs(
        run_a_dir=Path("models/rlvr/from_cond2"),
        run_b_dir=Path("models/rlvr/from_cond4"),
        dry_run=True,
    )
    assert "run_a" in res
    assert "run_b" in res
    assert "comparison" in res
    assert res["comparison"]["delta_test_suite_accuracy"] > 0
    assert res["comparison"]["delta_auc_cumulative_reward"] > 0

    markdown = generate_markdown_report(res)
    assert "RLVR Sample Efficiency & Asymptotic Correctness Report" in markdown
    assert "Convergence Speed & Sample Efficiency" in markdown
    assert "Asymptotic Correctness" in markdown


def test_compare_runs_real_files(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    (run_a / "training_state.json").write_text(
        json.dumps({"reward_history": [-0.8, -0.4, -0.1, 0.0]}), encoding="utf-8"
    )
    (run_b / "training_state.json").write_text(
        json.dumps({"reward_history": [-0.2, 0.1, 0.5, 0.8]}), encoding="utf-8"
    )

    eval_a = tmp_path / "eval_a.json"
    eval_b = tmp_path / "eval_b.json"
    eval_a.write_text(
        json.dumps(
            {
                "spider": {
                    "test_suite_accuracy_ts": 52.0,
                    "execution_accuracy_ex": 58.0,
                    "recovery_rate_mrr": 20.0,
                }
            }
        ),
        encoding="utf-8",
    )
    eval_b.write_text(
        json.dumps(
            {
                "spider": {
                    "test_suite_accuracy_ts": 60.0,
                    "execution_accuracy_ex": 66.0,
                    "recovery_rate_mrr": 40.0,
                }
            }
        ),
        encoding="utf-8",
    )

    res = compare_runs(
        run_a_dir=run_a,
        run_b_dir=run_b,
        eval_a_path=eval_a,
        eval_b_path=eval_b,
        dry_run=False,
    )
    assert res["run_a"]["efficiency"]["total_steps"] == 4
    assert res["run_b"]["efficiency"]["steps_to_milestone"] == 2
    assert res["comparison"]["delta_test_suite_accuracy"] == 8.0
