"""Unit tests for report generation and Weave logging."""

import json
from pathlib import Path

from scripts.generate_report import (
    generate_ablation_markdown,
    generate_reports,
    log_trajectories_to_weave,
)


def test_generate_ablation_markdown_formatting() -> None:
    """Test markdown generation includes tables, metrics, and sections."""
    eval_data = {
        "spider": {
            "total_samples": 100,
            "test_suite_accuracy_ts": 85.0,
            "execution_accuracy_ex": 89.0,
            "valid_execution_rate_va": 99.0,
            "recovery_rate_mrr": 45.0,
        }
    }
    md = generate_ablation_markdown(eval_data)

    assert "# AskTheDatabase: Systematic Ablation & Benchmark Report" in md
    assert "Multi-Turn SkyRL-Agent (Ours)" in md
    assert "**spider**" in md
    assert "85.0%" in md
    assert "Error Taxonomy" in md


def test_generate_reports_file_output(tmp_path: Path) -> None:
    """Test generating report files from evaluation JSON."""
    summary_file = tmp_path / "evaluation_summary.json"
    report_file = tmp_path / "ablation_report.md"
    leaderboard_file = tmp_path / "evaluation_leaderboard.md"

    summary_file.write_text(
        json.dumps(
            {
                "spider_dk": {
                    "total_samples": 50,
                    "test_suite_accuracy_ts": 80.0,
                    "execution_accuracy_ex": 84.0,
                    "valid_execution_rate_va": 98.0,
                    "recovery_rate_mrr": 40.0,
                }
            }
        ),
        encoding="utf-8",
    )

    r_path, l_path = generate_reports(
        summary_path=summary_file,
        output_report_path=report_file,
        output_leaderboard_path=leaderboard_file,
    )

    assert r_path.exists()
    assert l_path.exists()
    assert "spider_dk" in r_path.read_text(encoding="utf-8")


def test_log_trajectories_to_weave_graceful_fallback() -> None:
    """Test weave trajectory logging returns boolean without unhandled exception."""
    res = log_trajectories_to_weave([{"turn": 1, "action": "test"}])
    assert isinstance(res, bool)
