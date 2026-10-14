"""Unit tests for MVE Benchmarking Evaluation & Decision Gate Adjudication (Sub-Issue #63)."""

from pathlib import Path

from ask_the_database.distillation.mve_evaluator import (
    ConditionMetrics,
    GateEvaluation,
    adjudicate_decision_gates,
    load_mve_config,
)
from scripts.run_mve_evaluation import get_empirical_mve_metrics, run_mve_adjudication


def test_condition_metrics_dataclass() -> None:
    """Verify ConditionMetrics serialization and formatting."""
    cm = ConditionMetrics(
        condition_id=4,
        name="Harness-Zero SFT",
        split="spider_dev",
        test_suite_correctness=58.42,
        execution_accuracy=64.19,
        valid_sql=97.05,
        error_recovery_rate=38.41,
        average_turns=2.85,
        average_latency_ms=350.2,
    )
    d = cm.to_dict()
    assert d["condition_id"] == 4
    assert d["test_suite_correctness"] == 58.42
    assert d["execution_accuracy"] == 64.19
    assert d["error_recovery_rate"] == 38.41


def test_gate_evaluation_dataclass() -> None:
    """Verify GateEvaluation properties."""
    ge = GateEvaluation(
        gate_name="Primary Gate",
        status="PASSED",
        threshold=">= 5.0 pp",
        achieved="+6.8 pp",
        margin=1.8,
        description="Generalization check",
    )
    d = ge.to_dict()
    assert d["status"] == "PASSED"
    assert d["margin"] == 1.8


def test_adjudicate_decision_gates_passed() -> None:
    """Verify standard empirical metrics produce a GO verdict."""
    metrics = get_empirical_mve_metrics()
    report = adjudicate_decision_gates(metrics)

    assert report.verdict == "GO"
    assert len(report.gates) == 3
    assert all(g.status == "PASSED" for g in report.gates)
    assert report.metadata["delta_ts_vs_cond2"] == 6.8
    assert report.metadata["delta_ts_vs_cond3"] == 4.5
    assert report.metadata["rel_recovery_gain"] > 0.60

    markdown = report.to_markdown()
    assert "# Minimum Viable Experiment (MVE): Harness-Zero Benchmarking Report" in markdown
    assert "**Pre-Experimental Verdict:** **`GO`**" in markdown
    assert "Cond 4" in markdown


def test_adjudicate_decision_gates_primary_failed() -> None:
    """Verify that failing the primary generalization threshold triggers NO-GO."""
    # Condition 4 TS is only +2.0 pp over Condition 2 (needs >= 5.0 pp)
    mock_metrics = [
        ConditionMetrics(1, "Base", "spider_dev", 38.0, 44.0, 90.0, 14.0, 2.4, 300.0),
        ConditionMetrics(2, "SFT", "spider_dev", 51.0, 57.0, 94.0, 22.0, 2.6, 310.0),
        ConditionMetrics(3, "Filtered", "spider_dev", 52.0, 58.0, 95.0, 25.0, 2.7, 320.0),
        ConditionMetrics(4, "Harness-Zero", "spider_dev", 53.0, 59.0, 96.0, 38.0, 2.8, 330.0),
    ]

    report = adjudicate_decision_gates(mock_metrics)
    assert report.verdict == "NO-GO"
    gate1 = [g for g in report.gates if "Primary Gate" in g.gate_name][0]
    assert gate1.status == "FAILED"


def test_adjudicate_decision_gates_behavioral_failed() -> None:
    """Verify that failing behavioral error recovery triggers NO-GO."""
    # Condition 4 R_rec is identical to Condition 2 (0% relative gain)
    mock_metrics = [
        ConditionMetrics(1, "Base", "spider_dev", 38.0, 44.0, 90.0, 14.0, 2.4, 300.0),
        ConditionMetrics(2, "SFT", "spider_dev", 50.0, 56.0, 94.0, 25.0, 2.6, 310.0),
        ConditionMetrics(3, "Filtered", "spider_dev", 51.0, 57.0, 95.0, 26.0, 2.7, 320.0),
        ConditionMetrics(4, "Harness-Zero", "spider_dev", 58.0, 64.0, 97.0, 25.5, 2.8, 330.0),
    ]

    report = adjudicate_decision_gates(mock_metrics)
    assert report.verdict == "NO-GO"
    gate2 = [g for g in report.gates if "Behavioral Gate" in g.gate_name][0]
    assert gate2.status == "FAILED"


def test_adjudicate_decision_gates_efficiency_failed() -> None:
    """Verify that excessive turn inflation triggers NO-GO."""
    # Condition 4 average turns = 4.5 vs Condition 1 = 2.4 (delta = 2.1 > 1.5 max)
    mock_metrics = [
        ConditionMetrics(1, "Base", "spider_dev", 38.0, 44.0, 90.0, 14.0, 2.4, 300.0),
        ConditionMetrics(2, "SFT", "spider_dev", 50.0, 56.0, 94.0, 20.0, 2.6, 310.0),
        ConditionMetrics(3, "Filtered", "spider_dev", 52.0, 58.0, 95.0, 22.0, 2.7, 320.0),
        ConditionMetrics(4, "Harness-Zero", "spider_dev", 58.0, 64.0, 97.0, 35.0, 4.5, 350.0),
    ]

    report = adjudicate_decision_gates(mock_metrics)
    assert report.verdict == "NO-GO"
    gate3 = [g for g in report.gates if "Efficiency Gate" in g.gate_name][0]
    assert gate3.status == "FAILED"


def test_stop_condition_filtered_matches_harness_zero() -> None:
    """Stop condition: If Condition 3 (Filtered) matches Condition 4, verdict is NO-GO."""
    mock_metrics = [
        ConditionMetrics(1, "Base", "spider_dev", 38.0, 44.0, 90.0, 14.0, 2.4, 300.0),
        ConditionMetrics(2, "SFT", "spider_dev", 50.0, 56.0, 94.0, 20.0, 2.6, 310.0),
        ConditionMetrics(3, "Filtered", "spider_dev", 58.0, 64.0, 97.0, 32.0, 2.8, 330.0),
        ConditionMetrics(4, "Harness-Zero", "spider_dev", 58.0, 64.0, 97.0, 35.0, 2.8, 330.0),
    ]

    report = adjudicate_decision_gates(mock_metrics)
    assert report.verdict == "NO-GO"
    assert "Gains stem from rejection filtering alone" in report.summary


def test_load_mve_config() -> None:
    """Verify loading MVE conditions YAML."""
    cfg = load_mve_config()
    assert "conditions" in cfg
    assert "decision_gates" in cfg
    assert cfg.decision_gates.primary_ts_gain_vs_cond2 == 5.0


def test_run_mve_adjudication_dry_run(tmp_path: Path) -> None:
    """Verify end-to-end dry-run execution and file exports."""
    rep_out = tmp_path / "report.md"
    sum_out = tmp_path / "summary.json"

    report = run_mve_adjudication(
        report_output_path=rep_out,
        summary_output_path=sum_out,
        dry_run=True,
    )

    assert report.verdict == "GO"
    assert rep_out.exists()
    assert sum_out.exists()
    assert "Minimum Viable Experiment" in rep_out.read_text()
