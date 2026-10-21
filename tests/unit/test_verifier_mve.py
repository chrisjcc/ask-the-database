"""Unit tests for Verifier MVE evaluation and decision gate adjudication (Sub-Issue #70)."""

import json
from pathlib import Path

from ask_the_database.verification.mve_adjudicator import (
    VerifierConditionMetrics,
    adjudicate_verifier_gates,
    get_empirical_verifier_metrics,
    load_verifier_mve_config,
)
from scripts.run_verifier_mve import run_verifier_mve_evaluation


def test_load_verifier_mve_config() -> None:
    """Verify loading MVE verifier YAML configuration."""
    cfg = load_verifier_mve_config()
    assert "conditions" in cfg
    assert len(cfg["conditions"]) == 4
    assert "decision_gates" in cfg
    assert "gate_1_absolute_gain" in cfg["decision_gates"]
    assert "gate_2_harness_zero_dominance" in cfg["decision_gates"]


def test_adjudicate_verifier_gates_empirical() -> None:
    """Verify empirical decision gate adjudication matches pre-registered criteria."""
    metrics = get_empirical_verifier_metrics()
    report = adjudicate_verifier_gates(metrics)

    # Standalone verifier fails Gate 1 (+4.6 pp < +5.0 pp) and Gate 2 (56.2% < 58.4% + 2.0%)
    assert report.standalone_verdict == "NO-GO"
    assert report.verdict == "NO-GO"

    # Compound architecture passes Gate 2b (+3.4 pp >= +3.0 pp)
    assert report.compound_verdict == "GO"

    gates_by_name = {g.gate_name: g for g in report.gates}

    # Gate 1
    g1 = next(g for name, g in gates_by_name.items() if "Gate 1" in name)
    assert g1.status == "FAILED"
    assert g1.margin == -0.4  # 4.6 - 5.0

    # Gate 2
    g2 = next(g for name, g in gates_by_name.items() if "Gate 2:" in name)
    assert g2.status == "FAILED"
    assert g2.margin == -2.2  # 56.2 - 58.4

    # Gate 2b
    g2b = next(g for name, g in gates_by_name.items() if "Gate 2b" in name)
    assert g2b.status == "PASSED"
    assert g2b.margin == 0.4  # 61.8 - 61.4

    # Gate 3
    g3 = next(g for name, g in gates_by_name.items() if "Gate 3" in name)
    assert g3.status == "PASSED"
    assert g3.margin == 80.0  # 1200 - 1120

    # Gate 4
    g4 = next(g for name, g in gates_by_name.items() if "Gate 4" in name)
    assert g4.status == "PASSED"
    assert g4.margin == 0.6  # 3.6 - 3.0


def test_adjudicate_verifier_gates_hypothetical_go() -> None:
    """Verify that a hypothetical high-performing verifier achieves GO verdict."""
    metrics = [
        VerifierConditionMetrics(1, "Baseline", "spider_dev", 50.0, 55.0, 90.0, 300.0, 1.0),
        VerifierConditionMetrics(2, "Harness-Zero", "spider_dev", 58.0, 63.0, 95.0, 350.0, 1.0),
        VerifierConditionMetrics(
            3, "Verifier", "spider_dev", 62.0, 67.0, 98.0, 1000.0, 5.0, majority_vote_agreement=85.0
        ),
        VerifierConditionMetrics(
            4, "Compound", "spider_dev", 65.0, 70.0, 99.0, 1100.0, 5.5, majority_vote_agreement=88.0
        ),
    ]
    report = adjudicate_verifier_gates(metrics)
    assert report.standalone_verdict == "GO"
    assert report.compound_verdict == "GO"
    assert report.verdict == "GO"


def test_verifier_mve_report_serialization() -> None:
    """Verify dictionary serialization and Markdown generation."""
    metrics = get_empirical_verifier_metrics()
    report = adjudicate_verifier_gates(metrics)

    d = report.to_dict()
    assert d["verdict"] == "NO-GO"
    assert len(d["conditions"]) == 8
    assert len(d["gates"]) == 5

    md = report.to_markdown()
    assert "# LLM-as-a-Verifier Minimum Viable Experiment" in md
    assert "Overall Gate Verdict: NO-GO" in md
    assert "Harness-Zero Dominance Finding" in md
    assert "Synergistic Compound Architecture" in md


def test_run_verifier_mve_pipeline_dry_run(tmp_path: Path) -> None:
    """Verify end-to-end execution of run_verifier_mve script."""
    rep_file = tmp_path / "mve_results.md"
    sum_file = tmp_path / "mve_summary.json"

    report = run_verifier_mve_evaluation(
        report_output_path=rep_file,
        summary_output_path=sum_file,
        dry_run=True,
    )

    assert report.verdict == "NO-GO"
    assert rep_file.is_file()
    assert sum_file.is_file()

    with open(sum_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["verdict"] == "NO-GO"
    assert data["standalone_verdict"] == "NO-GO"
    assert data["compound_verdict"] == "GO"
    assert len(data["conditions"]) == 8
