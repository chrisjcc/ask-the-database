"""Integration tests for 7-Condition Comparative PoC Execution and Hypothesis Adjudication (Phase 5).

Validates:
1. End-to-end execution of run_opdvr_comparative_benchmark across all 7 conditions (C0 to C6).
2. Adjudication of all four scientific hypotheses:
   - H1: Sample Efficiency (Delta R_TS at Step 100 >= +4.0 pp).
   - H2: Exploration Plateau Elimination (Extra Hard stalls < 5 steps).
   - H3: Syntactic Diversity (AST variety ratio >= 2.5x vs OPD).
   - H4: Privileged Distillation (7B student matches >= 98.0% of 14B baseline).
3. Report serialization and schema completeness for JSON and Markdown artifacts.
4. Single-condition evaluation mode.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_opdvr_comparative_benchmark import (
    CONDITIONS,
    adjudicate_scientific_hypotheses,
    run_opdvr_comparative_benchmark,
)


def test_comparative_benchmark_all_conditions(tmp_path: Path) -> None:
    """Verify full 7-condition execution, report generation, and hypothesis adjudication."""
    json_path = tmp_path / "opdvr_results.json"
    md_path = tmp_path / "results_report.md"

    results = run_opdvr_comparative_benchmark(
        condition="all",
        mock_eval=True,
        output_json=json_path,
        output_md=md_path,
    )

    assert results["status"] == "PASSED"
    assert results["num_conditions_evaluated"] == 7
    assert len(results["conditions"]) == 7

    # Check all conditions are present
    for cid in CONDITIONS:
        assert cid in results["conditions"]
        res = results["conditions"][cid]
        assert "test_suite_acc" in res
        assert "execution_acc" in res
        assert "extra_hard_stall_steps" in res
        assert "ast_diversity_entropy" in res

    # Verify Hypothesis Adjudication
    adj = results["hypothesis_adjudication"]
    assert adj["status"] == "PASSED"

    hypotheses = adj["hypotheses"]
    assert hypotheses["H1_sample_efficiency"]["passed"] is True
    assert hypotheses["H1_sample_efficiency"]["observed_delta_pp"] >= 4.0

    assert hypotheses["H2_exploration_stall"]["passed"] is True
    assert hypotheses["H2_exploration_stall"]["observed_stalls"] < 5

    assert hypotheses["H3_syntactic_diversity"]["passed"] is True
    assert hypotheses["H3_syntactic_diversity"]["observed_ratio"] >= 2.5

    assert hypotheses["H4_privileged_distillation"]["passed"] is True
    assert hypotheses["H4_privileged_distillation"]["observed_parity_pct"] >= 98.0

    # Verify JSON file
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["status"] == "PASSED"
    assert len(loaded_json["conditions"]) == 7

    # Verify Markdown report
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Empirical Benchmark Findings & 7-Condition Comparative Report" in md_text
    assert "Hypothesis Adjudication Status" in md_text
    assert "Full GRPD (Proposed)" in md_text


def test_comparative_benchmark_single_condition(tmp_path: Path) -> None:
    """Verify executing a single targeted condition evaluation."""
    json_path = tmp_path / "single_results.json"
    md_path = tmp_path / "single_report.md"

    results = run_opdvr_comparative_benchmark(
        condition="c4_grpd",
        mock_eval=True,
        output_json=json_path,
        output_md=md_path,
    )

    assert results["status"] == "PARTIAL"
    assert results["num_conditions_evaluated"] == 1
    assert "c4_grpd" in results["conditions"]
    c4 = results["conditions"]["c4_grpd"]
    assert c4["id"] == "C4"
    assert c4["test_suite_acc"] == 68.38
    assert json_path.exists()


def test_adjudicate_hypotheses_boundary_failure() -> None:
    """Verify hypothesis failure detection when thresholds are violated."""
    mock_data = {
        "c1_pure_rlvr": {"step_100_ts_acc": 59.0, "extra_hard_stall_steps": 27},
        "c2_vanilla_opd": {"ast_diversity_entropy": 3.0},
        "c4_grpd": {
            "step_100_ts_acc": 60.0,  # Delta is only +1.0 pp (below target +4.0 pp)
            "extra_hard_stall_steps": 10,  # 10 stalls (above target < 5)
            "ast_diversity_entropy": 3.1,  # Ratio 1.03x (below target >= 2.5x)
        },
        "c6_priv_grpd": {"test_suite_acc": 60.0},  # 60.0 / 71.18 = 84.3% (< 98%)
    }

    adj = adjudicate_scientific_hypotheses(mock_data)
    assert adj["status"] == "FAILED"
    assert adj["hypotheses"]["H1_sample_efficiency"]["passed"] is False
    assert adj["hypotheses"]["H2_exploration_stall"]["passed"] is False
    assert adj["hypotheses"]["H3_syntactic_diversity"]["passed"] is False
    assert adj["hypotheses"]["H4_privileged_distillation"]["passed"] is False
