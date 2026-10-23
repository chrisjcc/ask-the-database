"""Unit tests for the Zero-Budget Evaluation Harness MVE benchmark and reporting."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.run_zero_budget_mve import (
    compute_spearman_rho,
    generate_markdown_report,
    rank_data,
    run_benchmark,
)


def test_spearman_rank_calculation() -> None:
    """Test Spearman rho and rank conversion helper functions."""
    # Identical ranking
    vals1 = [10.0, 20.0, 30.0, 40.0]
    vals2 = [100.0, 200.0, 300.0, 400.0]
    r1 = rank_data(vals1)
    r2 = rank_data(vals2)
    assert r1 == [1.0, 2.0, 3.0, 4.0]
    assert r2 == [1.0, 2.0, 3.0, 4.0]
    assert compute_spearman_rho(r1, r2) == 1.0

    # Completely inverted ranking
    vals_inv = [400.0, 300.0, 200.0, 100.0]
    r_inv = rank_data(vals_inv)
    assert compute_spearman_rho(r1, r_inv) == -1.0


def test_mve_benchmark_adjudication() -> None:
    """Test full MVE benchmark execution and verification of 4 Go/No-Go gates."""
    result: dict[str, Any] = run_benchmark()

    assert result["verdict"] == "GO"
    gates = result["gates"]

    assert gates["gate_1_latency"]["passed"] is True
    assert gates["gate_2_correlation"]["passed"] is True
    assert gates["gate_3_attribution"]["passed"] is True
    assert gates["gate_4_immunity"]["passed"] is True

    # Check that Gate 2 correlation is >= 0.88
    spearman_rho = gates["gate_2_correlation"]["spearman_rho"]
    assert spearman_rho >= 0.88


def test_markdown_report_generation() -> None:
    """Test that Markdown report contains all pre-registered sections and badges."""
    result = run_benchmark()
    report_md = generate_markdown_report(result)

    assert "# Empirical MVE Results: Zero-Budget Evaluation Harness" in report_md
    assert "**GO (ADOPT HARNESS)**" in report_md
    assert "Gate 1" in report_md
    assert "Gate 2" in report_md
    assert "Gate 3" in report_md
    assert "Gate 4" in report_md
    assert "Spearman Rank Correlation" in report_md
    assert "hallucinated_column" in report_md


def test_smoke_eval_cli_golden(tmp_path: Path) -> None:
    """Test CLI execution of scripts/run_smoke_eval.py --golden."""
    out_json = tmp_path / "summary.json"
    cmd = [
        sys.executable,
        "scripts/run_smoke_eval.py",
        "--golden",
        "--output-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "GOLDEN SANITY VERIFICATION" in res.stdout
    assert out_json.exists()


def test_smoke_eval_cli_flawed(tmp_path: Path) -> None:
    """Test CLI execution of scripts/run_smoke_eval.py --flawed."""
    out_json = tmp_path / "flawed_summary.json"
    cmd = [
        sys.executable,
        "scripts/run_smoke_eval.py",
        "--flawed",
        "--output-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "DIAGNOSTIC FLAW SIMULATION" in res.stdout
    assert out_json.exists()
