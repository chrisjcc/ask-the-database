"""CLI entrypoint for LLM-as-a-Verifier Minimum Viable Experiment (MVE) evaluation.

Evaluates 4 comparative configurations:
1. Baseline: Conventional SFT (greedy N=1)
2. Harness-Zero: Procedural exploration internalized (greedy N=1)
3. LLM-as-a-Verifier: Baseline model with Best-of-5 tournament selection (N=5)
4. Harness-Zero + Verifier: Compound model with Best-of-5 tournament selection (N=5)

Adjudicates the Pre-Experimental Decision Gates and exports:
- Markdown research report: `docs/research/llm-as-a-verifier/mve_results.md`
- JSON summary: `results/verifier_mve_summary.json`

Usage:
    python scripts/run_verifier_mve.py --dry-run
    python scripts/run_verifier_mve.py --config configs/verification/mve_verifier_conditions.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ask_the_database.verification.mve_adjudicator import (
    VerifierMVEReport,
    adjudicate_verifier_gates,
    get_empirical_verifier_metrics,
    load_verifier_mve_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_verifier_mve_evaluation(
    config_path: str | Path | None = None,
    report_output_path: str | Path = "docs/research/llm-as-a-verifier/mve_results.md",
    summary_output_path: str | Path = "results/verifier_mve_summary.json",
    dry_run: bool = False,
) -> VerifierMVEReport:
    """Orchestrate Verifier MVE evaluation, adjudicate decision gates, and output reports."""
    logger.info("Loading Verifier MVE configuration from: %s", config_path or "default")
    cfg = load_verifier_mve_config(config_path)

    metrics = get_empirical_verifier_metrics()
    logger.info("Evaluating %d condition metric records across benchmark splits.", len(metrics))

    # Adjudicate pre-experimental decision gates
    report = adjudicate_verifier_gates(metrics, cfg)
    logger.info("Adjudication complete. Decision Gate Verdict: %s", report.verdict)
    logger.info("  - Standalone Verifier Verdict: %s", report.standalone_verdict)
    logger.info("  - Compound Architecture Verdict: %s", report.compound_verdict)

    # Export markdown research report
    rep_path = Path(report_output_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(report.to_markdown(), encoding="utf-8")
    logger.info("Published Verifier MVE Benchmarking Report to: %s", rep_path)

    # Export JSON summary
    sum_path = Path(summary_output_path)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Saved Verifier MVE Evaluation Summary JSON to: %s", sum_path)

    return report


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Verifier MVE evaluation and generate decision gate report."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/verification/mve_verifier_conditions.yaml",
        help="Path to Verifier MVE conditions configuration YAML.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="docs/research/llm-as-a-verifier/mve_results.md",
        help="Destination path for Markdown research report.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/verifier_mve_summary.json",
        help="Destination path for JSON evaluation summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast dry-run evaluation check.",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args(sys.argv[1:])
    run_verifier_mve_evaluation(
        config_path=args.config,
        report_output_path=args.report_output,
        summary_output_path=args.summary_output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
