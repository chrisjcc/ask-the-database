"""CLI entrypoint for running the Zero-Budget Evaluation Harness (make smoke-eval).

Evaluates the 35 curated smoke cases in-process in < 3.0s, outputting an ANSI
diagnostic summary, per-trap failure attribution, and actionable remediation advice.

Usage:
    python scripts/run_smoke_eval.py --golden
    python scripts/run_smoke_eval.py --flawed
    python scripts/run_smoke_eval.py --predictions path/to/predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from ask_the_database.evaluation import (
    HarnessSummary,
    ZeroBudgetHarness,
)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format tabular data into an aligned ASCII text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)) for row in rows]
    return f"{header_str}\n{sep_str}\n" + "\n".join(row_strs)


def print_summary_report(summary: HarnessSummary, title: str) -> None:
    """Print structured diagnostic report to stdout."""
    print("=" * 80)
    print(f" ASKTHEDATABASE ZERO-BUDGET EVALUATION HARNESS: {title.upper()}")
    print("=" * 80)
    print(f"Total Cases Evaluated : {summary.total_cases}")
    print(f"Passed Cases          : {summary.passed_cases}")
    print(f"Failed Cases          : {summary.failed_cases}")
    print(f"Overall Pass Rate     : {summary.pass_rate:.1%}")
    print(f"Base Pass Rate        : {summary.base_pass_rate:.1%}")
    print(f"Mutant Pass Rate      : {summary.mutant_pass_rate:.1%}")
    print(f"Mutant Bugs Caught    : {summary.mutation_detected_count}")
    print(f"Diagnostic Attribution: {summary.attribution_accuracy:.1%}")
    print(
        f"Total Execution Time  : {summary.total_latency_ms:.2f} ms ({summary.total_latency_ms / 1000.0:.3f} s)"
    )
    print(f"Average Latency / Case: {summary.average_latency_ms:.2f} ms")
    print("-" * 80)

    # Per-Case Verdicts
    print("\n[ CASE EVALUATION BREAKDOWN ]")
    headers = ["ID", "Expected Trap", "Status", "Attributed Trap", "Match?", "Latency (ms)"]
    rows = []
    for v in summary.case_verdicts:
        match_str = "YES" if v.attribution_match else ("N/A" if v.passed else "NO")
        rows.append(
            [
                v.case_id,
                v.expected_trap,
                v.status,
                v.detected_trap or "-",
                match_str,
                f"{v.latency_ms:.1f}",
            ]
        )
    print(format_table(headers, rows))

    # Error-Trap Concentration Table
    print("\n[ ERROR-TRAP CONCENTRATION BREAKDOWN ]")
    trap_headers = ["Trap Category", "Miss Count", "Share of Failures (%)"]
    trap_rows = []
    for trap, count in summary.misses_by_trap.items():
        share = (count / summary.failed_cases * 100.0) if summary.failed_cases > 0 else 0.0
        trap_rows.append([trap, str(count), f"{share:.1f}%"])
    print(format_table(trap_headers, trap_rows))

    # Actionable Decision Table Recommendations
    if summary.remediation_recommendations:
        print("\n[ ACTIONABLE REMEDIATION DECISION TABLE ]")
        rec_headers = ["Trap", "Severity", "Pipeline Target", "Prescriptive Action"]
        rec_rows = [
            [
                r["trap"],
                r["severity"],
                r["pipeline_target"],
                r["prescriptive_action"],
            ]
            for r in summary.remediation_recommendations
        ]
        print(format_table(rec_headers, rec_rows))
    print("=" * 80)


def main() -> int:
    """Main CLI driver for running the zero-budget smoke evaluation."""
    parser = argparse.ArgumentParser(
        description="Run AskTheDatabase Zero-Budget Evaluation Harness (< 3.0s local smoke check)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--golden",
        action="store_true",
        help="Evaluate reference golden queries (verification sanity check).",
    )
    group.add_argument(
        "--flawed",
        action="store_true",
        help="Evaluate injected flawed queries (trap diagnosis & decision table check).",
    )
    group.add_argument(
        "--predictions",
        type=Path,
        help="Path to JSONL file with {'id': str, 'sql': str} predictions to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to export evaluation summary JSON.",
    )
    args = parser.parse_args()

    harness = ZeroBudgetHarness()

    if args.flawed:
        title = "Diagnostic Flaw Simulation"
        summary = harness.evaluate_flawed()
    elif args.predictions:
        title = f"Candidate Evaluation ({args.predictions.name})"
        cases = harness.load_cases()
        cases_by_id = {c["id"]: c for c in cases}

        with open(args.predictions, encoding="utf-8") as f:
            preds = [json.loads(line) for line in f if line.strip()]

        pairs = []
        for p in preds:
            cid = p["id"]
            if cid in cases_by_id:
                pairs.append((cases_by_id[cid], p.get("sql") or p.get("query", "")))

        summary = harness.evaluate_batch(pairs)
    else:
        # Default: evaluate golden queries
        title = "Golden Sanity Verification"
        summary = harness.evaluate_golden()

    print_summary_report(summary, title)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\nSaved summary to: {args.output_json}")

    # Return non-zero if golden evaluation fails or latency exceeds Gate 1 budget
    if (args.golden or not (args.flawed or args.predictions)) and summary.pass_rate < 1.0:
        return 1
    if summary.total_latency_ms > 3000.0:
        print(
            f"\n[GATE 1 FAILED] Runtime {summary.total_latency_ms:.2f}ms exceeds 3000ms threshold!"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
