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
from typing import Any

from ask_the_database.evaluation import HarnessSummary, ZeroBudgetHarness


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
    headers = [
        "ID",
        "Expected Trap",
        "Status",
        "Attributed Trap",
        "Match?",
        "Latency (ms)",
    ]
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
        description="Run AskTheDatabase Zero-Budget Evaluation Harness (< 3.0s local smoke check).",
        allow_abbrev=False,
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
    group.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=str,
        help="Path to fine-tuned model checkpoint or LoRA adapter directory to evaluate live.",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        type=Path,
        help="Optional path to export evaluation summary JSON.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum interactive agent turns when evaluating a checkpoint (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without loading real model weights for fast testing.",
    )
    args = parser.parse_args()

    harness = ZeroBudgetHarness()
    checkpoint_gate: dict[str, Any] | None = None

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
    elif args.checkpoint:
        title = f"Checkpoint Evaluation ({Path(args.checkpoint).name})"
        cases_file = Path("data/fixtures/smoke_cases.jsonl")
        fixtures_sql = Path("data/fixtures/seed_fixtures.sql")

        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from generate_smoke_predictions import (
            generate_predictions,
            load_smoke_cases,
            setup_temporary_database,
        )

        cases = load_smoke_cases(cases_file)
        temp_db_path = setup_temporary_database(fixtures_sql)
        try:
            preds = generate_predictions(
                model_path=args.checkpoint,
                cases=cases,
                db_path=temp_db_path,
                max_turns=args.max_turns,
                device="auto",
                dry_run=args.dry_run,
            )
        finally:
            if temp_db_path.exists():
                temp_db_path.unlink()

        total_p = len(preds)
        inspected_count = sum(1 for p in preds if p.get("inspected_schema", False))
        ast_crash_count = sum(1 for p in preds if p.get("ast_crash", False))
        inspection_rate = (inspected_count / total_p * 100.0) if total_p else 0.0
        gate_passed = (inspection_rate >= 80.0) and (ast_crash_count == 0)

        checkpoint_gate = {
            "checkpoint": args.checkpoint,
            "total_cases": total_p,
            "schema_inspection_rate": round(inspection_rate, 2),
            "schema_inspection_threshold": 80.0,
            "schema_inspection_passed": inspection_rate >= 80.0,
            "ast_crash_count": ast_crash_count,
            "ast_crash_threshold": 0,
            "ast_crash_passed": ast_crash_count == 0,
            "gate_verdict": "GO" if gate_passed else "NO-GO",
            "ready_for_stage_4_rlvr": gate_passed,
        }

        cases_by_id = {c["id"]: c for c in cases}
        pairs = [
            (cases_by_id[p["id"]], p.get("sql") or p.get("query", ""))
            for p in preds
            if p["id"] in cases_by_id
        ]
        summary = harness.evaluate_batch(pairs)
    else:
        # Default: evaluate golden queries
        title = "Golden Sanity Verification"
        summary = harness.evaluate_golden()

    print_summary_report(summary, title)

    if checkpoint_gate is not None:
        print("\n" + "=" * 80)
        print(" CHECKPOINT SANITY GATE VERDICT (STEP 3 PRE-RLVR GATE)")
        print("=" * 80)
        insp_status = "PASS" if checkpoint_gate["schema_inspection_passed"] else "FAIL"
        ast_status = "PASS" if checkpoint_gate["ast_crash_passed"] else "FAIL"
        print(
            f"[{insp_status}] Schema Pre-Inspection Rate : {checkpoint_gate['schema_inspection_rate']:.1f}% "
            f"(Threshold: >= {checkpoint_gate['schema_inspection_threshold']}%)"
        )
        print(
            f"[{ast_status}] AST / Tool Syntax Crashes   : {checkpoint_gate['ast_crash_count']} "
            f"(Threshold: == {checkpoint_gate['ast_crash_threshold']})"
        )
        print("-" * 80)
        if checkpoint_gate["ready_for_stage_4_rlvr"]:
            print(
                " VERDICT: GO -> Checkpoint satisfies pre-RLVR gate. Proceed to Stage 4 Multi-Turn RLVR."
            )
        else:
            print(
                " VERDICT: NO-GO -> Checkpoint fails pre-RLVR gate. Review SFT training/distillation."
            )
        print("=" * 80)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = summary.to_dict()
        if checkpoint_gate is not None:
            summary_payload["checkpoint_gate"] = checkpoint_gate
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nSaved summary to: {args.output_json}")

    # Return non-zero if golden evaluation fails or latency exceeds Gate 1 budget
    if (
        args.golden or not (args.flawed or args.predictions or args.checkpoint)
    ) and summary.pass_rate < 1.0:
        return 1
    if summary.total_latency_ms > 3000.0:
        print(
            f"\n[GATE 1 FAILED] Runtime {summary.total_latency_ms:.2f}ms exceeds 3000ms threshold!"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
