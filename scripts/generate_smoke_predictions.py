"""Generate model predictions for the 35 zero-budget smoke cases.

Loads a fine-tuned LoRA or base model checkpoint, populates a temporary
SQLite database from seed_fixtures.sql, executes interactive or single-turn
rollouts for each smoke case, and saves predictions to a JSONL file.

Usage:
    # On GPU server (DESY):
    python scripts/generate_smoke_predictions.py \\
        --model-path models/mve/sft_baseline \\
        --output results/baseline_smoke_preds.jsonl

    python scripts/generate_smoke_predictions.py \\
        --model-path models/sft/harness_zero \\
        --output results/harness_zero_smoke_preds.jsonl

    # Dry-run smoke test (CPU / offline):
    python scripts/generate_smoke_predictions.py --dry-run --output results/sample_preds.jsonl
"""

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ask_the_database.inference.pipeline import NL2SQLPipeline


def setup_temporary_database(fixtures_sql_path: Path) -> Path:
    """Create a temporary SQLite database initialized with seed fixtures."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
        temp_db_path = Path(temp_db.name)

    conn = sqlite3.connect(temp_db_path)
    with open(fixtures_sql_path, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return temp_db_path


def load_smoke_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load smoke test cases from JSONL."""
    cases: list[dict[str, Any]] = []
    with open(cases_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def generate_predictions(
    model_path: str | None,
    cases: list[dict[str, Any]],
    db_path: Path,
    max_turns: int = 1,
    device: str = "auto",
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Generate SQL predictions for each case using NL2SQLPipeline."""
    if dry_run:
        print("[DRY-RUN] Generating simulated candidate predictions...")
        predictions = []
        for case in cases:
            # Simulate a realistic prediction (either golden or expected flaw based on ID)
            sql = case.get("expected_flaw") or case.get("query", "SELECT 1;")
            predictions.append(
                {
                    "id": case["id"],
                    "sql": sql,
                    "turns": 3,
                    "inspected_schema": True,
                    "ast_crash": False,
                }
            )
        return predictions

    print(f"Initializing NL2SQLPipeline with model: {model_path} (max_turns={max_turns})...")
    pipeline = NL2SQLPipeline(model_path=model_path, device=device, max_turns=max_turns)

    predictions = []
    for case in tqdm(cases, desc="Generating SQL"):
        cid = case["id"]
        question = case["question"]
        gold_sql = case.get("query", "SELECT 1;")

        trajectory = pipeline.run_interactive(
            question=question,
            db_path=str(db_path),
            db_id="smoke_fixture",
            max_turns=max_turns,
            gold_sql=gold_sql,
        )
        final_sql = trajectory.final_query.strip()
        inspected = any(s.action == "inspect_schema" for s in trajectory.steps)
        ast_crash = any(
            s.action == "error" or "syntax error" in s.output.lower() for s in trajectory.steps
        )
        predictions.append(
            {
                "id": cid,
                "sql": final_sql,
                "turns": len(trajectory.steps),
                "inspected_schema": inspected,
                "ast_crash": ast_crash,
            }
        )

    return predictions


def main() -> int:
    """CLI driver for smoke case prediction generation."""
    parser = argparse.ArgumentParser(
        description="Generate candidate SQL predictions for Zero-Budget Harness smoke cases."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights or LoRA adapter directory (e.g. models/mve/sft_baseline).",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=Path("data/fixtures/smoke_cases.jsonl"),
        help="Path to smoke cases JSONL.",
    )
    parser.add_argument(
        "--fixtures-sql",
        type=Path,
        default=Path("data/fixtures/seed_fixtures.sql"),
        help="Path to seed fixtures SQL schema file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write predictions JSONL output.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum interactive agent turns (default: 5 for multi-turn rollout).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: 'auto', 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without loading real model weights for fast testing.",
    )
    args = parser.parse_args()

    if not args.cases_file.exists():
        print(f"Error: Cases file not found at {args.cases_file}", file=sys.stderr)
        return 1

    if not args.fixtures_sql.exists():
        print(f"Error: Seed fixtures SQL not found at {args.fixtures_sql}", file=sys.stderr)
        return 1

    print(f"Setting up temporary database from {args.fixtures_sql}...")
    temp_db_path = setup_temporary_database(args.fixtures_sql)

    try:
        cases = load_smoke_cases(args.cases_file)
        print(f"Loaded {len(cases)} smoke cases.")

        predictions = generate_predictions(
            model_path=args.model_path,
            cases=cases,
            db_path=temp_db_path,
            max_turns=args.max_turns,
            device=args.device,
            dry_run=args.dry_run,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        print(f"\nSuccessfully wrote {len(predictions)} predictions to: {args.output}")
        print("\nNext step: evaluate these predictions with:")
        print(f"  python scripts/run_smoke_eval.py --predictions {args.output}")
    finally:
        if temp_db_path.exists():
            temp_db_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
