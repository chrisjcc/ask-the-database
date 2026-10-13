"""Unit tests for NL2SQLPipeline and comprehensive benchmark evaluation harness."""

from pathlib import Path

from ask_the_database.inference.pipeline import NL2SQLPipeline
from scripts.evaluate_benchmark import run_benchmark_evaluation


def test_nl2sql_pipeline_interactive_rollout(sample_sqlite_db: Path) -> None:
    """Test NL2SQLPipeline runs multi-turn interactive trajectory and captures query."""
    pipeline = NL2SQLPipeline(max_turns=4)

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
    traj = pipeline.run_interactive(
        question="Who is the oldest user?",
        db_path=str(sample_sqlite_db),
        db_id="company",
        gold_sql=gold_sql,
    )

    assert len(traj.steps) > 0
    assert traj.final_execution_success is True
    assert traj.final_query != ""
    assert traj.question == "Who is the oldest user?"

    dump = traj.to_dict()
    assert dump["db_id"] == "company"
    assert dump["final_execution_success"] is True


def test_run_benchmark_evaluation_with_custom_samples(
    sample_sqlite_db: Path, tmp_path: Path
) -> None:
    """Test evaluation harness calculates TS, EX, VA, MRR across custom samples."""
    out_dir = tmp_path / "eval_results"
    summary_file = out_dir / "evaluation_summary.json"

    gold_sql = "SELECT name, age FROM users WHERE age > 25 ORDER BY age ASC;"
    custom_samples = [
        {
            "question": "Which users are older than 25?",
            "gold_sql": gold_sql,
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
        }
    ]

    cfg = {
        "evaluation": {
            "model_path": None,
            "max_turns": 3,
            "benchmarks": ["spider", "spider_robustness_test"],
            "custom_samples": {
                "spider": custom_samples,
                "spider_robustness_test": custom_samples,
            },
            "output_dir": str(out_dir),
            "summary_file": str(summary_file),
        }
    }

    leaderboard = run_benchmark_evaluation(cfg)

    assert "spider" in leaderboard
    assert "spider_robustness_test" in leaderboard
    assert leaderboard["spider"]["total_samples"] == 1
    assert leaderboard["spider"]["valid_execution_rate_va"] == 100.0
    assert leaderboard["spider"]["execution_accuracy_ex"] == 100.0
    assert leaderboard["spider"]["test_suite_accuracy_ts"] == 100.0

    assert summary_file.exists()
