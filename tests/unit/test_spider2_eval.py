"""Unit tests for Spider 2.0-SQLite benchmark preparation and evaluation."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ask_the_database.data.spider2 import prepare_spider2_sqlite_tasks, resolve_db_path
from ask_the_database.environment.base import ExecutionResult
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator


@pytest.fixture
def mock_spider2_raw(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create synthetic raw Spider 2.0-Lite files for testing."""
    raw_file = tmp_path / "spider2-lite.jsonl"
    db_dir = tmp_path / "databases"
    gold_dir = tmp_path / "gold"
    out_file = tmp_path / "processed" / "spider2_sqlite.jsonl"

    db_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)

    # 1. Create mock sqlite DB
    db_path = db_dir / "E_commerce.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE orders (order_id INT, customer_id INT, amount REAL);")
    conn.execute("INSERT INTO orders VALUES (1, 101, 150.5), (2, 102, 200.0), (3, 101, 50.0);")
    conn.close()

    # 2. Create mock gold CSV
    gold_df = pd.DataFrame({"customer_id": [101, 102], "total_amount": [200.5, 200.0]})
    gold_df.to_csv(gold_dir / "local001.csv", index=False)

    # 3. Create mock spider2-lite.jsonl with a mix of BQ, SF, and local SQLite tasks
    raw_lines = [
        {"instance_id": "bq001", "db": "ga4", "question": "BQ question 1"},
        {
            "instance_id": "local001",
            "db": "E_commerce",
            "question": "What is the total spend per customer?",
            "external_knowledge": "rfm.md",
        },
        {"instance_id": "sf001", "db": "PATENTS", "question": "Snowflake question 1"},
        {
            "instance_id": "local002",
            "db": "E_commerce",
            "question": "What is the average order amount?",
            "external_knowledge": None,
        },
    ]

    with open(raw_file, "w", encoding="utf-8") as f:
        for line in raw_lines:
            f.write(json.dumps(line) + "\n")

    return raw_file, db_dir, gold_dir, out_file


def test_prepare_spider2_sqlite_tasks(mock_spider2_raw: tuple[Path, Path, Path, Path]) -> None:
    """Verify that only local SQLite tasks are filtered and correctly structured."""
    raw_file, db_dir, gold_dir, out_file = mock_spider2_raw

    tasks = prepare_spider2_sqlite_tasks(
        source_file=raw_file,
        db_dir=db_dir,
        gold_dir=gold_dir,
        output_file=out_file,
        allow_download=False,
    )

    # Out of 4 raw tasks, exactly 2 are local SQLite tasks
    assert len(tasks) == 2
    assert tasks[0]["instance_id"] == "local001"
    assert "rfm.md" in tasks[0]["question"]
    assert tasks[0]["db_id"] == "E_commerce"
    assert Path(tasks[0]["db_path"]).exists()
    assert tasks[0]["gold_csv_path"] is not None
    assert Path(tasks[0]["gold_csv_path"]).exists()

    assert tasks[1]["instance_id"] == "local002"
    assert tasks[1]["gold_csv_path"] is None  # gold CSV for local002 was not created

    # Verify JSONL output file exists and has 2 lines
    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2


def test_resolve_db_path(tmp_path: Path) -> None:
    """Verify database path resolution across flat and nested structures."""
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()

    # Flat database
    flat_db = db_dir / "Analytics.sqlite"
    flat_db.touch()
    assert resolve_db_path("Analytics", db_dir) == flat_db

    # Nested database
    nested_dir = db_dir / "Enterprise"
    nested_dir.mkdir()
    nested_db = nested_dir / "Enterprise.sqlite"
    nested_db.touch()
    assert resolve_db_path("Enterprise", db_dir) == nested_db


def test_gold_csv_execution_equivalence(tmp_path: Path) -> None:
    """Verify that generated execution results match gold CSV reference tables."""
    evaluator = ExecutionEquivalenceEvaluator()

    # Create mock gold CSV
    gold_csv = tmp_path / "ref.csv"
    gold_df = pd.DataFrame({"id": [1, 2], "val": [10.5, 20.0]})
    gold_df.to_csv(gold_csv, index=False)

    # 1. Matching candidate result
    match_result = ExecutionResult(
        success=True,
        data=[(1, 10.5), (2, 20.0)],
        columns=["id", "val"],
        row_count=2,
        execution_time_ms=1.2,
        error_message=None,
        error_type=None,
    )

    loaded_df = pd.read_csv(gold_csv)
    gold_tuples = [tuple(r) for r in loaded_df.to_numpy()]
    gold_res = ExecutionResult(
        success=True,
        data=gold_tuples,
        columns=list(loaded_df.columns),
        row_count=len(gold_tuples),
        execution_time_ms=0.0,
        error_message=None,
        error_type=None,
    )

    cmp_match = evaluator.compare(match_result, gold_res)
    assert cmp_match.is_equivalent is True

    # 2. Non-matching candidate result
    mismatch_result = ExecutionResult(
        success=True,
        data=[(1, 99.9), (2, 20.0)],
        columns=["id", "val"],
        row_count=2,
        execution_time_ms=1.1,
        error_message=None,
        error_type=None,
    )
    cmp_mismatch = evaluator.compare(mismatch_result, gold_res)
    assert cmp_mismatch.is_equivalent is False


def test_resolve_gold_paths_and_local_map(tmp_path: Path) -> None:
    """Verify resolution of multi-candidate gold CSVs and local-map.jsonl."""
    from ask_the_database.data.spider2.preprocessor import (
        Spider2SQLitePreprocessor,
        resolve_gold_paths,
    )

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "local002_a.csv").touch()
    (gold_dir / "local002_b.csv").touch()
    (gold_dir / "local010.csv").touch()

    # Multi-candidate resolution
    paths_002 = resolve_gold_paths("local002", gold_dir)
    assert len(paths_002) == 2
    assert any("local002_a.csv" in str(p) for p in paths_002)
    assert any("local002_b.csv" in str(p) for p in paths_002)

    # Single-candidate resolution
    paths_010 = resolve_gold_paths("local010", gold_dir)
    assert len(paths_010) == 1
    assert "local010.csv" in str(paths_010[0])

    # local-map.jsonl support
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    (db_dir / "local-map.jsonl").write_text(
        json.dumps({"local002": "E_commerce"}), encoding="utf-8"
    )
    (db_dir / "E_commerce.sqlite").touch()

    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps({"instance_id": "local002", "db": "", "question": "What is total revenue?"})
        + "\n",
        encoding="utf-8",
    )

    preprocessor = Spider2SQLitePreprocessor(db_dir=db_dir, gold_dir=gold_dir)
    tasks = preprocessor.extract_tasks(tasks_file, allow_download=False)
    assert len(tasks) == 1
    assert tasks[0]["db_id"] == "E_commerce"
    assert "E_commerce.sqlite" in tasks[0]["db_path"]
    assert len(tasks[0]["gold_csv_paths"]) == 2
