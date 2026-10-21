"""Unit tests validating Zero-Budget Evaluation Harness fixtures and smoke cases.

Ensures schema integrity, balanced trap distribution, hermetic in-memory execution,
deterministic reference outputs, and mutation-hardened flaw detection.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fixtures"
SEED_SQL_PATH = FIXTURES_DIR / "seed_fixtures.sql"
SEED_MUTANT_SQL_PATH = FIXTURES_DIR / "seed_fixtures_mutant.sql"
SMOKE_CASES_PATH = FIXTURES_DIR / "smoke_cases.jsonl"

EXPECTED_TRAP_CATEGORIES = {
    "join_fanout",
    "dialect_drift",
    "hallucinated_column",
    "aggregation_without_grouping",
    "null_boundary",
}


@pytest.fixture
def base_db() -> sqlite3.Connection:
    """Provides a fresh in-memory SQLite database initialized with base seed fixtures."""
    conn = sqlite3.connect(":memory:")
    with open(SEED_SQL_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    return conn


@pytest.fixture
def mutant_db() -> sqlite3.Connection:
    """Provides a fresh in-memory SQLite database initialized with mutant seed fixtures."""
    conn = sqlite3.connect(":memory:")
    with open(SEED_MUTANT_SQL_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    return conn


@pytest.fixture
def smoke_cases() -> list[dict[str, Any]]:
    """Loads all smoke cases from JSONL."""
    assert SMOKE_CASES_PATH.exists(), f"Missing smoke cases file: {SMOKE_CASES_PATH}"
    cases: list[dict[str, Any]] = []
    with open(SMOKE_CASES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(item)
    return cases


def test_smoke_cases_schema_and_distribution(smoke_cases: list[dict[str, Any]]) -> None:
    """Test that smoke cases meet schema requirements and balanced 7-per-trap distribution."""
    assert len(smoke_cases) == 35, f"Expected 35 smoke cases, found {len(smoke_cases)}"

    category_counts: dict[str, int] = {}
    seen_ids: set[str] = set()

    required_fields = {
        "id",
        "question",
        "query",
        "trap_category",
        "trap_description",
        "expected_flaw",
        "difficulty",
        "tables",
    }

    for idx, case in enumerate(smoke_cases):
        # Check required fields
        for field in required_fields:
            assert field in case, f"Case index {idx} missing required field '{field}'"

        case_id = case["id"]
        assert case_id not in seen_ids, f"Duplicate case ID: {case_id}"
        seen_ids.add(case_id)

        # Value validations
        assert isinstance(case["question"], str) and len(case["question"].strip()) > 0
        assert isinstance(case["query"], str) and len(case["query"].strip()) > 0
        assert isinstance(case["expected_flaw"], str) and len(case["expected_flaw"].strip()) > 0
        assert case["trap_category"] in EXPECTED_TRAP_CATEGORIES, (
            f"Invalid trap_category: {case['trap_category']}"
        )
        assert isinstance(case["tables"], list) and len(case["tables"]) > 0

        cat = case["trap_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Verify each trap category has exactly 7 cases
    assert set(category_counts.keys()) == EXPECTED_TRAP_CATEGORIES
    for cat, count in category_counts.items():
        assert count == 7, f"Trap category '{cat}' must have exactly 7 cases, found {count}"


def test_seed_fixtures_loading_speed() -> None:
    """Test that seed fixtures load in < 100ms in memory (Gate 1 prerequisite)."""
    assert SEED_SQL_PATH.exists(), f"Missing {SEED_SQL_PATH}"
    assert SEED_MUTANT_SQL_PATH.exists(), f"Missing {SEED_MUTANT_SQL_PATH}"

    # Base fixtures
    t0 = time.perf_counter()
    conn_base = sqlite3.connect(":memory:")
    with open(SEED_SQL_PATH, encoding="utf-8") as f:
        conn_base.executescript(f.read())
    elapsed_base = time.perf_counter() - t0
    conn_base.close()

    assert elapsed_base < 0.1, f"Base seed fixture took {elapsed_base:.4f}s to load (> 100ms)"

    # Mutant fixtures
    t1 = time.perf_counter()
    conn_mut = sqlite3.connect(":memory:")
    with open(SEED_MUTANT_SQL_PATH, encoding="utf-8") as f:
        conn_mut.executescript(f.read())
    elapsed_mut = time.perf_counter() - t1
    conn_mut.close()

    assert elapsed_mut < 0.1, f"Mutant seed fixture took {elapsed_mut:.4f}s to load (> 100ms)"


def test_reference_queries_execution_and_determinism(
    base_db: sqlite3.Connection,
    mutant_db: sqlite3.Connection,
    smoke_cases: list[dict[str, Any]],
) -> None:
    """Test that all 35 reference queries execute cleanly and produce deterministic non-empty results."""
    for case in smoke_cases:
        query = case["query"]
        case_id = case["id"]

        # Run against base
        cur_base = base_db.cursor()
        cur_base.execute(query)
        res_base_1 = cur_base.fetchall()
        assert len(res_base_1) > 0, f"Query for {case_id} produced empty result on base DB"

        # Determinism check
        cur_base.execute(query)
        res_base_2 = cur_base.fetchall()
        assert res_base_1 == res_base_2, f"Query for {case_id} is non-deterministic on base DB"

        # Run against mutant
        cur_mut = mutant_db.cursor()
        cur_mut.execute(query)
        res_mut = cur_mut.fetchall()
        assert len(res_mut) > 0, f"Query for {case_id} produced empty result on mutant DB"


def test_flawed_queries_catch_rate(
    base_db: sqlite3.Connection,
    mutant_db: sqlite3.Connection,
    smoke_cases: list[dict[str, Any]],
) -> None:
    """Test that 100% of injected flawed queries fail or produce discrepancies (Gate 4)."""
    detected_count = 0

    for case in smoke_cases:
        case_id = case["id"]
        ref_query = case["query"]
        flawed_query = case["expected_flaw"]

        cur_base = base_db.cursor()
        cur_base.execute(ref_query)
        ref_base_res = cur_base.fetchall()

        cur_mut = mutant_db.cursor()
        cur_mut.execute(ref_query)
        ref_mut_res = cur_mut.fetchall()

        is_caught = False

        # 1. Test execution on base DB
        try:
            cur_base.execute(flawed_query)
            flaw_base_res = cur_base.fetchall()
            if flaw_base_res != ref_base_res:
                is_caught = True
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            is_caught = True

        # 2. If not caught on base DB, test on mutant DB (mutation testing benefit)
        if not is_caught:
            try:
                cur_mut.execute(flawed_query)
                flaw_mut_res = cur_mut.fetchall()
                if flaw_mut_res != ref_mut_res:
                    is_caught = True
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                is_caught = True

        assert is_caught, (
            f"Flawed query for {case_id} unexpectedly matched reference on both base and mutant!"
        )
        if is_caught:
            detected_count += 1

    assert detected_count == len(smoke_cases) == 35
