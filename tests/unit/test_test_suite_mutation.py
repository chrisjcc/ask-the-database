"""Unit tests for TestSuiteEvaluator database mutations and TS verification."""

import glob
import tempfile
from pathlib import Path

from ask_the_database.environment.test_suite_evaluator import TestSuiteEvaluator


def test_identical_queries_pass_all_suites(sample_sqlite_db: Path) -> None:
    """Test that two semantically identical queries pass all mutated test suites."""
    evaluator = TestSuiteEvaluator(num_mutations=4)
    query = "SELECT name, age FROM users WHERE age >= 25 ORDER BY age;"

    result = evaluator.evaluate_test_suite(
        str(sample_sqlite_db),
        gen_sql=query,
        gold_sql=query,
        requires_strict_order=True,
    )

    assert result.passed_all_suites is True
    assert result.suites_evaluated == 5  # 1 base + 4 mutated
    assert result.suites_passed == 5
    assert result.failed_suite_index is None
    assert result.failure_reason is None


def test_base_mismatch_fails_early(sample_sqlite_db: Path) -> None:
    """Test that query failing base DB stops immediately without running mutations."""
    evaluator = TestSuiteEvaluator(num_mutations=4)
    gold = "SELECT name FROM users WHERE age > 30;"
    gen = "SELECT name FROM users WHERE age < 25;"

    result = evaluator.evaluate_test_suite(str(sample_sqlite_db), gen_sql=gen, gold_sql=gold)

    assert result.passed_all_suites is False
    assert result.suites_evaluated == 1
    assert result.suites_passed == 0
    assert result.failed_suite_index == 0


def test_missing_distinct_caught_by_mutation(sample_sqlite_db: Path) -> None:
    """Test that a query missing DISTINCT matches base DB but fails on duplicate mutation."""
    evaluator = TestSuiteEvaluator(num_mutations=4)

    # In base DB, department_names are all unique ('Engineering', 'Marketing', 'Finance')
    # Both queries return the exact same 3 rows on base DB!
    gold = "SELECT DISTINCT department_name FROM departments;"
    gen = "SELECT department_name FROM departments;"

    result = evaluator.evaluate_test_suite(str(sample_sqlite_db), gen_sql=gen, gold_sql=gold)

    # Must fail on mutation 2 (duplicate injection)
    assert result.passed_all_suites is False
    assert result.failed_suite_index == 2
    assert result.failure_reason == "MUTATED_STATE_MISMATCH"


def test_ephemeral_cleanup_after_evaluation(sample_sqlite_db: Path) -> None:
    """Test that ephemeral mutation clone directories are completely removed."""
    evaluator = TestSuiteEvaluator(num_mutations=4)
    query = "SELECT name FROM users;"

    evaluator.evaluate_test_suite(str(sample_sqlite_db), gen_sql=query, gold_sql=query)

    # Check that temporary directories with prefix 'sqlite_mutation_' are cleaned up
    temp_root = tempfile.gettempdir()
    remaining = glob.glob(f"{temp_root}/sqlite_mutation_*")
    assert len(remaining) == 0
