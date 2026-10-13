"""Unit tests verifying edge cases in the verifiable reward engine."""

from pathlib import Path

from ask_the_database.rewards.rubric import SQLRewardRubric


def test_float_epsilon_matching(sample_sqlite_db: Path) -> None:
    """Test floating point numbers within epsilon tolerance (1e-4) match with full reward."""
    rubric = SQLRewardRubric()
    # 1.00004 vs 1.00001 round to 1.0000
    candidate = "SELECT 1.00004;"
    gold = "SELECT 1.00001;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_valid_syntax is True
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is True
    assert out.total_reward > 0.0


def test_runtime_error_execution_penalty(sample_sqlite_db: Path) -> None:
    """Test that runtime SQLite execution errors receive the execution penalty (-0.5)."""
    rubric = SQLRewardRubric()
    # malformed JSON triggers sqlite3.OperationalError
    candidate = "SELECT json('invalid_json') FROM users;"
    gold = "SELECT name FROM users;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_valid_syntax is True
    assert out.is_valid_execution is False
    assert out.total_reward == -0.5
    assert out.diagnostic_payload["stage"] == "execution_error"


def test_empty_set_semantic_match(sample_sqlite_db: Path) -> None:
    """Test that two valid queries legitimately returning 0 rows receive full reward."""
    rubric = SQLRewardRubric()
    candidate = "SELECT name FROM users WHERE age > 9999;"
    gold = "SELECT name FROM users WHERE age > 9999;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_valid_syntax is True
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is True
    assert out.total_reward == 1.0


def test_alias_invariance(sample_sqlite_db: Path) -> None:
    """Test that column aliasing differences do not affect tabular multiset equivalence."""
    rubric = SQLRewardRubric()
    candidate = "SELECT age AS my_age, name AS my_name FROM users;"
    gold = "SELECT age, name FROM users;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_execution_equivalent is True
    assert out.total_reward == 1.0


def test_null_vs_empty_string_differentiation(sample_sqlite_db: Path) -> None:
    """Test that NULL is strictly distinguished from empty string."""
    rubric = SQLRewardRubric()
    candidate = "SELECT '' AS val;"
    gold = "SELECT NULL AS val;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is False
    assert out.total_reward <= 0.0


def test_aggregate_count_zero_vs_empty_rows(sample_sqlite_db: Path) -> None:
    """Test that COUNT(*) returning [(0,)] is NOT equivalent to empty rowset []."""
    rubric = SQLRewardRubric()
    # Candidate returns 1 row containing 0
    candidate = "SELECT COUNT(*) FROM users WHERE age > 9999;"
    # Gold returns 0 rows
    gold = "SELECT name FROM users WHERE age > 9999;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is False
    assert out.total_reward <= 0.0


def test_case_insensitive_string_values(sample_sqlite_db: Path) -> None:
    """Test string casing normalization in result comparison."""
    rubric = SQLRewardRubric()
    candidate = "SELECT 'alice' AS name;"
    gold = "SELECT 'ALICE' AS name;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))
    assert out.is_execution_equivalent is True
    assert out.total_reward == 1.0
