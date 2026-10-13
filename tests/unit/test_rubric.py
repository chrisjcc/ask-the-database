"""Unit tests for SQLRewardRubric hierarchical reward scoring."""

from pathlib import Path

from ask_the_database.rewards.rubric import SQLRewardRubric


def test_syntax_error_penalty(sample_sqlite_db: Path) -> None:
    """Test that broken syntax receives syntax penalty (-1.0)."""
    rubric = SQLRewardRubric()
    candidate = "SELECT FROM WHERE ;"
    gold = "SELECT name FROM users;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))

    assert out.is_valid_syntax is False
    assert out.is_valid_execution is False
    assert out.is_execution_equivalent is False
    assert out.total_reward == -1.0
    assert out.diagnostic_payload["stage"] == "syntax_error"


def test_runtime_execution_error_penalty(sample_sqlite_db: Path) -> None:
    """Test that query failing at DB runtime receives execution penalty (-0.5)."""
    rubric = SQLRewardRubric()
    candidate = "SELECT non_existent_column FROM users;"
    gold = "SELECT name FROM users;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))

    assert out.is_valid_syntax is True
    assert out.is_valid_execution is False
    assert out.total_reward == -0.5
    assert out.diagnostic_payload["stage"] == "execution_error"


def test_semantic_mismatch_bounded_invariant(sample_sqlite_db: Path) -> None:
    """Test that syntactically valid query returning wrong rows receives score <= 0.0."""
    rubric = SQLRewardRubric()
    candidate = "SELECT name FROM users WHERE age < 25;"
    gold = "SELECT name FROM users WHERE age > 30;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))

    assert out.is_valid_syntax is True
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is False
    # Hierarchical Bounded Invariant: must NEVER exceed 0.0
    assert out.total_reward <= 0.0
    assert out.diagnostic_payload["stage"] == "semantic_mismatch"


def test_full_correctness_reward(sample_sqlite_db: Path) -> None:
    """Test that query passing both base DB and test suite gets full reward (+1.0)."""
    rubric = SQLRewardRubric()
    query = "SELECT name, age FROM users WHERE age >= 25 ORDER BY age;"

    out = rubric.evaluate(query, query, str(sample_sqlite_db), requires_strict_order=True)

    assert out.is_valid_syntax is True
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is True
    assert out.is_test_suite_equivalent is True
    assert out.total_reward == 1.0
    assert out.diagnostic_payload["stage"] == "full_correctness"


def test_partial_reward_on_test_suite_failure(sample_sqlite_db: Path) -> None:
    """Test that query matching base DB but failing test suite gets partial reward (+0.5)."""
    rubric = SQLRewardRubric()
    # Missing DISTINCT query matches base DB but fails mutation 2
    gold = "SELECT DISTINCT department_name FROM departments;"
    candidate = "SELECT department_name FROM departments;"

    out = rubric.evaluate(candidate, gold, str(sample_sqlite_db))

    assert out.is_valid_syntax is True
    assert out.is_valid_execution is True
    assert out.is_execution_equivalent is True
    assert out.is_test_suite_equivalent is False
    assert out.total_reward == 0.5
    assert out.diagnostic_payload["stage"] == "test_suite_failure"
