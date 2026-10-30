"""Unit tests for SQLVerifier and Continuous Logit Expectation (Sub-Issue #68)."""

import math

import pytest

from ask_the_database.verification.sql_verifier import (
    DEFAULT_SCORING_TOKENS,
    SQLVerifier,
    VerifierCriterion,
    compute_continuous_expectation,
)


def test_compute_continuous_expectation_uniform() -> None:
    """Verify uniform logit distribution produces the expected mid-point score."""
    logits = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0}
    expected_score, probs = compute_continuous_expectation(logits, DEFAULT_SCORING_TOKENS)

    # Values are 0.0, 0.25, 0.50, 0.75, 1.00 -> Mean is 0.50
    assert math.isclose(expected_score, 0.50, abs_tol=1e-3)
    assert len(probs) == 5
    for p in probs.values():
        assert math.isclose(p, 0.20, abs_tol=1e-3)


def test_compute_continuous_expectation_skewed() -> None:
    """Verify that placing all logit mass on highest token yields 1.0."""
    logits = {"1": -10.0, "2": -10.0, "3": -10.0, "4": -10.0, "5": 20.0}
    expected_score, probs = compute_continuous_expectation(logits, DEFAULT_SCORING_TOKENS)

    assert expected_score >= 0.99
    assert probs["5"] >= 0.99


def test_compute_continuous_expectation_logprobs() -> None:
    """Verify calculation when is_logprobs=True."""
    # log(0.5) ~ -0.693, log(0.5) ~ -0.693
    logprobs = {"1": -0.6931, "5": -0.6931}
    expected_score, probs = compute_continuous_expectation(
        logprobs, DEFAULT_SCORING_TOKENS, is_logprobs=True
    )
    # Expected value: 0.5 * 0.0 + 0.5 * 1.0 = 0.50
    assert math.isclose(expected_score, 0.50, abs_tol=1e-3)


def test_compute_continuous_expectation_empty_and_irrelevant() -> None:
    """Verify graceful handling of empty or unrelated token maps."""
    score, probs = compute_continuous_expectation({})
    assert score == 0.0
    assert probs == {}

    score2, probs2 = compute_continuous_expectation({"apple": 2.0, "banana": 3.0})
    assert score2 == 0.0
    assert probs2 == {}


def test_verifier_criterion_dataclass() -> None:
    """Verify VerifierCriterion properties."""
    crit = VerifierCriterion(name="intent", description="Check filters", weight=0.35)
    assert crit.name == "intent"
    assert crit.weight == 0.35


def test_sql_verifier_initialization_from_yaml() -> None:
    """Verify SQLVerifier loads rules and normalizes criteria weights."""
    verifier = SQLVerifier("configs/verification/sql_verifier_rules.yaml")
    assert verifier.model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert "intent_and_filters" in verifier.criteria
    assert "schema_grounding_and_joins" in verifier.criteria

    # Check that criteria weights sum to 1.0
    total_weight = sum(c.weight for c in verifier.criteria.values())
    assert math.isclose(total_weight, 1.0, abs_tol=1e-3)


def test_sql_verifier_prompt_builder() -> None:
    """Verify prompt formatting with criteria."""
    verifier = SQLVerifier()
    crit = VerifierCriterion("schema_joins", "Check joins", 0.3)
    prompt = verifier.build_verification_prompt(
        question="Find active customers",
        schema_ddl="CREATE TABLE customers(id INT, active BOOL);",
        sql_query="SELECT id FROM customers WHERE active = 1",
        criterion=crit,
    )
    assert "Find active customers" in prompt
    assert "CREATE TABLE customers" in prompt
    assert "SELECT id FROM customers WHERE active = 1" in prompt
    assert "schema_joins" in prompt


def test_sql_verifier_client_mock() -> None:
    """Verify custom scoring client is called and computes expectation."""

    # Mock client returns high logit for score 5 when 'where' is in query
    def mock_client(prompt: str) -> dict[str, float]:
        sql_part = prompt.split("```sql")[1].split("```")[0] if "```sql" in prompt else ""
        if "where" in sql_part.lower():
            return {"1": 0.0, "2": 0.0, "3": 0.0, "4": 2.0, "5": 8.0}
        return {"1": 8.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}

    verifier = SQLVerifier(client=mock_client)
    res_good = verifier.score_query(
        sql_query="SELECT name FROM users WHERE id = 10",
        question="Find user 10",
        schema_ddl="CREATE TABLE users(id INT, name TEXT);",
    )
    assert res_good.overall_score >= 0.80

    res_bad = verifier.score_query(
        sql_query="SELECT name FROM users",
        question="Find user 10",
        schema_ddl="CREATE TABLE users(id INT, name TEXT);",
    )
    assert res_bad.overall_score <= 0.20


def test_sql_verifier_fallback_heuristic_syntax_error() -> None:
    """Verify that syntax errors produce severe penalties in heuristic fallback."""
    verifier = SQLVerifier()
    res_syntax_err = verifier.score_query(
        sql_query="SELEC * FORM broken syntax!!",
        question="Get all data",
        schema_ddl="CREATE TABLE t(id INT);",
    )
    assert res_syntax_err.overall_score <= 0.20


def test_sql_verifier_probabilistic_pivot_tournament() -> None:
    """Verify tournament ranking sorts candidates descending by score."""
    verifier = SQLVerifier()
    candidates = [
        {"sql": "SELEC * BROKEN SYNTAX", "id": "cand_bad"},
        {"sql": "SELECT name FROM students WHERE gpa > 3.5", "id": "cand_good"},
        {"sql": "SELECT * FROM students", "id": "cand_mediocre"},
    ]

    ranked = verifier.probabilistic_pivot_tournament(
        candidates,
        question="Find students with gpa above 3.5",
        schema_ddl="CREATE TABLE students(id INT, name TEXT, gpa FLOAT);",
    )
    assert len(ranked) == 3
    # Best should be cand_good
    assert ranked[0].candidate_id == "cand_good"
    # Scores must be in descending order
    assert ranked[0].overall_score >= ranked[1].overall_score >= ranked[2].overall_score

    best = verifier.select_best(
        candidates,
        question="Find students with gpa above 3.5",
        schema_ddl="CREATE TABLE students(id INT, name TEXT, gpa FLOAT);",
    )
    assert "gpa > 3.5" in best.sql_query


def test_sql_verifier_select_best_empty() -> None:
    """Verify ValueError is raised on empty candidates list."""
    verifier = SQLVerifier()
    with pytest.raises(ValueError, match="No candidates provided"):
        verifier.select_best([], "question", "ddl")
