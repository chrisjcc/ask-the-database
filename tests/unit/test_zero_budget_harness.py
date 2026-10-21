"""Unit tests for the Zero-Budget Evaluation Harness engine.

Validates in-process SQLite execution speed (< 3.0s), trap scoring attribution,
mutation hardening, decision table remediation routing, and query plan checks.
"""

import pytest

from ask_the_database.evaluation.zero_budget_harness import (
    CaseVerdict,
    DecisionTable,
    HarnessSummary,
    QueryPlanGuard,
    TrapCategory,
    TrapScorer,
    ZeroBudgetHarness,
)


@pytest.fixture
def harness() -> ZeroBudgetHarness:
    """Provides an initialized ZeroBudgetHarness."""
    return ZeroBudgetHarness()


def test_harness_golden_evaluation_pass(harness: ZeroBudgetHarness) -> None:
    """Verifies that all 35 canonical reference queries achieve 100% pass rate in < 3s."""
    summary: HarnessSummary = harness.evaluate_golden()

    assert summary.total_cases == 35
    assert summary.passed_cases == 35
    assert summary.failed_cases == 0
    assert summary.pass_rate == 1.0
    assert summary.base_pass_rate == 1.0
    assert summary.mutant_pass_rate == 1.0
    assert summary.mutation_detected_count == 0

    # Gate 1: Latency <= 3.0 seconds (3000 ms)
    assert summary.total_latency_ms <= 3000.0, (
        f"Execution too slow: {summary.total_latency_ms:.2f} ms"
    )
    assert len(summary.case_verdicts) == 35
    for verdict in summary.case_verdicts:
        assert verdict.passed is True
        assert verdict.status == "PASS"


def test_harness_flawed_evaluation_detection_and_attribution(
    harness: ZeroBudgetHarness,
) -> None:
    """Verifies 100% bug detection (Gate 4) and >= 85% trap attribution accuracy (Gate 3)."""
    summary: HarnessSummary = harness.evaluate_flawed()

    # Gate 4: False-positive immunity (100% bugs detected)
    assert summary.total_cases == 35
    assert summary.passed_cases == 0
    assert summary.failed_cases == 35
    assert summary.pass_rate == 0.0

    # Gate 3: Diagnostic attribution accuracy >= 85%
    assert summary.attribution_accuracy >= 0.85, (
        f"Attribution accuracy below 85%: {summary.attribution_accuracy:.2%}"
    )

    # Balanced trap distribution across all 5 traps
    for trap in [
        TrapCategory.JOIN_FANOUT.value,
        TrapCategory.DIALECT_DRIFT.value,
        TrapCategory.HALLUCINATED_COLUMN.value,
        TrapCategory.AGGREGATION_WITHOUT_GROUPING.value,
        TrapCategory.NULL_BOUNDARY.value,
    ]:
        assert summary.misses_by_trap[trap] == 7

    assert summary.misses_by_trap[TrapCategory.UNKNOWN.value] == 0


def test_trap_scorer_categories() -> None:
    """Test individual trap classification branches in TrapScorer."""
    scorer = TrapScorer()

    # 1. Dialect Drift
    drift = scorer.score_failure(
        candidate_sql="SELECT EXTRACT(YEAR FROM order_date) FROM orders;",
        gold_sql="SELECT strftime('%Y', order_date) FROM orders;",
        error_message=None,
        candidate_data=None,
        gold_data=None,
    )
    assert drift == TrapCategory.DIALECT_DRIFT.value

    # 2. Hallucinated Column
    hallucinated = scorer.score_failure(
        candidate_sql="SELECT missing_col FROM orders;",
        gold_sql="SELECT total_amount FROM orders;",
        error_message="no such column: missing_col",
        candidate_data=None,
        gold_data=None,
    )
    assert hallucinated == TrapCategory.HALLUCINATED_COLUMN.value

    # 3. Aggregation Without Grouping
    agg = scorer.score_failure(
        candidate_sql="SELECT dept_id, MAX(salary) FROM employees;",
        gold_sql="SELECT dept_id, MAX(salary) FROM employees GROUP BY dept_id;",
        error_message=None,
        candidate_data=[(1, 185000.0)],
        gold_data=[(1, 185000.0), (2, 160000.0), (3, 110000.0)],
    )
    assert agg == TrapCategory.AGGREGATION_WITHOUT_GROUPING.value

    # 4. Null Boundary
    null_trap = scorer.score_failure(
        candidate_sql="SELECT emp_id FROM employees WHERE manager_id = 0;",
        gold_sql="SELECT emp_id FROM employees WHERE manager_id IS NULL;",
        error_message=None,
        candidate_data=[],
        gold_data=[(101,)],
    )
    assert null_trap == TrapCategory.NULL_BOUNDARY.value

    # 5. Join Fanout
    fanout = scorer.score_failure(
        candidate_sql="SELECT COUNT(c.customer_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id;",
        gold_sql="SELECT COUNT(DISTINCT c.customer_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id;",
        error_message=None,
        candidate_data=[(8,)],
        gold_data=[(6,)],
    )
    assert fanout == TrapCategory.JOIN_FANOUT.value


def test_decision_table_generation() -> None:
    """Test prescriptive remediation generation from trap concentrations."""
    misses = {
        TrapCategory.JOIN_FANOUT.value: 8,
        TrapCategory.HALLUCINATED_COLUMN.value: 5,
        TrapCategory.DIALECT_DRIFT.value: 2,
        TrapCategory.AGGREGATION_WITHOUT_GROUPING.value: 0,
        TrapCategory.NULL_BOUNDARY.value: 1,
    }
    total_failures = 16

    remediations = DecisionTable.generate_remediations(misses, total_failures)

    assert len(remediations) == 4
    traps_recommended = {r["trap"] for r in remediations}
    assert TrapCategory.JOIN_FANOUT.value in traps_recommended
    assert TrapCategory.HALLUCINATED_COLUMN.value in traps_recommended

    # Highest ratio trap should have HIGH severity
    join_rec = next(r for r in remediations if r["trap"] == TrapCategory.JOIN_FANOUT.value)
    assert join_rec["severity"] == "HIGH"
    assert "cardinality" in join_rec["prescriptive_action"].lower()


def test_query_plan_guard(harness: ZeroBudgetHarness) -> None:
    """Test QueryPlanGuard EXPLAIN QUERY PLAN analysis on seed schema."""
    conn = harness._create_database(is_mutant=False)
    try:
        report = QueryPlanGuard.inspect(conn, "SELECT * FROM customers WHERE customer_id = 201;")
        assert len(report.plan_rows) > 0

        # Syntax error inspection handling
        err_report = QueryPlanGuard.inspect(conn, "SELECT FROM WHERE;")
        assert err_report.is_safe is False
        assert len(err_report.warnings) > 0
    finally:
        conn.close()


def test_run_case_error_and_mismatch(harness: ZeroBudgetHarness) -> None:
    """Test that runtime syntax errors and result mismatches are correctly recorded."""
    cases = harness.load_cases()
    first_case = cases[0]

    # Syntax error
    err_verdict: CaseVerdict = harness.run_case(first_case, "SELECT INVALID SYNTAX * FROM;")
    assert err_verdict.passed is False
    assert err_verdict.status == "ERROR"
    assert err_verdict.error_message is not None

    # Mismatch
    mismatch_verdict: CaseVerdict = harness.run_case(first_case, "SELECT 999999.0 AS total_amount;")
    assert mismatch_verdict.passed is False
    assert mismatch_verdict.status == "MISMATCH"
