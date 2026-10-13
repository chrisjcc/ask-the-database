"""Unit tests for ExecutionEquivalenceEvaluator table comparison logic."""

from ask_the_database.environment.base import ExecutionResult
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator


def make_exec_result(
    success: bool = True,
    data: list[tuple[object, ...]] | None = None,
    columns: list[str] | None = None,
    error_message: str | None = None,
    error_type: str | None = None,
) -> ExecutionResult:
    """Helper to construct ExecutionResult objects for testing."""
    return ExecutionResult(
        success=success,
        data=data,
        columns=columns or ["col1"],
        row_count=len(data) if data is not None else 0,
        execution_time_ms=5.0,
        error_message=error_message,
        error_type=error_type,
    )


def test_identical_results() -> None:
    """Test identical tabular rows match."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[("Alice", 25), ("Bob", 30)])
    res2 = make_exec_result(data=[("Alice", 25), ("Bob", 30)])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is True
    assert cmp.error_reason is None


def test_order_agnostic_multiset_match() -> None:
    """Test rows appearing in different order match when strict order is False."""
    evaluator = ExecutionEquivalenceEvaluator()
    res_gold = make_exec_result(data=[("Alice", 25), ("Bob", 30), ("Charlie", 35)])
    res_gen = make_exec_result(data=[("Bob", 30), ("Charlie", 35), ("Alice", 25)])

    cmp = evaluator.compare(res_gen, res_gold, requires_strict_order=False)
    assert cmp.is_equivalent is True


def test_strict_order_mismatch() -> None:
    """Test rows appearing in different order fail when strict order is True."""
    evaluator = ExecutionEquivalenceEvaluator()
    res_gold = make_exec_result(data=[("Alice", 25), ("Bob", 30)])
    res_gen = make_exec_result(data=[("Bob", 30), ("Alice", 25)])

    cmp = evaluator.compare(res_gen, res_gold, requires_strict_order=True)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "ORDER_MISMATCH"


def test_float_epsilon_tolerance() -> None:
    """Test floating point numbers rounded to 4 decimal places match."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[(3.14159, "pi")])
    res2 = make_exec_result(data=[(3.14160, "pi")])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is True


def test_numeric_string_normalization() -> None:
    """Test numeric strings match integer/float representations."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[("100", "test")])
    res2 = make_exec_result(data=[(100.0, "test")])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is True


def test_row_count_mismatch() -> None:
    """Test differing number of rows returns ROW_COUNT_MISMATCH."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[("Alice",)])
    res2 = make_exec_result(data=[("Alice",), ("Bob",)])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "ROW_COUNT_MISMATCH"


def test_content_mismatch() -> None:
    """Test same row count with different values returns CONTENT_MISMATCH."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[("Alice", 25)])
    res2 = make_exec_result(data=[("Alice", 30)])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "CONTENT_MISMATCH"


def test_gen_execution_failure() -> None:
    """Test failed generated query returns GEN_EXEC_ERROR."""
    evaluator = ExecutionEquivalenceEvaluator()
    res_gold = make_exec_result(data=[("Alice",)])
    res_gen = make_exec_result(success=False, data=None, error_message="Syntax error")

    cmp = evaluator.compare(res_gen, res_gold)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "GEN_EXEC_ERROR"


def test_gold_execution_failure() -> None:
    """Test failed gold query returns GOLD_EXEC_ERROR."""
    evaluator = ExecutionEquivalenceEvaluator()
    res_gold = make_exec_result(success=False, data=None, error_message="DB locked")
    res_gen = make_exec_result(data=[("Alice",)])

    cmp = evaluator.compare(res_gen, res_gold)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "GOLD_EXEC_ERROR"


def test_null_vs_empty_string() -> None:
    """Test NULL is not equal to empty string."""
    evaluator = ExecutionEquivalenceEvaluator()
    res1 = make_exec_result(data=[(None,)])
    res2 = make_exec_result(data=[("",)])

    cmp = evaluator.compare(res1, res2)
    assert cmp.is_equivalent is False
    assert cmp.error_reason == "CONTENT_MISMATCH"
