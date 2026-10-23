"""Execution equivalence evaluator for SQL query results."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ask_the_database.environment.base import ExecutionResult


@dataclass(frozen=True)
class ExecutionEquivalenceResult:
    """Comparison result between generated and gold query execution."""

    is_equivalent: bool
    gold_executed: bool
    gen_executed: bool
    gold_row_count: int
    gen_row_count: int
    strict_order_checked: bool
    error_reason: (
        str | None
    )  # "GOLD_EXEC_ERROR", "GEN_EXEC_ERROR", "ROW_COUNT_MISMATCH", "ORDER_MISMATCH", "CONTENT_MISMATCH", None


class ExecutionEquivalenceEvaluator:
    """Evaluates tabular result set equivalence between two SQL queries."""

    def __init__(self, float_tolerance: float = 1e-4) -> None:
        self.float_tolerance = float_tolerance

    def _normalize_cell(self, cell: Any) -> Any:
        """Normalize individual cell values for comparison."""
        if cell is None:
            return None
        if isinstance(cell, float):
            return round(cell, 4)
        if isinstance(cell, int | bool):
            return cell
        # String normalization: strip whitespace and standardize casing
        str_val = str(cell).strip()
        # Attempt numeric cast if string is a number
        try:
            float_val = float(str_val)
            return round(float_val, 4)
        except ValueError:
            return str_val.lower()

    def _normalize_row(self, row: tuple[Any, ...]) -> tuple[Any, ...]:
        """Normalize a single row of values."""
        return tuple(self._normalize_cell(c) for c in row)

    def compare(
        self,
        gen_result: ExecutionResult,
        gold_result: ExecutionResult,
        requires_strict_order: bool = False,
    ) -> ExecutionEquivalenceResult:
        """Compare two execution results for multiset semantic equivalence."""
        if not gold_result.success or gold_result.data is None:
            return ExecutionEquivalenceResult(
                is_equivalent=False,
                gold_executed=False,
                gen_executed=gen_result.success,
                gold_row_count=0,
                gen_row_count=gen_result.row_count,
                strict_order_checked=requires_strict_order,
                error_reason="GOLD_EXEC_ERROR",
            )

        if not gen_result.success or gen_result.data is None:
            return ExecutionEquivalenceResult(
                is_equivalent=False,
                gold_executed=True,
                gen_executed=False,
                gold_row_count=gold_result.row_count,
                gen_row_count=0,
                strict_order_checked=requires_strict_order,
                error_reason="GEN_EXEC_ERROR",
            )

        # Normalize rows
        norm_gold = [self._normalize_row(r) for r in gold_result.data]
        norm_gen = [self._normalize_row(r) for r in gen_result.data]

        # Check row count
        if len(norm_gold) != len(norm_gen):
            return ExecutionEquivalenceResult(
                is_equivalent=False,
                gold_executed=True,
                gen_executed=True,
                gold_row_count=len(norm_gold),
                gen_row_count=len(norm_gen),
                strict_order_checked=requires_strict_order,
                error_reason="ROW_COUNT_MISMATCH",
            )

        # Strict sequence equality (if ORDER BY is enforced)
        if requires_strict_order:
            is_match = norm_gold == norm_gen
            return ExecutionEquivalenceResult(
                is_equivalent=is_match,
                gold_executed=True,
                gen_executed=True,
                gold_row_count=len(norm_gold),
                gen_row_count=len(norm_gen),
                strict_order_checked=True,
                error_reason=None if is_match else "ORDER_MISMATCH",
            )

        # Order-agnostic multiset (bag) comparison
        gold_counter = Counter(norm_gold)
        gen_counter = Counter(norm_gen)

        is_match = gold_counter == gen_counter
        return ExecutionEquivalenceResult(
            is_equivalent=is_match,
            gold_executed=True,
            gen_executed=True,
            gold_row_count=len(norm_gold),
            gen_row_count=len(norm_gen),
            strict_order_checked=False,
            error_reason=None if is_match else "CONTENT_MISMATCH",
        )
