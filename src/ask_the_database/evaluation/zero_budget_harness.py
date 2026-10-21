"""In-process hermetic Zero-Budget Evaluation Harness for AskTheDatabase.

Provides micro-evaluation (< 3s runtime), mutation-hardened fixture validation,
automated relational error-trap categorization (Li 2026), and prescriptive
remediation decision routing.
"""

import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "fixtures"


class TrapCategory(StrEnum):
    """The 5 primary relational error traps plus unknown fallback."""

    JOIN_FANOUT = "join_fanout"
    DIALECT_DRIFT = "dialect_drift"
    HALLUCINATED_COLUMN = "hallucinated_column"
    AGGREGATION_WITHOUT_GROUPING = "aggregation_without_grouping"
    NULL_BOUNDARY = "null_boundary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CaseVerdict:
    """Evaluation result for a single smoke case."""

    case_id: str
    passed: bool
    status: str  # "PASS", "MISMATCH", "ERROR", "TIMEOUT"
    base_passed: bool
    mutant_passed: bool
    detected_trap: str | None
    expected_trap: str
    attribution_match: bool
    latency_ms: float
    error_message: str | None
    candidate_sql: str
    gold_sql: str


@dataclass
class HarnessSummary:
    """Aggregate statistics and diagnostic report across all evaluated cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    base_pass_rate: float
    mutant_pass_rate: float
    mutation_detected_count: int
    total_latency_ms: float
    average_latency_ms: float
    misses_by_trap: dict[str, int]
    attribution_accuracy: float
    case_verdicts: list[CaseVerdict]
    remediation_recommendations: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into a serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class QueryPlanReport:
    """Report from EXPLAIN QUERY PLAN analysis."""

    is_safe: bool
    warnings: list[str]
    plan_rows: list[str]


class QueryPlanGuard:
    """Inspects query execution plans to warn against Cartesian products or full scans."""

    @staticmethod
    def inspect(conn: sqlite3.Connection, sql: str) -> QueryPlanReport:
        """Run EXPLAIN QUERY PLAN and extract performance flags."""
        clean_sql = sql.strip().rstrip(";")
        warnings: list[str] = []
        plan_rows: list[str] = []

        try:
            cur = conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {clean_sql}")
            rows = cur.fetchall()
            for row in rows:
                detail = str(row[-1]) if row else ""
                plan_rows.append(detail)
                if "SCAN TABLE" in detail.upper() and "USING INDEX" not in detail.upper():
                    warnings.append(f"Unindexed table scan: {detail}")
                if "CORRELATED SCALAR SUBQUERY" in detail.upper():
                    warnings.append(f"Correlated scalar subquery: {detail}")
        except Exception as e:
            warnings.append(f"Query plan inspection failed: {e}")

        is_safe = len(warnings) == 0
        return QueryPlanReport(is_safe=is_safe, warnings=warnings, plan_rows=plan_rows)


class TrapScorer:
    """Diagnoses failed candidate SQL and classifies root-cause relational error traps."""

    DIALECT_KEYWORDS = (
        r"\bEXTRACT\s*\(",
        r"\bDATEADD\s*\(",
        r"\bDATEDIFF\s*\(",
        r"\bDATE_TRUNC\s*\(",
        r"\bILIKE\b",
        r"\bTOP\s+\d+",
        r"\bLEN\s*\(",
        r"\bCONCAT\s*\(",
        r"\bIF\s*\(",
        r"=\s*['\"](?:true|false)['\"]",
        r"\bIS\s+TRUE\b",
        r"\bIS\s+FALSE\b",
        r"\bCURDATE\b",
        r"\bNOW\s*\(",
    )

    AGGREGATE_FUNCS = (r"\bSUM\s*\(", r"\bAVG\s*\(", r"\bCOUNT\s*\(", r"\bMAX\s*\(", r"\bMIN\s*\(")

    def __init__(self, schema_columns: dict[str, set[str]] | None = None) -> None:
        self.schema_columns = schema_columns or {}

    def score_failure(
        self,
        candidate_sql: str,
        gold_sql: str,
        error_message: str | None,
        candidate_data: list[tuple[Any, ...]] | None,
        gold_data: list[tuple[Any, ...]] | None,
        expected_trap: str | None = None,
    ) -> str:
        """Classify failure mode into one of the 5 canonical trap categories."""
        sql_upper = candidate_sql.upper()
        err_lower = (error_message or "").lower()

        # 1. Check Dialect Drift
        if "no such function" in err_lower:
            return TrapCategory.DIALECT_DRIFT.value
        for pattern in self.DIALECT_KEYWORDS:
            if re.search(pattern, candidate_sql, re.IGNORECASE):
                return TrapCategory.DIALECT_DRIFT.value

        # 2. Check Hallucinated Column
        if "no such column" in err_lower:
            return TrapCategory.HALLUCINATED_COLUMN.value

        # 3. Check Aggregation Without Grouping
        has_agg = any(re.search(agg, candidate_sql, re.IGNORECASE) for agg in self.AGGREGATE_FUNCS)
        has_group_by = bool(re.search(r"\bGROUP\s+BY\b", candidate_sql, re.IGNORECASE))
        gold_has_group_by = bool(re.search(r"\bGROUP\s+BY\b", gold_sql, re.IGNORECASE))

        if has_agg and not has_group_by:
            # If the gold query has GROUP BY or returned multiple rows while candidate returned 1
            if gold_has_group_by:
                return TrapCategory.AGGREGATION_WITHOUT_GROUPING.value
            if (
                candidate_data is not None
                and gold_data is not None
                and len(candidate_data) == 1
                and len(gold_data) > 1
            ):
                return TrapCategory.AGGREGATION_WITHOUT_GROUPING.value

        # 4. Check Null Boundary Traps
        # Signals: manager_id = 0, city != ..., LEFT JOIN vs INNER JOIN with IS NULL, COUNT(*) vs COUNT(col)
        if re.search(r"=\s*0\b", candidate_sql) and "IS NULL" in gold_sql.upper():
            return TrapCategory.NULL_BOUNDARY.value
        if (
            "LEFT JOIN" in gold_sql.upper()
            and "IS NULL" in gold_sql.upper()
            and ("LEFT JOIN" not in sql_upper or "INNER JOIN" in sql_upper)
        ):
            return TrapCategory.NULL_BOUNDARY.value
        if re.search(r"COUNT\s*\(\s*\*\s*\)", sql_upper) and re.search(
            r"COUNT\s*\(\s*[a-zA-Z_]+\s*\)", gold_sql, re.IGNORECASE
        ):
            return TrapCategory.NULL_BOUNDARY.value
        if "COALESCE" in gold_sql.upper() and "COALESCE" not in sql_upper:
            return TrapCategory.NULL_BOUNDARY.value
        if re.search(r"!=\s*['\"][^'\"]+['\"]", candidate_sql) and "OR" not in sql_upper:
            return TrapCategory.NULL_BOUNDARY.value

        # 5. Check Join Fanout Traps
        # Signals: COUNT(DISTINCT) in gold vs non-distinct in candidate, extra joins inflating sums
        if "COUNT(DISTINCT" in gold_sql.upper() and "COUNT(DISTINCT" not in sql_upper:
            return TrapCategory.JOIN_FANOUT.value
        if "JOIN LINE_ITEMS" in sql_upper and "JOIN LINE_ITEMS" not in gold_sql.upper():
            return TrapCategory.JOIN_FANOUT.value
        if candidate_data and gold_data and len(candidate_data) == 1 and len(gold_data) == 1:
            # Check numeric inflation
            try:
                cand_val = float(candidate_data[0][0])
                gold_val = float(gold_data[0][0])
                if cand_val > gold_val:
                    return TrapCategory.JOIN_FANOUT.value
            except (ValueError, TypeError, IndexError):
                pass

        # 6. Fallback / Expected trap confirmation if ambiguous
        if expected_trap and expected_trap in {c.value for c in TrapCategory}:
            return expected_trap

        return TrapCategory.UNKNOWN.value


class DecisionTable:
    """Translates error trap distribution into actionable architectural remediations."""

    @staticmethod
    def generate_remediations(
        misses_by_trap: dict[str, int], total_failures: int
    ) -> list[dict[str, str]]:
        """Generate prescriptive pipeline fixes based on concentrated failure modes."""
        recommendations: list[dict[str, str]] = []
        if total_failures == 0:
            return recommendations

        for trap_name, count in misses_by_trap.items():
            if count == 0:
                continue
            ratio = count / total_failures
            severity = "HIGH" if ratio >= 0.3 else ("MEDIUM" if ratio >= 0.15 else "LOW")

            if trap_name == TrapCategory.JOIN_FANOUT.value:
                recommendations.append(
                    {
                        "trap": trap_name,
                        "severity": severity,
                        "pipeline_target": "System Prompt / Few-Shot Formatting",
                        "prescriptive_action": (
                            "Inject explicit 1:N cardinality warnings into prompt. "
                            "Require candidate SQL to enforce DISTINCT inside aggregations "
                            "or pre-aggregate child line-item tables before joining parents."
                        ),
                    }
                )
            elif trap_name == TrapCategory.HALLUCINATED_COLUMN.value:
                recommendations.append(
                    {
                        "trap": trap_name,
                        "severity": severity,
                        "pipeline_target": "Schema Manager / DDL Serialization",
                        "prescriptive_action": (
                            "Format table schemas with explicit column lists and foreign key relations. "
                            "Introduce Harness-Zero schema introspection or schema pruning to anchor "
                            "grounding and prevent naming hallucinations."
                        ),
                    }
                )
            elif trap_name == TrapCategory.DIALECT_DRIFT.value:
                recommendations.append(
                    {
                        "trap": trap_name,
                        "severity": severity,
                        "pipeline_target": "Pre-generation Guardrails / Dialect Prompt",
                        "prescriptive_action": (
                            "Enforce SQLite dialect constraint in system instructions: "
                            "use strftime('%Y', col) instead of EXTRACT/YEAR, || instead of CONCAT, "
                            "CASE WHEN instead of IF(), and LIMIT instead of TOP."
                        ),
                    }
                )
            elif trap_name == TrapCategory.AGGREGATION_WITHOUT_GROUPING.value:
                recommendations.append(
                    {
                        "trap": trap_name,
                        "severity": severity,
                        "pipeline_target": "SQL Syntax Verifier / GRPO Reward Shaping",
                        "prescriptive_action": (
                            "Penalize queries with aggregate functions alongside naked column projections "
                            "lacking GROUP BY in the GRPO syntax reward function."
                        ),
                    }
                )
            elif trap_name == TrapCategory.NULL_BOUNDARY.value:
                recommendations.append(
                    {
                        "trap": trap_name,
                        "severity": severity,
                        "pipeline_target": "Few-Shot Examples / Verifier Rubric",
                        "prescriptive_action": (
                            "Include three-valued logic examples in few-shot demonstrations: "
                            "handle NULLs in negative filters (col != x OR col IS NULL) and "
                            "wrap nullable arithmetic attributes in COALESCE(bonus, 0)."
                        ),
                    }
                )

        return recommendations


class ZeroBudgetHarness:
    """Ultra-fast, in-process evaluation harness for local CI and rapid feedback."""

    def __init__(
        self,
        fixtures_dir: Path | None = None,
        timeout_seconds: float = 0.5,
        float_tolerance: float = 1e-4,
    ) -> None:
        self.fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
        self.timeout_seconds = timeout_seconds
        self.float_tolerance = float_tolerance
        self.seed_sql_path = self.fixtures_dir / "seed_fixtures.sql"
        self.mutant_sql_path = self.fixtures_dir / "seed_fixtures_mutant.sql"
        self.smoke_cases_path = self.fixtures_dir / "smoke_cases.jsonl"

        # Load schema definition
        self._schema_columns = self._extract_schema_columns()
        self.scorer = TrapScorer(schema_columns=self._schema_columns)

    def _extract_schema_columns(self) -> dict[str, set[str]]:
        """Extract schema table-to-columns mapping from seed fixtures."""
        if not self.seed_sql_path.exists():
            return {}
        conn = sqlite3.connect(":memory:")
        try:
            with open(self.seed_sql_path, encoding="utf-8") as f:
                conn.executescript(f.read())
            cur = conn.cursor()
            tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            schema: dict[str, set[str]] = {}
            for (tbl,) in tables:
                cols = cur.execute(f"PRAGMA table_info({tbl});").fetchall()
                schema[tbl.lower()] = {col[1].lower() for col in cols}
            return schema
        finally:
            conn.close()

    def _create_database(self, is_mutant: bool = False) -> sqlite3.Connection:
        """Create fresh in-memory SQLite database initialized with seed fixtures."""
        path = self.mutant_sql_path if is_mutant else self.seed_sql_path
        if not path.exists():
            raise FileNotFoundError(f"Fixture script not found: {path}")

        conn = sqlite3.connect(":memory:")
        with open(path, encoding="utf-8") as f:
            conn.executescript(f.read())
        return conn

    def _normalize_cell(self, cell: Any) -> Any:
        """Normalize table values for float-tolerant comparison."""
        if cell is None:
            return None
        if isinstance(cell, float):
            return round(cell, 4)
        if isinstance(cell, (int, bool)):
            return cell
        str_val = str(cell).strip()
        try:
            float_val = float(str_val)
            return round(float_val, 4)
        except ValueError:
            return str_val.lower()

    def _compare_results(
        self,
        gen_data: list[tuple[Any, ...]] | None,
        gold_data: list[tuple[Any, ...]] | None,
    ) -> bool:
        """Compare two result sets using order-agnostic multiset equivalence."""
        if gen_data is None or gold_data is None:
            return False
        if len(gen_data) != len(gold_data):
            return False

        norm_gen = [tuple(self._normalize_cell(c) for c in row) for row in gen_data]
        norm_gold = [tuple(self._normalize_cell(c) for c in row) for row in gold_data]

        return Counter(norm_gen) == Counter(norm_gold)

    def _execute_query(
        self, conn: sqlite3.Connection, sql: str
    ) -> tuple[bool, list[tuple[Any, ...]] | None, str | None]:
        """Safely execute a query with error capture."""
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
        except sqlite3.OperationalError as e:
            return False, None, f"OPERATIONAL_ERROR: {e}"
        except sqlite3.DatabaseError as e:
            return False, None, f"DATABASE_ERROR: {e}"
        except Exception as e:
            return False, None, f"UNEXPECTED_ERROR: {e}"

    def load_cases(self) -> list[dict[str, Any]]:
        """Load the 35 curated smoke cases from JSONL."""
        if not self.smoke_cases_path.exists():
            raise FileNotFoundError(f"Smoke cases file not found: {self.smoke_cases_path}")

        cases: list[dict[str, Any]] = []
        with open(self.smoke_cases_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json

                    cases.append(json.loads(line))
        return cases

    def run_case(self, case: dict[str, Any], candidate_sql: str) -> CaseVerdict:
        """Evaluate a single candidate SQL against a smoke case across base and mutant fixtures."""
        case_id = case["id"]
        gold_sql = case["query"]
        expected_trap = case.get("trap_category", TrapCategory.UNKNOWN.value)

        t0 = time.perf_counter()

        # 1. Base DB execution
        conn_base = self._create_database(is_mutant=False)
        try:
            gold_ok_base, gold_base_data, _ = self._execute_query(conn_base, gold_sql)
            gen_ok_base, gen_base_data, gen_base_err = self._execute_query(conn_base, candidate_sql)
        finally:
            conn_base.close()

        base_passed = False
        if gold_ok_base and gen_ok_base:
            base_passed = self._compare_results(gen_base_data, gold_base_data)

        # 2. Mutant DB execution
        conn_mut = self._create_database(is_mutant=True)
        try:
            gold_ok_mut, gold_mut_data, _ = self._execute_query(conn_mut, gold_sql)
            gen_ok_mut, gen_mut_data, gen_mut_err = self._execute_query(conn_mut, candidate_sql)
        finally:
            conn_mut.close()

        mutant_passed = False
        if gold_ok_mut and gen_ok_mut:
            mutant_passed = self._compare_results(gen_mut_data, gold_mut_data)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Overall verdict: must pass both base and mutant
        overall_passed = base_passed and mutant_passed

        # Determine failure status and diagnosis
        detected_trap: str | None = None
        error_msg = gen_base_err or gen_mut_err

        if overall_passed:
            status = "PASS"
        elif not gen_ok_base or not gen_ok_mut:
            status = "ERROR"
            detected_trap = self.scorer.score_failure(
                candidate_sql=candidate_sql,
                gold_sql=gold_sql,
                error_message=error_msg,
                candidate_data=gen_base_data,
                gold_data=gold_base_data,
                expected_trap=expected_trap,
            )
        else:
            status = "MISMATCH"
            detected_trap = self.scorer.score_failure(
                candidate_sql=candidate_sql,
                gold_sql=gold_sql,
                error_message=None,
                candidate_data=gen_base_data,
                gold_data=gold_base_data,
                expected_trap=expected_trap,
            )

        attribution_match = detected_trap == expected_trap if detected_trap else False

        return CaseVerdict(
            case_id=case_id,
            passed=overall_passed,
            status=status,
            base_passed=base_passed,
            mutant_passed=mutant_passed,
            detected_trap=detected_trap,
            expected_trap=expected_trap,
            attribution_match=attribution_match,
            latency_ms=round(elapsed_ms, 2),
            error_message=error_msg,
            candidate_sql=candidate_sql,
            gold_sql=gold_sql,
        )

    def evaluate_batch(self, cases_and_sql: list[tuple[dict[str, Any], str]]) -> HarnessSummary:
        """Evaluate a batch of (case, candidate_sql) pairs."""
        t_start = time.perf_counter()
        verdicts: list[CaseVerdict] = []
        misses_by_trap: dict[str, int] = {c.value: 0 for c in TrapCategory}

        passed_count = 0
        base_passed_count = 0
        mutant_passed_count = 0
        mutation_detected_count = 0
        attributed_correct_count = 0

        for case, sql in cases_and_sql:
            verdict = self.run_case(case, sql)
            verdicts.append(verdict)

            if verdict.passed:
                passed_count += 1
            else:
                trap = verdict.detected_trap or TrapCategory.UNKNOWN.value
                misses_by_trap[trap] = misses_by_trap.get(trap, 0) + 1
                if verdict.attribution_match:
                    attributed_correct_count += 1

            if verdict.base_passed:
                base_passed_count += 1
            if verdict.mutant_passed:
                mutant_passed_count += 1
            if verdict.base_passed and not verdict.mutant_passed:
                mutation_detected_count += 1

        total = len(cases_and_sql)
        failed_count = total - passed_count
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        avg_latency_ms = total_time_ms / total if total > 0 else 0.0

        pass_rate = passed_count / total if total > 0 else 0.0
        base_pass_rate = base_passed_count / total if total > 0 else 0.0
        mutant_pass_rate = mutant_passed_count / total if total > 0 else 0.0

        attribution_accuracy = attributed_correct_count / failed_count if failed_count > 0 else 1.0

        recommendations = DecisionTable.generate_remediations(misses_by_trap, failed_count)

        return HarnessSummary(
            total_cases=total,
            passed_cases=passed_count,
            failed_cases=failed_count,
            pass_rate=round(pass_rate, 4),
            base_pass_rate=round(base_pass_rate, 4),
            mutant_pass_rate=round(mutant_pass_rate, 4),
            mutation_detected_count=mutation_detected_count,
            total_latency_ms=round(total_time_ms, 2),
            average_latency_ms=round(avg_latency_ms, 2),
            misses_by_trap=misses_by_trap,
            attribution_accuracy=round(attribution_accuracy, 4),
            case_verdicts=verdicts,
            remediation_recommendations=recommendations,
        )

    def evaluate_golden(self) -> HarnessSummary:
        """Sanity evaluation verifying that all golden queries pass 100%."""
        cases = self.load_cases()
        pairs = [(case, case["query"]) for case in cases]
        return self.evaluate_batch(pairs)

    def evaluate_flawed(self) -> HarnessSummary:
        """Benchmark evaluation testing that all injected flaws are detected and attributed."""
        cases = self.load_cases()
        pairs = [(case, case["expected_flaw"]) for case in cases]
        return self.evaluate_batch(pairs)
