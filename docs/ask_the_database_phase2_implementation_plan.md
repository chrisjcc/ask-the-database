# AskTheDatabase — Phase 2 Granular Implementation Plan
## Verifiable Reward Engine & Test-Suite Verification (Issues #6 to #10)

---

## 1. Phase 2 Overview & Scientific Scope

The core objective of **Phase 2** is to construct the framework-independent **Verifiable Reward Engine**. In natural language-to-SQL translation, reward design is substantially more critical than the RL algorithm itself. 

A naive binary reward ($1.0$ if the query returns the gold table, $0.0$ otherwise) produces an excessively sparse reward landscape where policy gradient exploration stalls. Conversely, unconstrained heuristic shaping encourages **reward hacking** (e.g., generating syntactically bloated queries that match trivial schema keywords without answering the prompt). Furthermore, small benchmark tables frequently suffer from **spurious execution equivalence** ("getting the right answer for the wrong reason").

Phase 2 establishes:
1. **AST-Level Syntax Validation**: Robust AST parsing, dialect normalization, and error classification via `sqlglot`.
2. **Execution Equivalence (EX)**: Order-agnostic multiset (bag) matching, float tolerance ($\epsilon = 10^{-4}$), and NULL semantics.
3. **Test-Suite (TS) Database Mutation**: Multi-state database perturbation to mathematically eliminate spurious false positives.
4. **Hierarchical Bounded Rubric**: A composite scoring engine where partial credit validity signals can **never** overshadow correctness.
5. **Comprehensive Unit Testing**: Hermetic edge-case validation covering empty sets, float tolerances, division by zero, and syntax permutations.

---

## 2. Directory Layout & Target File Inventory

```text
ask-the-database/
├── configs/
│   └── reward/
│       ├── rubric_default.yaml       # Default reward weights and penalties
│       └── rubric_test_suite.yaml   # Test-suite verified reward weights
│
├── src/
│   └── ask_the_database/
│       ├── environment/
│       │   └── test_suite_evaluator.py  # Issue #8: DB state mutation engine
│       └── rewards/
│           ├── __init__.py
│           ├── base.py                  # BaseReward and DiagnosticPayload
│           ├── syntax_reward.py         # Issue #6: AST validator & linter
│           ├── execution_reward.py      # Issue #7: Multiset table comparator
│           └── rubric.py                # Issue #9: Hierarchical reward composite
│
└── tests/
    └── unit/
        ├── test_syntax_reward.py        # Issue #6 testing
        ├── test_execution_reward.py     # Issue #7 testing
        ├── test_test_suite_mutation.py  # Issue #8 testing
        └── test_rubric.py               # Issue #9 & #10 composite testing
```

---

## 3. Granular Issue-by-Issue Implementation Specifications

---

### Issue #6: SQL Syntax & AST Validator Implementation

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/rewards/syntax_reward.py`.
- Parse candidate SQL strings using `sqlglot` with the `"sqlite"` read dialect.
- Extract structural diagnostics without running the query against a database:
  - Detect syntax errors, unclosed parentheses, missing commas, and invalid keywords.
  - Verify that the query has a valid projection (`SELECT`) and valid source table reference (`FROM`).
  - Extract all referenced table identifiers and column identifiers for schema linking.
- Classify syntax errors into structured types (`PARSE_ERROR`, `EMPTY_QUERY`, `FORBIDDEN_KEYWORD`, `VALID_SYNTAX`).

#### 2. Exact File Contracts

##### `src/ask_the_database/rewards/syntax_reward.py`
```python
from dataclasses import dataclass, field
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


@dataclass(frozen=True)
class SyntaxValidationResult:
    """Diagnostic outcome of AST syntax validation."""
    is_valid: bool
    error_type: str | None  # "EMPTY", "PARSE_ERROR", "NON_SELECT", None
    error_message: str | None
    referenced_tables: frozenset[str] = field(default_factory=frozenset)
    referenced_columns: frozenset[str] = field(default_factory=frozenset)
    has_aggregation: bool = False
    has_group_by: bool = False
    has_order_by: bool = False


class SQLSyntaxValidator:
    """Fast, database-free AST parser and syntax validator."""

    def __init__(self, dialect: str = "sqlite") -> None:
        self.dialect = dialect

    def validate(self, sql_query: str) -> SyntaxValidationResult:
        """Parse query string and extract structural AST features."""
        clean_query = sql_query.strip().rstrip(";")
        if not clean_query:
            return SyntaxValidationResult(
                is_valid=False,
                error_type="EMPTY",
                error_message="Query string is empty.",
            )

        try:
            expressions = sqlglot.parse(clean_query, read=self.dialect)
        except ParseError as e:
            return SyntaxValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message=str(e),
            )

        if not expressions or expressions[0] is None:
            return SyntaxValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message="Could not generate valid AST expression.",
            )

        ast = expressions[0]

        # Enforce that query is a SELECT or WITH statement
        if not isinstance(ast, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            if not (isinstance(ast, exp.Expression) and ast.find(exp.Select)):
                return SyntaxValidationResult(
                    is_valid=False,
                    error_type="NON_SELECT",
                    error_message=f"Root statement must be a SELECT, found: {type(ast).__name__}",
                )

        # Extract table names
        tables = {
            table.name.lower()
            for table in ast.find_all(exp.Table)
            if table.name
        }

        # Extract column names
        columns = {
            col.name.lower()
            for col in ast.find_all(exp.Column)
            if col.name
        }

        has_agg = bool(list(ast.find_all(exp.AggFunc)))
        has_group = bool(ast.find(exp.Group))
        has_order = bool(ast.find(exp.Order))

        return SyntaxValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            referenced_tables=frozenset(tables),
            referenced_columns=frozenset(columns),
            has_aggregation=has_agg,
            has_group_by=has_group,
            has_order_by=has_order,
        )
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_syntax_reward.py`:
- Valid `SELECT name FROM singer WHERE age > 30` returns `is_valid=True`, `referenced_tables={"singer"}`.
- Malformed `SELECT name FROM WHERE age >` returns `is_valid=False`, `error_type="PARSE_ERROR"`.
- Non-select `UPDATE singer SET age = 20` returns `is_valid=False`, `error_type="NON_SELECT"`.

---

### Issue #7: Execution Equivalence Evaluator (EX)

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/rewards/execution_reward.py`.
- Execute candidate query $Q_{\text{gen}}$ and gold standard query $Q_{\text{gold}}$ against the sandboxed target database state.
- Compare output relation tables using **multiset (bag) equivalence**:
  - If the gold query explicitly specifies an `ORDER BY`, enforce **strict sequence equality**.
  - If no `ORDER BY` is present, enforce **order-agnostic multiset equality** (rows can appear in any order, but duplicate rows must match exact frequencies).
- Handle robust data-type normalization:
  - Round floating point numbers to 4 decimal places ($\epsilon = 10^{-4}$).
  - Normalize integer vs string numeric representations (e.g., `"1"` vs `1`).
  - Normalize case-insensitive string values if dialect requires.
  - Distinguish between a genuine empty result (`[]`) and an execution error (`None`).

#### 2. Exact File Contracts

##### `src/ask_the_database/rewards/execution_reward.py`
```python
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
    error_reason: str | None  # "GEN_EXEC_ERROR", "ROW_COUNT_MISMATCH", "CONTENT_MISMATCH", None


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
        if isinstance(cell, (int, bool)):
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
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_execution_reward.py`:
- `SELECT a, b FROM t` vs `SELECT a, b FROM t` $\rightarrow$ `is_equivalent=True`.
- Shuffled rows without `ORDER BY` $\rightarrow$ `is_equivalent=True`.
- Float values `3.14159` vs `3.1416` $\rightarrow$ `is_equivalent=True`.
- Shuffled rows with `ORDER BY` enforced $\rightarrow$ `is_equivalent=False`.

---

### Issue #8: Test-Suite (TS) Database Mutation Engine

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/environment/test_suite_evaluator.py`.
- Solve **spurious execution equivalence**: On small databases, queries that select the wrong column or have flawed `WHERE` clauses often match the reference result by chance.
- Generate $K$ (default: $K=5$) perturbed ephemeral database clones for each test database:
  1. **Random Row Drop Mutation**: Deletes $30\%$ of non-foreign-key rows to test aggregation stability.
  2. **Null Injection Mutation**: Inserts `NULL` values into nullable columns to test null-handling logic (`IS NULL`, `COALESCE`).
  3. **Duplicate Value Mutation**: Duplicates specific rows to catch queries lacking necessary `DISTINCT` keywords.
  4. **Boundary Inversion Mutation**: Mutates integer/date values $\pm 1$ around threshold filters.
- Evaluate candidate queries across all $K$ mutated database clones:
  $$\text{TS\_Match} = \prod_{k=1}^{K} \mathbb{I}\left(Q_{\text{gen}}(\mathcal{D}_k) \equiv Q_{\text{gold}}(\mathcal{D}_k)\right)$$

#### 2. Exact File Contracts

##### `src/ask_the_database/environment/test_suite_evaluator.py`
```python
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator


@dataclass(frozen=True)
class TestSuiteEvaluationResult:
    """Evaluation result across multiple mutated database states."""
    passed_all_suites: bool
    suites_evaluated: int
    suites_passed: int
    failed_suite_index: int | None
    failure_reason: str | None


class TestSuiteEvaluator:
    """Generates perturbed SQLite clones and verifies query equivalence across states."""

    def __init__(self, num_mutations: int = 4) -> None:
        self.num_mutations = num_mutations
        self.comparator = ExecutionEquivalenceEvaluator()

    def _create_mutated_clone(self, base_db_path: str, mutation_type: int) -> str:
        """Create an ephemeral SQLite database clone with specific data mutations."""
        temp_dir = tempfile.mkdtemp(prefix="sqlite_mutation_")
        clone_path = os.path.join(temp_dir, "mutated.sqlite")
        shutil.copy2(base_db_path, clone_path)

        conn = sqlite3.connect(clone_path)
        cursor = conn.cursor()

        try:
            # Get all user tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                if mutation_type == 1:
                    # Drop 30% of rows randomly
                    cursor.execute(f"DELETE FROM {table} WHERE (ROWID % 3) = 0;")
                elif mutation_type == 2:
                    # Duplicate first row to break queries missing DISTINCT
                    cursor.execute(f"INSERT INTO {table} SELECT * FROM {table} LIMIT 1;")
                elif mutation_type == 3:
                    # Shift numeric values by +1 to break tight inequality filters
                    cursor.execute(f"PRAGMA table_info({table});")
                    cols = cursor.fetchall()
                    for col in cols:
                        col_name, col_type = col[1], col[2].upper()
                        if "INT" in col_type or "NUM" in col_type:
                            cursor.execute(f"UPDATE {table} SET {col_name} = {col_name} + 1 WHERE {col_name} IS NOT NULL;")
                            break
            conn.commit()
        except sqlite3.Error:
            pass  # Mutation failed due to constraints; clone remains valid
        finally:
            conn.close()

        return clone_path

    def evaluate_test_suite(
        self,
        base_db_path: str,
        gen_sql: str,
        gold_sql: str,
        requires_strict_order: bool = False,
    ) -> TestSuiteEvaluationResult:
        """Evaluate candidate query against base DB plus mutated clones."""
        # 1. Base DB check
        base_env = SQLiteEnvironment(base_db_path)
        res_gold = base_env.execute(gold_sql)
        res_gen = base_env.execute(gen_sql)

        base_match = self.comparator.compare(res_gen, res_gold, requires_strict_order)
        if not base_match.is_equivalent:
            return TestSuiteEvaluationResult(
                passed_all_suites=False,
                suites_evaluated=1,
                suites_passed=0,
                failed_suite_index=0,
                failure_reason=base_match.error_reason,
            )

        passed_count = 1
        # 2. Mutated DB checks
        for idx in range(1, self.num_mutations + 1):
            clone_path = self._create_mutated_clone(base_db_path, mutation_type=idx)
            try:
                clone_env = SQLiteEnvironment(clone_path)
                m_gold = clone_env.execute(gold_sql)
                m_gen = clone_env.execute(gen_sql)

                match = self.comparator.compare(m_gen, m_gold, requires_strict_order)
                if not match.is_equivalent:
                    return TestSuiteEvaluationResult(
                        passed_all_suites=False,
                        suites_evaluated=passed_count + 1,
                        suites_passed=passed_count,
                        failed_suite_index=idx,
                        failure_reason="MUTATED_STATE_MISMATCH",
                    )
                passed_count += 1
            finally:
                # Clean up ephemeral directory
                shutil.rmtree(Path(clone_path).parent, ignore_errors=True)

        return TestSuiteEvaluationResult(
            passed_all_suites=True,
            suites_evaluated=passed_count,
            suites_passed=passed_count,
            failed_suite_index=None,
            failure_reason=None,
        )
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_test_suite_mutation.py`:
- Two identical queries pass 100% of mutations.
- A query with a missing `DISTINCT` keyword that happens to match the base DB fails on Mutation #2 (duplicate row injection).
- Ephemeral mutation files are completely removed from disk after evaluation.

---

### Issue #9: Hierarchical Reward Rubric Composite

#### 1. Objectives & Responsibilities
- Implement `src/ask_the_database/rewards/rubric.py`.
- Formulate the composite reward score $\mathcal{R}_{\text{total}} \in [-1.0, 1.0]$:
  $$\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{syntax}} + \mathcal{R}_{\text{schema}} + \mathcal{R}_{\text{exec}} + \mathcal{R}_{\text{TS}}$$
- Guarantee the **Hierarchical Bounding Invariant**: A query with an execution mismatch can **never** receive a net positive reward ($> 0.0$), even if its syntax and schema linking are flawless.
- Output a comprehensive `DiagnosticPayload` dictionary providing fine-grained telemetry for logging to Weights & Biases / Weave.

#### 2. Exact File Contracts

##### `src/ask_the_database/rewards/rubric.py`
```python
from dataclasses import asdict, dataclass
from typing import Any
from omegaconf import DictConfig
from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.environment.test_suite_evaluator import TestSuiteEvaluator
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator
from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator


@dataclass(frozen=True)
class RewardOutput:
    """Consolidated reward output and telemetry payload."""
    total_reward: float
    is_valid_syntax: bool
    is_valid_execution: bool
    is_execution_equivalent: bool
    is_test_suite_equivalent: bool
    diagnostic_payload: dict[str, Any]


class SQLRewardRubric:
    """Framework-independent composite reward evaluator for SQL RLVR."""

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        self.syntax_validator = SQLSyntaxValidator()
        self.execution_evaluator = ExecutionEquivalenceEvaluator(
            float_tolerance=self.cfg.get("float_tolerance", 1e-4)
        )
        self.use_test_suite = self.cfg.get("use_test_suite", True)
        self.test_suite_evaluator = TestSuiteEvaluator(
            num_mutations=self.cfg.get("num_mutations", 4)
        ) if self.use_test_suite else None

        # Weights
        self.w_syntax_penalty = self.cfg.weights.syntax_penalty    # e.g., -1.0
        self.w_exec_penalty = self.cfg.weights.exec_penalty        # e.g., -0.5
        self.w_schema_bonus = self.cfg.weights.schema_bonus        # e.g., +0.2
        self.w_base_exec = self.cfg.weights.base_execution_reward  # e.g., +0.5
        self.w_ts_exec = self.cfg.weights.test_suite_reward        # e.g., +0.5

    def evaluate(
        self,
        candidate_sql: str,
        gold_sql: str,
        db_path: str,
        requires_strict_order: bool = False,
    ) -> RewardOutput:
        """Calculate the total hierarchical reward and diagnostic telemetry."""
        # 1. Tier 1: Syntax Validation
        syntax_res = self.syntax_validator.validate(candidate_sql)
        if not syntax_res.is_valid:
            return RewardOutput(
                total_reward=self.w_syntax_penalty,
                is_valid_syntax=False,
                is_valid_execution=False,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "syntax_error",
                    "error_type": syntax_res.error_type,
                    "error_message": syntax_res.error_message,
                },
            )

        # 2. Tier 2: Execution Feasibility
        env = SQLiteEnvironment(db_path)
        gen_exec = env.execute(candidate_sql)
        gold_exec = env.execute(gold_sql)

        if not gen_exec.success:
            return RewardOutput(
                total_reward=self.w_exec_penalty,
                is_valid_syntax=True,
                is_valid_execution=False,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "execution_error",
                    "error_type": gen_exec.error_type,
                    "error_message": gen_exec.error_message,
                },
            )

        # 3. Tier 3: Execution Equivalence on Base Database
        exec_cmp = self.execution_evaluator.compare(gen_exec, gold_exec, requires_strict_order)
        if not exec_cmp.is_equivalent:
            # Bounded: execution failed, small schema bonus awarded but total stays <= 0.0
            net_score = min(0.0, self.w_exec_penalty + self.w_schema_bonus)
            return RewardOutput(
                total_reward=net_score,
                is_valid_syntax=True,
                is_valid_execution=True,
                is_execution_equivalent=False,
                is_test_suite_equivalent=False,
                diagnostic_payload={
                    "stage": "semantic_mismatch",
                    "reason": exec_cmp.error_reason,
                    "gold_rows": exec_cmp.gold_row_count,
                    "gen_rows": exec_cmp.gen_row_count,
                },
            )

        # 4. Tier 4: Test-Suite Verification
        if self.use_test_suite and self.test_suite_evaluator is not None:
            ts_res = self.test_suite_evaluator.evaluate_test_suite(
                db_path, candidate_sql, gold_sql, requires_strict_order
            )
            if not ts_res.passed_all_suites:
                # Passed base DB but failed mutated states: spurious false positive!
                return RewardOutput(
                    total_reward=self.w_base_exec,  # Partial reward (+0.5)
                    is_valid_syntax=True,
                    is_valid_execution=True,
                    is_execution_equivalent=True,
                    is_test_suite_equivalent=False,
                    diagnostic_payload={
                        "stage": "test_suite_failure",
                        "suites_passed": ts_res.suites_passed,
                        "failed_index": ts_res.failed_suite_index,
                    },
                )

        # Passed all verification tiers: Full Reward (+1.0)
        return RewardOutput(
            total_reward=self.w_base_exec + self.w_ts_exec,
            is_valid_syntax=True,
            is_valid_execution=True,
            is_execution_equivalent=True,
            is_test_suite_equivalent=True,
            diagnostic_payload={
                "stage": "full_correctness",
                "gold_rows": exec_cmp.gold_row_count,
            },
        )
```

##### Hydra Configuration: `configs/reward/rubric_default.yaml`
```yaml
float_tolerance: 1e-4
use_test_suite: true
num_mutations: 4

weights:
  syntax_penalty: -1.0
  exec_penalty: -0.5
  schema_bonus: 0.2
  base_execution_reward: 0.5
  test_suite_reward: 0.5
```

#### 3. Definition of Done & Verification Gate
Execute `pytest tests/unit/test_rubric.py`:
- Malformed syntax yields $-1.0$.
- Runtime DB error yields $-0.5$.
- Syntactically valid query with wrong result yields $\le 0.0$ (no false positive reward hacking).
- Base DB match + mutated test-suite match yields $+1.0$.

---

### Issue #10: Unit Testing for Reward Rubric Edge Cases

#### 1. Objectives & Responsibilities
- Create `tests/unit/test_rewards.py` verifying extreme database and query edge cases:
  1. **Empty Result Sets**: Both queries returning 0 rows (valid query answering empty table) vs mismatch.
  2. **Division by Zero**: Query containing `salary / 0` returning runtime SQLite error.
  3. **Duplicate Rows**: Catching queries with missing `DISTINCT` via multiset frequency comparison.
  4. **Float Precision & Rounding**: Verifying that `3.14159265` matches `3.1416` under $\epsilon = 1e-4$.
  5. **Column Alias Differences**: `SELECT age AS user_age` vs `SELECT age FROM users` (must be equivalent).
  6. **NULL vs Empty String**: Verifying that `NULL` is never treated as equal to `""`.

#### 2. Exact Test Matrix

```python
def test_float_epsilon_matching(rubric, test_db):
    gen = "SELECT 1.00004;"
    gold = "SELECT 1.00001;"
    out = rubric.evaluate(gen, gold, test_db)
    assert out.is_execution_equivalent is True
    assert out.total_reward > 0.0

def test_division_by_zero_penalty(rubric, test_db):
    gen = "SELECT 1 / 0 FROM users;"
    gold = "SELECT 1 FROM users;"
    out = rubric.evaluate(gen, gold, test_db)
    assert out.is_valid_execution is False
    assert out.total_reward == -0.5

def test_empty_set_semantic_match(rubric, test_db):
    gen = "SELECT name FROM users WHERE age > 9999;"
    gold = "SELECT name FROM users WHERE age > 9999;"
    out = rubric.evaluate(gen, gold, test_db)
    assert out.is_execution_equivalent is True
    assert out.total_reward == 1.0

def test_alias_invariance(rubric, test_db):
    gen = "SELECT age AS my_age, name AS my_name FROM users;"
    gold = "SELECT age, name FROM users;"
    out = rubric.evaluate(gen, gold, test_db)
    assert out.is_execution_equivalent is True
    assert out.total_reward == 1.0
```

#### 3. Definition of Done & Pass/Fail Verification Gate
```bash
pytest tests/unit/test_rewards.py -v
```
*Pass Criteria*: 100% passing across all edge case tests with zero unhandled exceptions.

---

## 4. Phase 2 Summary & Transition to Phase 3

With Phase 2 complete, the scientific core of the project is fully independent of any training infrastructure:
- Any SQL candidate can be parsed, executed in isolation, tested against multiple mutated database states, and awarded a mathematically grounded reward in $[-1.0, 1.0]$.
- **Phase 3** will build directly on top of this by implementing the **SFT Warmup Training Pipeline** and the **Framework Adapter Layer** to begin feeding these rewards into language model policy updates.
