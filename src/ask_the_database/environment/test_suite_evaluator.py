"""Test-suite database mutation and semantic equivalence evaluator."""

import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ask_the_database.environment.sqlite_env import SQLiteEnvironment

if TYPE_CHECKING:
    pass


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

    __test__ = False

    def __init__(self, num_mutations: int = 4, comparator: Any = None) -> None:
        self.num_mutations = num_mutations
        if comparator is None:
            from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator

            self.comparator = ExecutionEquivalenceEvaluator()
        else:
            self.comparator = comparator

    def _create_mutated_clone(self, base_db_path: str, mutation_type: int) -> str:
        """Create an ephemeral SQLite database clone with specific data mutations."""
        temp_dir = tempfile.mkdtemp(prefix="sqlite_mutation_")
        clone_path = os.path.join(temp_dir, "mutated.sqlite")
        shutil.copy2(base_db_path, clone_path)

        conn = sqlite3.connect(clone_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF;")

        try:
            # Get all user tables
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                col_info = cursor.fetchall()
                if not col_info:
                    continue

                if mutation_type == 1:
                    # 1. Random row drop: delete ~30% of rows
                    cursor.execute(f"DELETE FROM {table} WHERE (ROWID % 3) = 0;")

                elif mutation_type == 2:
                    # 2. Duplicate value mutation to catch missing DISTINCT
                    # Check for single integer primary key
                    pk_cols = [c for c in col_info if c[5]]
                    non_pk_cols = [c[1] for c in col_info if not c[5]]

                    if len(pk_cols) == 1 and "INT" in pk_cols[0][2].upper() and non_pk_cols:
                        pk_name = pk_cols[0][1]
                        col_str = ", ".join(non_pk_cols)
                        cursor.execute(
                            f"INSERT INTO {table} ({pk_name}, {col_str}) "
                            f"SELECT (SELECT COALESCE(MAX({pk_name}), 0) + 1 FROM {table}), {col_str} "
                            f"FROM {table} LIMIT 1;"
                        )
                    else:
                        cursor.execute(f"INSERT INTO {table} SELECT * FROM {table} LIMIT 1;")

                elif mutation_type == 3:
                    # 3. Numeric boundary shift: shift first numeric column by +1
                    for col in col_info:
                        col_name, col_type = col[1], col[2].upper()
                        if "INT" in col_type or "REAL" in col_type or "NUM" in col_type:
                            cursor.execute(
                                f"UPDATE {table} SET {col_name} = {col_name} + 1 "
                                f"WHERE {col_name} IS NOT NULL;"
                            )
                            break

                elif mutation_type == 4:
                    # 4. Null injection: inject NULL into nullable columns
                    nullable_cols = [c[1] for c in col_info if not c[3] and not c[5]]
                    for n_col in nullable_cols:
                        cursor.execute(f"UPDATE {table} SET {n_col} = NULL WHERE (ROWID % 2) = 0;")

            conn.commit()
        except sqlite3.Error:
            pass  # If a specific mutation violates table structure, keep current state
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
