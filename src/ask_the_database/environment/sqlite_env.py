"""Secure, isolated, read-only SQLite sandbox environment."""

import concurrent.futures
import contextlib
import os
import sqlite3
import time
from typing import Any

import sqlglot
from sqlglot import exp

from ask_the_database.environment.base import BaseSQLEnvironment, ExecutionResult
from ask_the_database.environment.schema_manager import SchemaManager


class SQLSecurityViolationError(Exception):
    """Raised when a query attempts state mutation or violates sandbox safety rules."""

    pass


class SQLTimeoutError(Exception):
    """Raised when a query exceeds the execution time limit."""

    pass


class SQLiteEnvironment(BaseSQLEnvironment):
    """Hermetic SQLite execution environment enforcing read-only safety, AST verification,

    and strict execution timeouts.
    """

    # Forbidden AST node types that indicate DDL, DML, or administrative commands
    FORBIDDEN_EXPRESSION_TYPES: tuple[type[exp.Expression], ...] = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
        exp.Pragma,
        exp.TruncateTable,
        exp.Set,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
    )

    def __init__(
        self,
        db_path: str,
        timeout_seconds: float = 3.0,
        max_result_rows: int = 500,
        enforce_ast_safety: bool = True,
        forbidden_tokens: list[str] | None = None,
    ) -> None:
        self.db_path = os.path.abspath(db_path)
        self.timeout_seconds = timeout_seconds
        self.max_result_rows = max_result_rows
        self.enforce_ast_safety = enforce_ast_safety
        self.forbidden_tokens = [
            t.upper()
            for t in (
                forbidden_tokens
                or [
                    "DROP",
                    "DELETE",
                    "INSERT",
                    "UPDATE",
                    "ALTER",
                    "CREATE",
                    "ATTACH",
                    "DETACH",
                    "REINDEX",
                    "VACUUM",
                ]
            )
        ]
        self._schema_manager: SchemaManager | None = None

    @property
    def schema_manager(self) -> SchemaManager:
        """Lazily initialize schema manager."""
        if self._schema_manager is None:
            self._schema_manager = SchemaManager(self.db_path)
        return self._schema_manager

    def _verify_ast_safety(self, query: str) -> None:
        """Parse query using sqlglot and verify read-only safety.

        Raises:
            SQLSecurityViolationError: If mutation or stacked queries are detected.
            sqlglot.errors.ParseError: If query syntax is invalid.
        """
        trimmed = query.strip().rstrip(";").strip()
        if not trimmed:
            raise SQLSecurityViolationError("Query is empty.")

        try:
            expressions = sqlglot.parse(query, read="sqlite")
        except Exception as e:
            # Re-raise to be handled by caller as syntax or security error
            raise e

        # Filter out empty None expressions if any
        expressions = [e for e in expressions if e is not None]

        if not expressions:
            raise SQLSecurityViolationError("No executable expressions parsed.")

        if len(expressions) > 1:
            raise SQLSecurityViolationError(
                f"Stacked queries detected ({len(expressions)} statements). Only single queries allowed."
            )

        expr = expressions[0]

        # Verify root statement is a read-only query
        if not isinstance(expr, (exp.Select, exp.Union)):
            raise SQLSecurityViolationError(
                f"Prohibited SQL statement type: {type(expr).__name__}. Only SELECT queries are permitted."
            )

        # Walk AST to detect any nested mutation nodes
        for node in expr.walk():
            if isinstance(node, self.FORBIDDEN_EXPRESSION_TYPES):
                raise SQLSecurityViolationError(
                    f"Prohibited SQL operation inside query: {type(node).__name__}."
                )

    def _open_read_only_connection(self) -> sqlite3.Connection:
        """Create a connection in URI read-only mode with query_only pragma."""
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def execute(self, query: str) -> ExecutionResult:
        """Execute a SQL query inside the sandboxed environment with bounds."""
        start_time = time.perf_counter()

        # 1. AST Safety Verification
        if self.enforce_ast_safety:
            try:
                self._verify_ast_safety(query)
            except SQLSecurityViolationError as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    success=False,
                    data=None,
                    columns=None,
                    row_count=0,
                    execution_time_ms=elapsed_ms,
                    error_message=str(e),
                    error_type="SECURITY",
                )
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    success=False,
                    data=None,
                    columns=None,
                    row_count=0,
                    execution_time_ms=elapsed_ms,
                    error_message=f"SQL syntax parse error: {e}",
                    error_type="SYNTAX",
                )

        if not os.path.exists(self.db_path):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                data=None,
                columns=None,
                row_count=0,
                execution_time_ms=elapsed_ms,
                error_message=f"Database file not found: {self.db_path}",
                error_type="EXECUTION",
            )

        # 2. Execute Query with Concurrency and Timeout
        conn: sqlite3.Connection | None = None
        try:
            conn = self._open_read_only_connection()

            def run_worker() -> tuple[list[tuple[Any, ...]], list[str]]:
                assert conn is not None
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchmany(self.max_result_rows)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return rows, columns

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_worker)
                try:
                    rows, columns = future.result(timeout=self.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    if conn:
                        conn.interrupt()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    return ExecutionResult(
                        success=False,
                        data=None,
                        columns=None,
                        row_count=0,
                        execution_time_ms=elapsed_ms,
                        error_message=f"Query exceeded timeout limit of {self.timeout_seconds:.1f}s",
                        error_type="TIMEOUT",
                    )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=True,
                data=rows,
                columns=columns,
                row_count=len(rows),
                execution_time_ms=elapsed_ms,
                error_message=None,
                error_type=None,
            )

        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            err_str = str(e)
            if "interrupted" in err_str.lower():
                err_type = "TIMEOUT"
            elif (
                "attempt to write a readonly database" in err_str.lower()
                or "query_only" in err_str.lower()
            ):
                err_type = "SECURITY"
            else:
                err_type = "EXECUTION"
            return ExecutionResult(
                success=False,
                data=None,
                columns=None,
                row_count=0,
                execution_time_ms=elapsed_ms,
                error_message=err_str,
                error_type=err_type,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                data=None,
                columns=None,
                row_count=0,
                execution_time_ms=elapsed_ms,
                error_message=str(e),
                error_type="EXECUTION",
            )
        finally:
            if conn:
                with contextlib.suppress(Exception):
                    conn.close()

    def get_schema_summary(self) -> str:
        """Return formatted schema summary using SchemaManager."""
        return self.schema_manager.get_schema_summary()
