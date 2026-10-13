"""SQL syntax parsing and AST validation using sqlglot."""

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
        except Exception as e:
            return SyntaxValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message=f"AST parsing exception: {e}",
            )

        valid_exprs = [e for e in expressions if e is not None]
        if not valid_exprs:
            return SyntaxValidationResult(
                is_valid=False,
                error_type="PARSE_ERROR",
                error_message="Could not generate valid AST expression.",
            )

        ast = valid_exprs[0]

        # Enforce that query is a SELECT, set operation, or CTE with SELECT
        if not isinstance(
            ast, (exp.Query, exp.Select, exp.Union, exp.Intersect, exp.Except)
        ) and not (isinstance(ast, exp.Expression) and ast.find(exp.Select)):
            return SyntaxValidationResult(
                is_valid=False,
                error_type="NON_SELECT",
                error_message=f"Root statement must be a SELECT, found: {type(ast).__name__}",
            )

        # Extract table names
        tables = {table.name.lower() for table in ast.find_all(exp.Table) if table.name}

        # Extract column names
        columns = {col.name.lower() for col in ast.find_all(exp.Column) if col.name}

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
