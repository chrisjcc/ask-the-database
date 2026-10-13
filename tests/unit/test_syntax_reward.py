"""Unit tests for SQLSyntaxValidator AST parsing and validation."""

from ask_the_database.rewards.syntax_reward import SQLSyntaxValidator


def test_valid_select_query() -> None:
    """Test standard SELECT query validates and extracts tables and columns."""
    validator = SQLSyntaxValidator()
    result = validator.validate("SELECT name, age FROM singer WHERE age > 30;")

    assert result.is_valid is True
    assert result.error_type is None
    assert "singer" in result.referenced_tables
    assert "name" in result.referenced_columns
    assert "age" in result.referenced_columns
    assert result.has_aggregation is False
    assert result.has_group_by is False
    assert result.has_order_by is False


def test_malformed_syntax_error() -> None:
    """Test syntax error detection on broken SQL."""
    validator = SQLSyntaxValidator()
    result = validator.validate("SELECT name FROM WHERE age > ;")

    assert result.is_valid is False
    assert result.error_type == "PARSE_ERROR"
    assert result.error_message is not None


def test_non_select_statement() -> None:
    """Test non-SELECT statements are rejected with NON_SELECT."""
    validator = SQLSyntaxValidator()
    result = validator.validate("UPDATE singer SET age = 20 WHERE name = 'Alice';")

    assert result.is_valid is False
    assert result.error_type == "NON_SELECT"
    assert "Root statement must be a SELECT" in str(result.error_message)


def test_empty_query_string() -> None:
    """Test empty and whitespace-only queries."""
    validator = SQLSyntaxValidator()
    res1 = validator.validate("")
    assert res1.is_valid is False
    assert res1.error_type == "EMPTY"

    res2 = validator.validate("   ;   ")
    assert res2.is_valid is False
    assert res2.error_type == "EMPTY"


def test_aggregation_and_clauses_detection() -> None:
    """Test detection of aggregation, GROUP BY, and ORDER BY clauses."""
    validator = SQLSyntaxValidator()
    query = """
    SELECT department_id, COUNT(user_id) AS total_users, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department_id
    ORDER BY avg_sal DESC;
    """
    result = validator.validate(query)

    assert result.is_valid is True
    assert result.has_aggregation is True
    assert result.has_group_by is True
    assert result.has_order_by is True
    assert "employees" in result.referenced_tables


def test_join_multi_table_extraction() -> None:
    """Test extracting multiple tables from complex JOINs."""
    validator = SQLSyntaxValidator()
    query = """
    SELECT u.name, d.department_name
    FROM users u
    JOIN departments d ON u.department_id = d.id
    WHERE d.budget > 10000;
    """
    result = validator.validate(query)

    assert result.is_valid is True
    assert "users" in result.referenced_tables
    assert "departments" in result.referenced_tables


def test_set_operations_validity() -> None:
    """Test UNION and INTERSECT queries are valid read queries."""
    validator = SQLSyntaxValidator()
    union_query = "SELECT id FROM table_a UNION SELECT id FROM table_b;"
    result = validator.validate(union_query)

    assert result.is_valid is True
    assert "table_a" in result.referenced_tables
    assert "table_b" in result.referenced_tables
