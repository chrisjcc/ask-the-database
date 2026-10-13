"""Unit tests verifying SQLiteEnvironment security, timeouts, and sandboxing safety."""

import sqlite3
from pathlib import Path

from ask_the_database.environment.sqlite_env import SQLiteEnvironment


def test_valid_select_query(sample_sqlite_db: Path) -> None:
    """Test simple SELECT query execution returns expected rows."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("SELECT name, age FROM users WHERE age > 21 ORDER BY age;")

    assert result.success is True
    assert result.data is not None
    assert len(result.data) == 4
    assert result.columns == ["name", "age"]
    assert result.data[0] == ("Alice", 25)
    assert result.error_type is None
    assert result.error_message is None


def test_valid_join_aggregation(sample_sqlite_db: Path) -> None:
    """Test multi-table JOIN with GROUP BY and HAVING."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    query = """
    SELECT d.department_name, COUNT(u.user_id) AS emp_count
    FROM departments d
    JOIN users u ON d.department_id = u.department_id
    GROUP BY d.department_name
    HAVING COUNT(u.user_id) >= 1
    ORDER BY d.department_name;
    """
    result = env.execute(query)

    assert result.success is True
    assert result.data is not None
    assert len(result.data) == 3
    assert result.columns == ["department_name", "emp_count"]
    assert ("Engineering", 2) in result.data


def test_block_drop_table(sample_sqlite_db: Path) -> None:
    """Test that DROP TABLE statements are intercepted and state remains intact."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("DROP TABLE users;")

    assert result.success is False
    assert result.error_type == "SECURITY"
    assert "Prohibited" in str(result.error_message)

    # Verify table was not dropped
    conn = sqlite3.connect(sample_sqlite_db)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM users;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 5


def test_block_delete_from(sample_sqlite_db: Path) -> None:
    """Test that DELETE statements are intercepted and no rows are removed."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("DELETE FROM users WHERE user_id = 1;")

    assert result.success is False
    assert result.error_type == "SECURITY"
    assert "Prohibited" in str(result.error_message)

    # Verify zero rows deleted
    conn = sqlite3.connect(sample_sqlite_db)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM users WHERE user_id = 1;")
    exists = cursor.fetchone()[0]
    conn.close()
    assert exists == 1


def test_block_stacked_injection(sample_sqlite_db: Path) -> None:
    """Test that stacked multi-query injections are intercepted."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("SELECT * FROM users; DROP TABLE users;")

    assert result.success is False
    assert result.error_type == "SECURITY"
    assert "Stacked queries detected" in str(result.error_message)


def test_block_nested_cte_mutation(sample_sqlite_db: Path) -> None:
    """Test that DML inside CTEs is intercepted."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute(
        "WITH cte AS (SELECT 1) INSERT INTO users (name, age) VALUES ('Malicious', 99);"
    )

    assert result.success is False
    assert result.error_type == "SECURITY"


def test_enforce_timeout(sample_sqlite_db: Path) -> None:
    """Test that runaway recursive queries are interrupted at timeout threshold."""
    # Use a short timeout of 1.0 second for rapid unit testing
    env = SQLiteEnvironment(str(sample_sqlite_db), timeout_seconds=1.0)
    query = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT count(*) FROM cnt;"

    result = env.execute(query)

    assert result.success is False
    assert result.error_type == "TIMEOUT"
    # Execution time should be approximately 1000ms (+/- 300ms)
    assert result.execution_time_ms >= 800.0


def test_result_row_limit(sample_sqlite_db: Path) -> None:
    """Test that queries exceeding max_result_rows are capped without memory overflow."""
    env = SQLiteEnvironment(str(sample_sqlite_db), max_result_rows=2)
    result = env.execute("SELECT * FROM users;")

    assert result.success is True
    assert result.data is not None
    assert len(result.data) == 2
    assert result.row_count == 2


def test_set_operations_allowed(sample_sqlite_db: Path) -> None:
    """Test that UNION, INTERSECT, and EXCEPT set operations are allowed as read-only queries."""
    env = SQLiteEnvironment(str(sample_sqlite_db))

    union_res = env.execute(
        "SELECT user_id FROM users WHERE age > 25 UNION SELECT user_id FROM users WHERE age < 25;"
    )
    assert union_res.success is True

    intersect_res = env.execute(
        "SELECT user_id FROM users WHERE age >= 20 INTERSECT SELECT user_id FROM users WHERE age <= 30;"
    )
    assert intersect_res.success is True

    except_res = env.execute(
        "SELECT user_id FROM users EXCEPT SELECT user_id FROM users WHERE age > 30;"
    )
    assert except_res.success is True


def test_syntax_error_handling(sample_sqlite_db: Path) -> None:
    """Test handling of invalid SQL syntax."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("SELECT FROM WHERE ;")

    assert result.success is False
    assert result.error_type == "SYNTAX"


def test_empty_query_handling(sample_sqlite_db: Path) -> None:
    """Test handling of empty query string."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("   ;   ")

    assert result.success is False
    assert result.error_type == "SECURITY"


def test_table_not_found_operational_error(sample_sqlite_db: Path) -> None:
    """Test runtime execution error when table does not exist."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    result = env.execute("SELECT * FROM non_existent_table;")

    assert result.success is False
    assert result.error_type == "EXECUTION"
    assert "no such table" in str(result.error_message).lower()


def test_sqlite_env_schema_summary_delegation(sample_sqlite_db: Path) -> None:
    """Test get_schema_summary delegation through SQLiteEnvironment."""
    env = SQLiteEnvironment(str(sample_sqlite_db))
    summary = env.get_schema_summary()

    assert "CREATE TABLE users" in summary


def test_sqlite_env_missing_database_file(tmp_path: Path) -> None:
    """Test execution when database file physically does not exist."""
    missing = tmp_path / "ghost.sqlite"
    env = SQLiteEnvironment(str(missing), enforce_ast_safety=False)
    result = env.execute("SELECT 1;")

    assert result.success is False
    assert result.error_type == "EXECUTION"
    assert "not found" in str(result.error_message)
