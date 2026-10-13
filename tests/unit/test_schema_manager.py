"""Unit tests for SchemaManager SQLite inspection and formatting."""

from pathlib import Path

import pytest

from ask_the_database.environment.schema_manager import SchemaManager


def test_schema_manager_tables(sample_sqlite_db: Path) -> None:
    """Test retrieving list of user tables."""
    manager = SchemaManager(str(sample_sqlite_db))
    tables = manager.get_tables()

    assert tables == ["departments", "orders", "users"]


def test_schema_manager_columns(sample_sqlite_db: Path) -> None:
    """Test retrieving column names for specific tables."""
    manager = SchemaManager(str(sample_sqlite_db))
    user_cols = manager.get_table_columns("users")

    assert user_cols == ["user_id", "name", "age", "department_id"]


def test_schema_manager_table_info(sample_sqlite_db: Path) -> None:
    """Test column metadata extraction."""
    manager = SchemaManager(str(sample_sqlite_db))
    info = manager.get_table_info("departments")

    assert len(info) == 3
    col_names = [c["name"] for c in info]
    assert "department_id" in col_names
    assert "department_name" in col_names
    assert "budget" in col_names

    pk_col = next(c for c in info if c["name"] == "department_id")
    assert pk_col["pk"] is True


def test_schema_manager_foreign_keys(sample_sqlite_db: Path) -> None:
    """Test relational foreign key extraction."""
    manager = SchemaManager(str(sample_sqlite_db))
    fks = manager.get_foreign_keys("users")

    assert len(fks) == 1
    assert fks[0]["to_table"] == "departments"
    assert fks[0]["from_col"] == "department_id"
    assert fks[0]["to_col"] == "department_id"


def test_schema_summary_contains_ddl(sample_sqlite_db: Path) -> None:
    """Test formatted DDL schema summary for prompt injection."""
    manager = SchemaManager(str(sample_sqlite_db))
    summary = manager.get_schema_summary()

    assert "CREATE TABLE departments" in summary
    assert "CREATE TABLE users" in summary
    assert "CREATE TABLE orders" in summary
    assert "FOREIGN KEY (department_id) REFERENCES departments(department_id)" in summary


def test_schema_manager_nonexistent_file(tmp_path: Path) -> None:
    """Test FileNotFoundError when pointing to a nonexistent database."""
    missing = tmp_path / "nonexistent.sqlite"
    with pytest.raises(FileNotFoundError):
        SchemaManager(str(missing))


def test_schema_manager_raw_ddl(sample_sqlite_db: Path) -> None:
    """Test get_raw_ddl for specific table and all tables."""
    manager = SchemaManager(str(sample_sqlite_db))
    all_ddl = manager.get_raw_ddl()
    assert "CREATE TABLE departments" in all_ddl
    assert "CREATE TABLE users" in all_ddl

    single_ddl = manager.get_raw_ddl("users")
    assert "CREATE TABLE users" in single_ddl
    assert "CREATE TABLE departments" not in single_ddl


def test_schema_manager_empty_db(tmp_path: Path) -> None:
    """Test schema summary on database with zero tables."""
    import sqlite3

    empty_db = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(empty_db)
    conn.close()

    manager = SchemaManager(str(empty_db))
    assert manager.get_tables() == []
    assert manager.get_schema_summary() == "No tables found."
