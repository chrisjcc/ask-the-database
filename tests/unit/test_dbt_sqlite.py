"""Unit tests for Spider 2.0-DBT DuckDB-to-SQLite conversion and validation pipeline."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from ask_the_database.data.dbt_sqlite.converter import (
    ConversionReport,
    DuckDBToSQLiteConverter,
)


@pytest.fixture
def mock_duckdb_path(tmp_path: Path) -> Path:
    """Create a temporary DuckDB database with representative schema and data types."""
    db_file = tmp_path / "mock_enterprise.duckdb"
    conn = duckdb.connect(str(db_file))

    # Table 1: Standard enterprise table with varied types
    conn.execute(
        """
        CREATE TABLE customers (
            customer_id BIGINT PRIMARY KEY,
            company_name VARCHAR,
            credit_limit DECIMAL(10, 2),
            is_active BOOLEAN,
            metadata JSON
        );
        """
    )
    conn.execute(
        """
        INSERT INTO customers VALUES
            (101, 'Acme Corp', 50000.50, true, '{"tier": "enterprise", "regions": ["US", "EU"]}'),
            (102, 'Beta LLC', 12500.00, false, '{"tier": "standard", "regions": ["APAC"]}'),
            (103, 'Gamma Inc', NULL, true, NULL);
        """
    )

    # Table 2: Table with reserved keyword identifier
    conn.execute(
        """
        CREATE TABLE "order" (
            order_id INT,
            customer_id INT,
            order_amount DOUBLE,
            created_at TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        INSERT INTO "order" VALUES
            (1, 101, 1500.75, TIMESTAMP '2024-01-15 10:30:00'),
            (2, 101, 3200.00, TIMESTAMP '2024-02-01 14:20:00'),
            (3, 102, 450.00, TIMESTAMP '2024-02-10 09:15:00');
        """
    )

    conn.close()
    return db_file


def test_type_mapping_rules() -> None:
    """Verify DuckDB type string mapping to SQLite storage affinities."""
    mapper = DuckDBToSQLiteConverter.map_duckdb_type_to_sqlite

    # Integer affinities
    assert mapper("INT") == "INTEGER"
    assert mapper("INTEGER") == "INTEGER"
    assert mapper("BIGINT") == "INTEGER"
    assert mapper("SMALLINT") == "INTEGER"
    assert mapper("BOOLEAN") == "INTEGER"
    assert mapper("BOOL") == "INTEGER"

    # Real affinities
    assert mapper("FLOAT") == "REAL"
    assert mapper("DOUBLE") == "REAL"
    assert mapper("DECIMAL(12, 4)") == "REAL"
    assert mapper("NUMERIC") == "REAL"

    # Blob affinities
    assert mapper("BLOB") == "BLOB"
    assert mapper("BYTEA") == "BLOB"

    # Text affinities
    assert mapper("VARCHAR") == "TEXT"
    assert mapper("TEXT") == "TEXT"
    assert mapper("TIMESTAMP") == "TEXT"
    assert mapper("DATE") == "TEXT"
    assert mapper("JSON") == "TEXT"
    assert mapper("UUID") == "TEXT"


def test_value_sanitization() -> None:
    """Verify serialization of varied data types for SQLite insertion."""
    sanitizer = DuckDBToSQLiteConverter._sanitize_value

    assert sanitizer(None) is None
    assert sanitizer(42) == 42
    assert sanitizer(3.14) == 3.14
    assert sanitizer("hello") == "hello"
    assert sanitizer(True) == 1
    assert sanitizer(False) == 0
    assert sanitizer([1, 2, 3]) == "[1, 2, 3]"
    assert sanitizer({"key": "val"}) == '{"key": "val"}'


def test_single_database_conversion(mock_duckdb_path: Path, tmp_path: Path) -> None:
    """Test full conversion of a DuckDB database into SQLite with parity verification."""
    target_sqlite = tmp_path / "enterprise.sqlite"
    converter = DuckDBToSQLiteConverter(batch_size=100)

    report = converter.convert_database(mock_duckdb_path, target_sqlite, verify_parity=True)

    assert isinstance(report, ConversionReport)
    assert report.error is None
    assert report.total_tables == 2
    assert report.total_rows == 6
    assert report.all_tables_verified is True
    assert report.conversion_latency_ms > 0

    # Verify SQLite database directly
    sql_conn = sqlite3.connect(str(target_sqlite))
    cust_rows = sql_conn.execute(
        "SELECT customer_id, company_name, is_active FROM customers ORDER BY customer_id;"
    ).fetchall()
    assert len(cust_rows) == 3
    assert cust_rows[0] == (101, "Acme Corp", 1)
    assert cust_rows[1] == (102, "Beta LLC", 0)

    order_rows = sql_conn.execute('SELECT COUNT(*) FROM "order";').fetchone()
    assert order_rows[0] == 3
    sql_conn.close()


def test_missing_source_database(tmp_path: Path) -> None:
    """Test error handling when source DuckDB file is nonexistent."""
    missing = tmp_path / "nonexistent.duckdb"
    target = tmp_path / "out.sqlite"
    converter = DuckDBToSQLiteConverter()

    report = converter.convert_database(missing, target)
    assert report.error is not None
    assert "does not exist" in report.error
    assert report.all_tables_verified is False


def test_directory_batch_conversion(tmp_path: Path) -> None:
    """Test batch conversion of multiple DuckDB files in a directory."""
    source_dir = tmp_path / "raw_duckdb"
    target_dir = tmp_path / "exported_sqlite"
    source_dir.mkdir(parents=True)

    # Create 2 DuckDB databases
    for name in ["db_finance", "db_logistics"]:
        p = source_dir / f"{name}.duckdb"
        con = duckdb.connect(str(p))
        con.execute(f"CREATE TABLE {name}_data (id INT, tag VARCHAR);")
        con.execute(f"INSERT INTO {name}_data VALUES (1, 'demo'), (2, 'test');")
        con.close()

    converter = DuckDBToSQLiteConverter()
    reports = converter.convert_directory(source_dir, target_dir, verify_parity=True)

    assert len(reports) == 2
    for r in reports:
        assert r.error is None
        assert r.all_tables_verified is True
        assert r.total_rows == 2

    # Verify output structure
    assert (target_dir / "db_finance" / "db_finance.sqlite").exists()
    assert (target_dir / "db_logistics" / "db_logistics.sqlite").exists()


def test_convert_cli_script(tmp_path: Path) -> None:
    """Verify scripts/convert_spider2_dbt.py execution via CLI."""
    source_dir = tmp_path / "cli_source"
    target_dir = tmp_path / "cli_target"
    report_file = tmp_path / "cli_report.json"
    source_dir.mkdir()

    # Create mock DuckDB
    con = duckdb.connect(str(source_dir / "cli_test.duckdb"))
    con.execute("CREATE TABLE metrics (val DOUBLE);")
    con.execute("INSERT INTO metrics VALUES (1.5), (2.5), (3.5);")
    con.close()

    cmd = [
        sys.executable,
        "scripts/convert_spider2_dbt.py",
        "--source-dir",
        str(source_dir),
        "--target-dir",
        str(target_dir),
        "--report-file",
        str(report_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert report_file.exists()

    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_discovered"] == 1
    assert data["total_successful"] == 1
    assert data["total_failed"] == 0


def test_timestamptz_conversion_hermetic(tmp_path: Path) -> None:
    """Verify conversion of TIMESTAMPTZ columns works hermetically without pytz runtime error."""
    source_db = tmp_path / "tz_test.duckdb"
    target_sql = tmp_path / "tz_test.sqlite"

    con = duckdb.connect(str(source_db))
    con.execute("CREATE TABLE events (event_id INT, occurred_at TIMESTAMPTZ);")
    con.execute(
        "INSERT INTO events VALUES (1, '2024-03-01 15:30:00+00'), (2, '2024-06-15 08:45:00+02');"
    )
    con.close()

    converter = DuckDBToSQLiteConverter()
    report = converter.convert_database(source_db, target_sql, verify_parity=True)

    assert report.error is None
    assert report.all_tables_verified is True
    assert report.total_rows == 2

    # Check SQLite contents
    sql_conn = sqlite3.connect(str(target_sql))
    rows = sql_conn.execute(
        "SELECT event_id, occurred_at FROM events ORDER BY event_id;"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == 1
    assert "2024-03-01" in str(rows[0][1])
    sql_conn.close()


def test_nested_hierarchy_conversion(tmp_path: Path) -> None:
    """Verify nested task directories (e.g. analytics001/solution/gold.duckdb) do not collide."""
    source_root = tmp_path / "source_root"
    target_root = tmp_path / "target_root"

    task1_dir = source_root / "task_01" / "solution"
    task2_dir = source_root / "task_02" / "solution"
    task1_dir.mkdir(parents=True)
    task2_dir.mkdir(parents=True)

    # Both named gold.duckdb
    for d, val in [(task1_dir, "t1_data"), (task2_dir, "t2_data")]:
        db_file = d / "gold.duckdb"
        con = duckdb.connect(str(db_file))
        con.execute("CREATE TABLE payload (info VARCHAR);")
        con.execute(f"INSERT INTO payload VALUES ('{val}');")
        con.close()

    converter = DuckDBToSQLiteConverter()
    reports = converter.convert_directory(source_root, target_root, verify_parity=True)

    assert len(reports) == 2
    assert all(r.all_tables_verified for r in reports)

    out1 = target_root / "task_01" / "solution" / "gold.sqlite"
    out2 = target_root / "task_02" / "solution" / "gold.sqlite"

    assert out1.exists()
    assert out2.exists()

    c1 = sqlite3.connect(str(out1))
    assert c1.execute("SELECT info FROM payload;").fetchone()[0] == "t1_data"
    c1.close()

    c2 = sqlite3.connect(str(out2))
    assert c2.execute("SELECT info FROM payload;").fetchone()[0] == "t2_data"
    c2.close()
