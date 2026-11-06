"""DuckDB-to-SQLite conversion and validation pipeline for Spider 2.0-DBT enterprise assets."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


@dataclass
class TableValidationResult:
    """Validation report for an individual relational table export."""

    table_name: str
    duckdb_row_count: int
    sqlite_row_count: int
    column_count: int
    is_parity_verified: bool
    error_message: str | None = None


@dataclass
class ConversionReport:
    """Summary report detailing database conversion metrics and parity checks."""

    database_id: str
    source_duckdb_path: str
    target_sqlite_path: str
    total_tables: int = 0
    total_rows: int = 0
    tables: list[TableValidationResult] = field(default_factory=list)
    conversion_latency_ms: float = 0.0
    all_tables_verified: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary."""
        return asdict(self)


class DuckDBToSQLiteConverter:
    """Converts DuckDB databases into hermetic SQLite databases with full schema and row parity validation."""

    def __init__(self, batch_size: int = 2000) -> None:
        self.batch_size = max(100, batch_size)

    @staticmethod
    def map_duckdb_type_to_sqlite(duckdb_type: str) -> str:
        """Map DuckDB data type strings to SQLite standard storage affinities."""
        t = duckdb_type.upper().strip()

        # 1. Integer types & Boolean
        if any(
            t.startswith(prefix)
            for prefix in (
                "INT",
                "TINYINT",
                "SMALLINT",
                "BIGINT",
                "HUGEINT",
                "UTINYINT",
                "USMALLINT",
                "UINTEGER",
                "UBIGINT",
                "BOOL",
            )
        ):
            return "INTEGER"

        # 2. Real / Floating Point / Decimal types
        if any(
            t.startswith(prefix)
            for prefix in (
                "FLOAT",
                "DOUBLE",
                "REAL",
                "DECIMAL",
                "NUMERIC",
            )
        ):
            return "REAL"

        # 3. Binary Blob types
        if any(
            t.startswith(prefix)
            for prefix in (
                "BLOB",
                "BYTEA",
                "BINARY",
                "VARBINARY",
            )
        ):
            return "BLOB"

        # 4. Default: String / Date / Timestamp / UUID / Complex Struct/List
        return "TEXT"

    @staticmethod
    def _sanitize_value(val: Any) -> Any:
        """Sanitize DuckDB values (e.g. lists, dicts, complex types) for SQLite storage."""
        if val is None:
            return None
        if isinstance(val, (int, float, str, bytes)):
            return val
        if isinstance(val, bool):
            return 1 if val else 0
        # Serialize complex list/struct/dict objects to JSON string
        if isinstance(val, (list, dict)):
            return json.dumps(val, default=str)
        return str(val)

    def convert_database(
        self,
        source_path: str | Path,
        target_path: str | Path,
        verify_parity: bool = True,
        db_id: str | None = None,
    ) -> ConversionReport:
        """Convert a single DuckDB database file into a SQLite database with row-count parity check."""
        t_start = time.perf_counter()
        src = Path(source_path).resolve()
        tgt = Path(target_path).resolve()
        if db_id is None:
            db_id = src.stem

        report = ConversionReport(
            database_id=db_id,
            source_duckdb_path=str(src),
            target_sqlite_path=str(tgt),
        )

        if not src.exists():
            report.error = f"Source DuckDB database file does not exist: {src}"
            return report

        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.exists():
            tgt.unlink()

        try:
            duck_conn = duckdb.connect(str(src), read_only=True)
            sql_conn = sqlite3.connect(str(tgt))

            # Retrieve all user tables
            table_rows = duck_conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
                "ORDER BY table_name;"
            ).fetchall()
            table_names = [r[0] for r in table_rows]

            # Fallback if information_schema is empty: SHOW TABLES
            if not table_names:
                table_names = [r[0] for r in duck_conn.execute("SHOW TABLES;").fetchall()]

            report.total_tables = len(table_names)
            total_db_rows = 0
            all_verified = True

            for tbl in table_names:
                # 1. Fetch column metadata
                col_info = duck_conn.execute(
                    f"SELECT column_name, data_type FROM information_schema.columns "
                    f"WHERE table_schema = 'main' AND table_name = '{tbl}' "
                    f"ORDER BY ordinal_position;"
                ).fetchall()

                if not col_info:
                    # Fallback DESCRIBE
                    col_info = [
                        (r[0], r[1]) for r in duck_conn.execute(f'DESCRIBE "{tbl}";').fetchall()
                    ]

                col_defs: list[str] = []
                col_names: list[str] = []
                select_exprs: list[str] = []
                for col_name, data_type in col_info:
                    c_esc = col_name.replace('"', '""')
                    sqlite_type = self.map_duckdb_type_to_sqlite(str(data_type))
                    col_defs.append(f'"{c_esc}" {sqlite_type}')
                    col_names.append(col_name)

                    dt_upper = str(data_type).upper()
                    # Handle timestamp with timezone without requiring Python pytz runtime import
                    if "WITH TIME ZONE" in dt_upper or dt_upper.endswith("TZ"):
                        select_exprs.append(f'CAST("{c_esc}" AS VARCHAR) AS "{c_esc}"')
                    else:
                        select_exprs.append(f'"{c_esc}"')

                # 2. Create table in SQLite
                t_esc = tbl.replace('"', '""')
                cols_sql = ", ".join(col_defs)
                create_stmt = f'CREATE TABLE "{t_esc}" ({cols_sql});'
                sql_conn.execute(create_stmt)

                # 3. Stream data from DuckDB to SQLite
                cur = duck_conn.cursor()
                select_sql = f'SELECT {", ".join(select_exprs)} FROM "{t_esc}";'
                cur.execute(select_sql)
                placeholders = ", ".join(["?"] * len(col_names))
                insert_stmt = f'INSERT INTO "{t_esc}" VALUES ({placeholders});'

                table_rows_count = 0
                while True:
                    batch = cur.fetchmany(self.batch_size)
                    if not batch:
                        break
                    sanitized_batch = [tuple(self._sanitize_value(v) for v in row) for row in batch]
                    sql_conn.executemany(insert_stmt, sanitized_batch)
                    table_rows_count += len(batch)

                sql_conn.commit()
                total_db_rows += table_rows_count

                # 4. Parity verification
                duck_count = duck_conn.execute(f'SELECT COUNT(*) FROM "{tbl}";').fetchone()[0]
                sql_count = sql_conn.execute(f'SELECT COUNT(*) FROM "{tbl}";').fetchone()[0]
                is_parity = (duck_count == sql_count) if verify_parity else True
                if not is_parity:
                    all_verified = False

                report.tables.append(
                    TableValidationResult(
                        table_name=tbl,
                        duckdb_row_count=duck_count,
                        sqlite_row_count=sql_count,
                        column_count=len(col_names),
                        is_parity_verified=is_parity,
                    )
                )

            duck_conn.close()
            sql_conn.close()

            report.total_rows = total_db_rows
            report.all_tables_verified = all_verified
            report.conversion_latency_ms = (time.perf_counter() - t_start) * 1000.0

            logger.info(
                "Converted '%s' -> '%s': %d tables, %d rows in %.1f ms (Parity: %s)",
                db_id,
                tgt.name,
                report.total_tables,
                report.total_rows,
                report.conversion_latency_ms,
                "VERIFIED" if all_verified else "FAILED",
            )
            return report

        except Exception as e:
            report.error = str(e)
            report.conversion_latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error("Failed to convert DuckDB database '%s': %s", src, e)
            return report

    def convert_directory(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        verify_parity: bool = True,
        max_databases: int | None = None,
    ) -> list[ConversionReport]:
        """Convert all DuckDB files found in source_dir into SQLite databases in target_dir."""
        src_path = Path(source_dir).resolve()
        tgt_path = Path(target_dir).resolve()
        tgt_path.mkdir(parents=True, exist_ok=True)

        duckdb_files = sorted(
            [f for f in src_path.glob("**/*") if f.suffix.lower() in (".duckdb", ".ddb")]
        )
        if max_databases is not None and max_databases > 0:
            duckdb_files = duckdb_files[:max_databases]

        logger.info("Discovered %d DuckDB databases in '%s'", len(duckdb_files), src_path)

        reports: list[ConversionReport] = []
        for duck_file in duckdb_files:
            rel_path = duck_file.relative_to(src_path)
            if rel_path.parent == Path("."):
                target_sqlite = tgt_path / duck_file.stem / f"{duck_file.stem}.sqlite"
                database_id = duck_file.stem
            else:
                target_sqlite = tgt_path / rel_path.parent / f"{duck_file.stem}.sqlite"
                database_id = str(rel_path.with_suffix(""))

            rep = self.convert_database(
                source_path=duck_file,
                target_path=target_sqlite,
                verify_parity=verify_parity,
                db_id=database_id,
            )
            reports.append(rep)

        return reports
