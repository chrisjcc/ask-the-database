"""Schema extraction and formatting for SQLite databases."""

import os
import sqlite3
from typing import Any


class SchemaManager:
    """Extracts schema, columns, primary keys, and foreign keys from SQLite databases."""

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Create a read-only SQLite connection for schema inspection."""
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0)
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def get_tables(self) -> list[str]:
        """Return a sorted list of user table names."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name;"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_table_columns(self, table_name: str) -> list[str]:
        """Return a list of column names for a given table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            return [row[1] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        """Return detailed column metadata for a given table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            # cid, name, type, notnull, dflt_value, pk
            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default_value": row[4],
                        "pk": bool(row[5]),
                    }
                )
            return columns

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """Return foreign key relationships for a given table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            # id, seq, table, from, to, on_update, on_delete, match
            fks = []
            for row in cursor.fetchall():
                fks.append(
                    {
                        "id": row[0],
                        "seq": row[1],
                        "to_table": row[2],
                        "from_col": row[3],
                        "to_col": row[4],
                    }
                )
            return fks

    def get_raw_ddl(self, table_name: str | None = None) -> str:
        """Return the raw CREATE TABLE SQL from sqlite_master."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if table_name:
                cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
                    (table_name,),
                )
                row = cursor.fetchone()
                return str(row[0]) if row and row[0] else ""
            cursor.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
                "ORDER BY name;"
            )
            rows = cursor.fetchall()
            return "\n\n".join(row[0] for row in rows if row[0])

    def get_schema_summary(self) -> str:
        """Generate a formatted schema representation suitable for LLM prompts."""
        tables = self.get_tables()
        if not tables:
            return "No tables found."

        summary_parts = []
        for table in tables:
            columns_info = self.get_table_info(table)
            fks = self.get_foreign_keys(table)

            col_defs = []
            for col in columns_info:
                type_str = f" {col['type']}" if col["type"] else ""
                pk_str = " PRIMARY KEY" if col["pk"] else ""
                col_defs.append(f"    {col['name']}{type_str}{pk_str}")

            for fk in fks:
                col_defs.append(
                    f"    FOREIGN KEY ({fk['from_col']}) REFERENCES {fk['to_table']}({fk['to_col']})"
                )

            table_ddl = f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);"
            summary_parts.append(table_ddl)

        return "\n\n".join(summary_parts)
