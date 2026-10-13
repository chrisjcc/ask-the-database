"""Interactive database tools registry for multi-turn SQL agent."""

from dataclasses import dataclass
from typing import Any

from ask_the_database.environment.schema_manager import SchemaManager
from ask_the_database.environment.sqlite_env import SQLiteEnvironment


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool invocation."""

    action: str
    output: str
    is_terminal: bool
    final_query: str | None = None


class DatabaseToolRegistry:
    """Registry of interactive tools exposed to the SQL agent."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.env = SQLiteEnvironment(db_path)
        self.schema_manager = SchemaManager(db_path)

    def inspect_schema(self, table_names: list[str]) -> ToolResult:
        """Inspect table DDL and relationships."""
        ddl_chunks: list[str] = []
        for table in table_names:
            table_clean = table.strip()
            if not table_clean:
                continue
            ddl = self.schema_manager.get_table_ddl(table_clean)
            if ddl:
                ddl_chunks.append(ddl)
            else:
                ddl_chunks.append(f"Error: Table '{table_clean}' not found in database.")
        output = "\n\n".join(ddl_chunks) if ddl_chunks else "No tables specified."
        return ToolResult(
            action="inspect_schema",
            output=output,
            is_terminal=False,
        )

    def sample_data(self, table_name: str, limit: int = 3) -> ToolResult:
        """Sample rows from a specific table."""
        table_clean = table_name.strip().strip("'\"")
        res = self.env.execute(f"SELECT * FROM {table_clean} LIMIT {limit};")
        if not res.success:
            return ToolResult(
                action="sample_data",
                output=f"Execution error: {res.error_message}",
                is_terminal=False,
            )
        col_str = " | ".join(res.columns or [])
        row_strs = [" | ".join(str(cell) for cell in row) for row in (res.data or [])]
        table_output = f"{col_str}\n" + "-" * max(len(col_str), 1) + "\n" + "\n".join(row_strs)
        return ToolResult(
            action="sample_data",
            output=table_output,
            is_terminal=False,
        )

    def execute_sql(self, sql_query: str) -> ToolResult:
        """Run trial query against database sandbox."""
        res = self.env.execute(sql_query)
        if not res.success:
            return ToolResult(
                action="execute_sql",
                output=f"Database Error: {res.error_message}",
                is_terminal=False,
            )
        row_preview = str(res.data[:5]) if res.data else "[]"
        return ToolResult(
            action="execute_sql",
            output=f"Success: {res.row_count} rows returned. Preview: {row_preview}",
            is_terminal=False,
        )

    def final_answer(self, sql_query: str) -> ToolResult:
        """Submit final SQL query and terminate episode."""
        return ToolResult(
            action="final_answer",
            output="Submitted final query.",
            is_terminal=True,
            final_query=sql_query.strip(),
        )

    def execute_action(self, action_name: str, action_payload: Any) -> ToolResult:
        """Dispatch action dynamically by name."""
        if action_name == "inspect_schema":
            if isinstance(action_payload, list):
                tables = action_payload
            elif isinstance(action_payload, str):
                cleaned = action_payload.strip("[]")
                tables = [t.strip().strip("'\"") for t in cleaned.split(",") if t.strip()]
            else:
                tables = []
            return self.inspect_schema(tables)
        elif action_name == "sample_data":
            return self.sample_data(str(action_payload))
        elif action_name == "execute_sql":
            return self.execute_sql(str(action_payload))
        elif action_name == "final_answer":
            return self.final_answer(str(action_payload))
        else:
            return ToolResult(
                action=action_name,
                output=f"Error: Unknown tool '{action_name}'. Valid tools: inspect_schema, sample_data, execute_sql, final_answer.",
                is_terminal=False,
            )
