"""Base abstractions for database execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Standardized result returned by the database sandbox."""

    success: bool
    data: list[tuple[Any, ...]] | None
    columns: list[str] | None
    row_count: int
    execution_time_ms: float
    error_message: str | None
    error_type: str | None  # e.g., "SYNTAX", "SECURITY", "TIMEOUT", "EXECUTION"


class BaseSQLEnvironment(ABC):
    """Abstract interface for database execution sandboxes."""

    @abstractmethod
    def execute(self, query: str) -> ExecutionResult:
        """Execute a SQL query within the sandboxed environment."""
        pass

    @abstractmethod
    def get_schema_summary(self) -> str:
        """Return a formatted string representing the schema for LLM prompting."""
        pass
