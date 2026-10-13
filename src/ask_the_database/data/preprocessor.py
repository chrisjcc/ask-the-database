"""Data preprocessing and grounding validation pipeline."""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from ask_the_database.environment.schema_manager import SchemaManager
from ask_the_database.environment.sqlite_env import SQLiteEnvironment

logger = logging.getLogger(__name__)


class ProcessedSample(BaseModel):
    """Normalized, grounded question-SQL sample adhering to canonical schema."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    source: str
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    schema_ddl: str
    difficulty: str
    gold_row_count: int
    gold_execution_hash: str


class DataPreprocessor:
    """Validates, grounds, and normalizes raw dataset records into canonical JSONL formats."""

    def __init__(
        self,
        databases_dir: str = "data/databases",
        processed_dir: str = "data/processed",
        timeout_seconds: float = 5.0,
    ) -> None:
        self.databases_dir = Path(databases_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self._schema_cache: dict[str, str] = {}

    def get_database_path(self, db_id: str) -> Path | None:
        """Find the physical .sqlite file for a given database ID."""
        candidate = self.databases_dir / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
        matches = list(self.databases_dir.glob(f"**/{db_id}.sqlite"))
        if matches:
            return matches[0]
        return None

    def get_schema_ddl(self, db_path: Path) -> str:
        """Fetch and cache schema DDL string for a database."""
        path_str = str(db_path)
        if path_str not in self._schema_cache:
            manager = SchemaManager(path_str)
            self._schema_cache[path_str] = manager.get_schema_summary()
        return self._schema_cache[path_str]

    @staticmethod
    def compute_execution_hash(data: list[tuple[Any, ...]] | None) -> str:
        """Generate a deterministic SHA-256 hash from query result rows."""
        if not data:
            return hashlib.sha256(b"EMPTY_RESULT").hexdigest()
        normalized = repr(sorted(str(row) for row in data))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def process_record(
        self,
        record: dict[str, Any],
        source: str,
        split: str,
        index: int,
    ) -> ProcessedSample | None:
        """Process a single record, verify its gold SQL executes, and return ProcessedSample."""
        db_id = record.get("db_id")
        if not db_id:
            logger.warning("Record %s missing db_id, skipping.", index)
            return None

        # Extract SQL and question across Spider and SkyRL naming conventions
        gold_sql = record.get("gold_sql") or record.get("query") or record.get("sql")
        question = record.get("question")
        difficulty = record.get("difficulty", "medium")

        if not gold_sql or not question:
            logger.warning("Record %s missing question or SQL, skipping.", index)
            return None

        db_path = self.get_database_path(db_id)
        if not db_path:
            logger.warning(
                "Database file for %s not found in %s, skipping.", db_id, self.databases_dir
            )
            return None

        # Schema extraction
        try:
            schema_ddl = self.get_schema_ddl(db_path)
        except Exception as e:
            logger.warning("Failed extracting schema for %s: %s", db_id, e)
            return None

        # Execute gold SQL in sandbox to verify execution correctness
        env = SQLiteEnvironment(
            str(db_path),
            timeout_seconds=self.timeout_seconds,
            enforce_ast_safety=True,
        )
        res = env.execute(gold_sql)

        if not res.success:
            logger.warning(
                "Gold SQL failed execution for db_id=%s [sample %s]: %s (Query: %s)",
                db_id,
                index,
                res.error_message,
                gold_sql,
            )
            return None

        instance_id = f"{source}_{split}_{index:05d}"
        exec_hash = self.compute_execution_hash(res.data)

        # Store relative db_path for portability
        try:
            rel_db_path = str(db_path.relative_to(Path.cwd()))
        except ValueError:
            rel_db_path = str(db_path)

        return ProcessedSample(
            instance_id=instance_id,
            source=source,
            db_id=db_id,
            question=question.strip(),
            gold_sql=gold_sql.strip(),
            db_path=rel_db_path,
            schema_ddl=schema_ddl,
            difficulty=str(difficulty).lower(),
            gold_row_count=res.row_count,
            gold_execution_hash=exec_hash,
        )

    def process_split(
        self,
        records: Iterable[dict[str, Any]],
        source: str,
        split: str,
        output_file: Path | str,
        max_samples: int | None = None,
    ) -> int:
        """Process an entire split and write valid samples to a JSONL file."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        valid_count = 0
        total_seen = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, record in enumerate(tqdm(records, desc=f"Processing {source}:{split}")):
                if max_samples and valid_count >= max_samples:
                    break
                total_seen += 1
                sample = self.process_record(record, source=source, split=split, index=idx)
                if sample is not None:
                    f.write(sample.model_dump_json() + "\n")
                    valid_count += 1

        logger.info(
            "Wrote %d valid samples out of %d seen to %s",
            valid_count,
            total_seen,
            out_path,
        )
        return valid_count
