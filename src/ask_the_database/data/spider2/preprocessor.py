"""Spider 2.0-SQLite benchmark data preprocessor and task extractor (135 tasks)."""

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPIDER2_LITE_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/xlang-ai/Spider2/main/spider2-lite/spider2-lite.jsonl"
)


def fetch_upstream_jsonl(target_path: Path) -> None:
    """Download spider2-lite.jsonl from upstream repository."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading spider2-lite.jsonl from %s...", SPIDER2_LITE_UPSTREAM_URL)
    urllib.request.urlretrieve(SPIDER2_LITE_UPSTREAM_URL, target_path)
    logger.info("Saved upstream benchmark definitions to '%s'", target_path)


def resolve_db_path(db_name: str, db_dir: Path) -> Path:
    """Search for database file across common naming patterns."""
    candidates = [
        db_dir / f"{db_name}.sqlite",
        db_dir / f"{db_name.lower()}.sqlite",
        db_dir / db_name / f"{db_name}.sqlite",
        Path("data/databases/spider2_dbt") / db_name / f"{db_name}.sqlite",
    ]
    for c in candidates:
        if c.exists():
            return c
    return db_dir / f"{db_name}.sqlite"


class Spider2SQLitePreprocessor:
    """Extracts, formats, and validates the 135 SQLite tasks from Spider 2.0-Lite."""

    def __init__(self, db_dir: str | Path, gold_dir: str | Path) -> None:
        self.db_dir = Path(db_dir)
        self.gold_dir = Path(gold_dir)

    def extract_tasks(
        self,
        source_file: str | Path,
        allow_download: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract and format the 135 local SQLite tasks from a spider2-lite.jsonl source."""
        src = Path(source_file)
        if not src.exists():
            if allow_download:
                try:
                    fetch_upstream_jsonl(src)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Source file '{src}' does not exist and download failed: {e}"
                    ) from e
            else:
                raise FileNotFoundError(f"Source file '{src}' does not exist: {src}")

        tasks: list[dict[str, Any]] = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                entry = json.loads(line)
                instance_id = str(entry.get("instance_id", ""))
                # The 135 SQLite tasks in Spider 2.0-Lite are prefixed with 'local'
                if not instance_id.startswith("local") and entry.get("db_type") != "sqlite":
                    continue

                db_name = str(entry.get("db", ""))
                question = str(entry.get("question", "")).strip()
                ext_knowledge = entry.get("external_knowledge")

                if ext_knowledge:
                    prompt_text = f"{question} (Context note: Refer to {ext_knowledge})"
                else:
                    prompt_text = question

                db_path = resolve_db_path(db_name, self.db_dir)
                gold_csv_path = self.gold_dir / f"{instance_id}.csv"

                tasks.append(
                    {
                        "instance_id": instance_id,
                        "question": prompt_text,
                        "db_id": db_name,
                        "db_path": str(db_path),
                        "gold_csv_path": str(gold_csv_path) if gold_csv_path.exists() else None,
                        "gold_sql": "SELECT 1;",
                        "benchmark": "spider2_sqlite",
                    }
                )

        return tasks

    def process_and_save(
        self,
        source_file: str | Path,
        output_file: str | Path,
        allow_download: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract tasks and save them to a target JSONL file."""
        tasks = self.extract_tasks(source_file, allow_download=allow_download)
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        logger.info("============================================================")
        logger.info("SPIDER 2.0-SQLITE BENCHMARK PREPARATION SUMMARY")
        logger.info("============================================================")
        logger.info("Total SQLite tasks processed : %d", len(tasks))
        logger.info("Saved to                     : %s", out)
        existing_dbs = sum(1 for t in tasks if Path(t["db_path"]).exists())
        existing_gold = sum(
            1 for t in tasks if t["gold_csv_path"] and Path(t["gold_csv_path"]).exists()
        )
        logger.info("Databases present on disk    : %d / %d", existing_dbs, len(tasks))
        logger.info("Gold CSVs present on disk    : %d / %d", existing_gold, len(tasks))
        logger.info("============================================================")
        return tasks


def prepare_spider2_sqlite_tasks(
    source_file: Path,
    db_dir: Path,
    gold_dir: Path,
    output_file: Path,
    allow_download: bool = True,
) -> list[dict[str, Any]]:
    """Functional convenience wrapper for Spider2SQLitePreprocessor."""
    preprocessor = Spider2SQLitePreprocessor(db_dir=db_dir, gold_dir=gold_dir)
    return preprocessor.process_and_save(
        source_file=source_file,
        output_file=output_file,
        allow_download=allow_download,
    )
