"""CLI script to prepare Spider 2.0-SQLite benchmark evaluation split (135 tasks)."""

import argparse
import logging
from pathlib import Path

from ask_the_database.data.spider2.preprocessor import prepare_spider2_sqlite_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_spider2_sqlite")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Spider 2.0-SQLite benchmark preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare Spider 2.0-SQLite 135-task benchmark split for evaluation."
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=Path("data/raw/spider2_lite/spider2-lite.jsonl"),
        help="Path to spider2-lite.jsonl file.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("data/databases/spider2_localdb"),
        help="Directory containing Spider 2.0 SQLite databases (.sqlite).",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/raw/spider2_lite/gold"),
        help="Directory containing ground-truth evaluation CSV files ({instance_id}.csv).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/spider2_sqlite.jsonl"),
        help="Target output JSONL path for the prepared evaluation split.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable automatic download of spider2-lite.jsonl if not found locally.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute CLI task preparation."""
    args = parse_args()
    prepare_spider2_sqlite_tasks(
        source_file=args.source_file,
        db_dir=args.db_dir,
        gold_dir=args.gold_dir,
        output_file=args.output_file,
        allow_download=not args.no_download,
    )


if __name__ == "__main__":
    main()
