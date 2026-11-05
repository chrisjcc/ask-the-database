"""CLI script for converting Spider 2.0-DBT DuckDB enterprise assets into validated SQLite databases."""

import argparse
import json
import logging
import sys
from pathlib import Path

from ask_the_database.data.dbt_sqlite.converter import DuckDBToSQLiteConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("convert_spider2_dbt")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Spider 2.0-DBT DuckDB databases to hermetic SQLite databases."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/spider2_dbt"),
        help="Directory containing source DuckDB databases (.duckdb, .ddb).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/databases/spider2_dbt"),
        help="Target directory where converted SQLite databases will be saved.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("results/spider2_dbt_conversion_summary.json"),
        help="Path where the JSON conversion and parity verification report is saved.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip row-count parity verification between DuckDB and SQLite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of databases to convert (useful for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Streaming batch size for row insertions.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute batch database conversion and parity validation."""
    args = parse_args()
    logger.info("Starting Spider 2.0-DBT DuckDB-to-SQLite conversion pipeline...")
    logger.info("Source directory : %s", args.source_dir)
    logger.info("Target directory : %s", args.target_dir)

    converter = DuckDBToSQLiteConverter(batch_size=args.batch_size)
    reports = converter.convert_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        verify_parity=not args.no_verify,
        max_databases=args.limit,
    )

    total_converted = len(reports)
    successful = [r for r in reports if r.error is None and r.all_tables_verified]
    failed = [r for r in reports if r.error is not None or not r.all_tables_verified]

    summary = {
        "total_discovered": total_converted,
        "total_successful": len(successful),
        "total_failed": len(failed),
        "databases": [r.to_dict() for r in reports],
    }

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("============================================================")
    logger.info("SPIDER 2.0-DBT ASSET CONVERSION SUMMARY")
    logger.info("============================================================")
    logger.info("Databases processed : %d", total_converted)
    logger.info("Successfully verified: %d", len(successful))
    logger.info("Failures / Mismatches: %d", len(failed))
    logger.info("Report written to    : %s", args.report_file)
    logger.info("============================================================")

    if failed:
        logger.error("Some databases failed conversion or parity verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()
