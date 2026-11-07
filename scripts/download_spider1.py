"""Automated downloader for Spider 1.0 SQLite database archives.

Downloads and extracts the Spider 1.0 SQLite databases from Hugging Face Hub (HAL-9001/spider-databases).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ask_the_database.data.downloader import DatasetDownloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_spider1")

DEFAULT_REPO_ID = "HAL-9001/spider-databases"
DEFAULT_FILENAME = "spider_data.zip"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DATABASES_DIR = Path("data/databases")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract Spider 1.0 SQLite database archive from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_FILENAME,
        help=f"Database archive filename in repository (default: {DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Target directory for raw downloaded archive (default: {DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--databases-dir",
        type=Path,
        default=DEFAULT_DATABASES_DIR,
        help=f"Target directory where extracted SQLite databases are placed (default: {DEFAULT_DATABASES_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if already present.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute standalone Spider 1.0 SQLite database download and extraction."""
    args = parse_args()
    logger.info("Initializing Spider 1.0 database archive download...")
    logger.info("Source repository: %s/%s", args.repo_id, args.filename)
    logger.info("Target directory : %s", args.databases_dir)

    downloader = DatasetDownloader(
        raw_dir=str(args.raw_dir),
        databases_dir=str(args.databases_dir),
    )

    db_path = downloader.download_database_archive(
        repo_id=args.repo_id,
        filename=args.filename,
        force_redownload=args.force,
    )

    sqlite_count = len(list(Path(db_path).glob("**/*.sqlite")))
    logger.info("============================================================")
    logger.info("SPIDER 1.0 DATABASE EXTRACTION SUMMARY")
    logger.info("============================================================")
    logger.info("Databases directory : %s", db_path)
    logger.info("SQLite files ready   : %d", sqlite_count)
    logger.info("============================================================")


if __name__ == "__main__":
    main()
