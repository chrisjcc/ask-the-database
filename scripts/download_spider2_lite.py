"""Automated downloader for Spider 2.0-Lite benchmark assets.

Downloads the 30 enterprise SQLite databases and 135 task gold CSV execution results.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GDRIVE_FILE_ID = "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG"
SPIDER2_REPO_URL = "https://github.com/xlang-ai/Spider2.git"


def download_github_metadata(target_raw_dir: Path) -> None:
    """Clones metadata (spider2-lite.jsonl) and gold execution CSVs from upstream GitHub."""
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = target_raw_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Cloning Spider2 repository metadata from %s...", SPIDER2_REPO_URL)
        subprocess.run(
            ["git", "clone", "--depth", "1", SPIDER2_REPO_URL, tmp_dir],
            check=True,
        )
        spider2_lite_dir = Path(tmp_dir) / "spider2-lite"

        src_jsonl = spider2_lite_dir / "spider2-lite.jsonl"
        if src_jsonl.exists():
            shutil.copy2(src_jsonl, target_raw_dir / "spider2-lite.jsonl")
            logger.info("Saved 'spider2-lite.jsonl' to '%s'", target_raw_dir)

        src_gold = spider2_lite_dir / "evaluation_suite" / "gold" / "exec_result"
        if src_gold.exists():
            count = 0
            for csv_file in src_gold.glob("*.csv"):
                shutil.copy2(csv_file, gold_dir / csv_file.name)
                count += 1
            logger.info("Copied %d gold CSV execution results to '%s'", count, gold_dir)


def download_sqlite_databases(target_db_dir: Path) -> None:
    """Downloads spider2-localdb.zip from Google Drive and unpacks SQLite databases."""
    target_db_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        logger.info("Installing 'gdown' package for Google Drive asset download...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "spider2-localdb.zip"
        logger.info("Downloading Spider 2.0 SQLite databases archive (~457 MB)...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(zip_path), quiet=False)

        logger.info("Extracting SQLite databases into '%s'...", target_db_dir)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_db_dir)

    # Clean up any __MACOSX metadata folder created during unzip
    macosx_dir = target_db_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir, ignore_errors=True)

    sqlite_count = len(list(target_db_dir.glob("*.sqlite")))
    logger.info(
        "Extraction complete: %d SQLite databases ready in '%s'", sqlite_count, target_db_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Spider 2.0-Lite SQLite assets.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/spider2_lite"),
        help="Target directory for raw task metadata and gold CSV files.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("data/databases/spider2_localdb"),
        help="Target directory for unpacked SQLite databases.",
    )
    parser.add_argument(
        "--skip-databases",
        action="store_true",
        help="Skip downloading the SQLite database archive.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip cloning metadata and gold CSV files from GitHub.",
    )

    args = parser.parse_args()

    if not args.skip_metadata:
        download_github_metadata(args.raw_dir)

    if not args.skip_databases:
        download_sqlite_databases(args.db_dir)

    logger.info("All Spider 2.0-Lite assets successfully downloaded and organized.")


if __name__ == "__main__":
    main()
