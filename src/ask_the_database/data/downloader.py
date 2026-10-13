"""Downloader for NL2SQL datasets and SQLite database archives."""

import logging
import shutil
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Manages downloading Hugging Face datasets and extracting SQLite database files."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        databases_dir: str = "data/databases",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.databases_dir = Path(databases_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.databases_dir.mkdir(parents=True, exist_ok=True)

    def download_database_archive(
        self,
        repo_id: str = "HAL-9001/spider-databases",
        filename: str = "spider_data.zip",
        force_redownload: bool = False,
    ) -> Path:
        """Download and extract SQLite database archive from Hugging Face Hub."""
        flag_file = self.databases_dir / ".extracted"
        if flag_file.exists() and not force_redownload:
            logger.info("Databases already downloaded and extracted in %s", self.databases_dir)
            return self.databases_dir

        logger.info("Downloading database archive %s/%s...", repo_id, filename)
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(self.raw_dir),
        )

        logger.info("Extracting SQLite databases to %s...", self.databases_dir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                # Avoid macOS metadata and non-sqlite or hidden files
                if "__MACOSX" in member or member.startswith("._"):
                    continue

                if member.endswith(".sqlite"):
                    parts = Path(member).parts
                    # Expected pattern: spider_data/database/<db_id>/<db_id>.sqlite
                    # or database/<db_id>/<db_id>.sqlite
                    db_id = parts[-2]
                    target_dir = self.databases_dir / db_id
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / Path(member).name

                    with archive.open(member) as source, open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)

        flag_file.write_text("extraction_complete\n")
        logger.info("All SQLite databases successfully extracted to %s", self.databases_dir)
        return self.databases_dir

    def download_dataset(
        self,
        repo_id: str = "xlangai/spider",
    ) -> DatasetDict | Dataset:
        """Download dataset splits from Hugging Face."""
        logger.info("Loading dataset %s from Hugging Face Hub...", repo_id)
        return load_dataset(repo_id)

    def get_database_path(self, db_id: str) -> str | None:
        """Return the physical path to a database's .sqlite file if it exists."""
        candidate = self.databases_dir / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return str(candidate)
        # Search recursively within databases_dir as fallback
        matches = list(self.databases_dir.glob(f"**/{db_id}.sqlite"))
        if matches:
            return str(matches[0])
        return None
