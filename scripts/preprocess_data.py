"""Script to download and preprocess benchmark datasets into verified JSONL splits."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ask_the_database.data.downloader import DatasetDownloader
from ask_the_database.data.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute end-to-end dataset download and verified preprocessing."""
    logger.info("Initializing dataset preprocessing with config:\n%s", cfg)

    databases_dir = Path(cfg.paths.databases_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    raw_dir = Path(cfg.paths.raw_dir)

    databases_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloader = DatasetDownloader(raw_dir=str(raw_dir), databases_dir=str(databases_dir))

    # 1. Download and extract SQLite databases if needed
    hf_db_repo = getattr(cfg.data, "hf_database_repo", "HAL-9001/spider-databases")
    hf_db_file = getattr(cfg.data, "hf_database_file", "spider_data.zip")
    downloader.download_database_archive(repo_id=hf_db_repo, filename=hf_db_file)

    # 2. Download benchmark dataset
    hf_dataset_repo = cfg.data.hf_dataset_repo
    dataset = downloader.download_dataset(repo_id=hf_dataset_repo)

    preprocessor = DataPreprocessor(
        databases_dir=str(databases_dir),
        processed_dir=str(processed_dir),
    )

    dataset_name = cfg.data.dataset_name
    max_samples = getattr(cfg.data, "max_samples", None)

    # 3. Process Train Split
    train_split_name = getattr(cfg.data, "train_split", "train")
    if train_split_name in dataset:
        train_out = processed_dir / "train.jsonl"
        logger.info("Preprocessing train split (%s) -> %s", train_split_name, train_out)
        train_count = preprocessor.process_split(
            records=dataset[train_split_name],
            source=dataset_name,
            split="train",
            output_file=train_out,
            max_samples=max_samples,
        )
        logger.info("Successfully processed %d train samples.", train_count)

    # 4. Process Validation Split
    val_split_name = getattr(cfg.data, "validation_split", "validation")
    if val_split_name in dataset:
        val_out = processed_dir / "val.jsonl"
        logger.info("Preprocessing validation split (%s) -> %s", val_split_name, val_out)
        val_count = preprocessor.process_split(
            records=dataset[val_split_name],
            source=dataset_name,
            split="val",
            output_file=val_out,
            max_samples=max_samples,
        )
        logger.info("Successfully processed %d validation samples.", val_count)

    logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
