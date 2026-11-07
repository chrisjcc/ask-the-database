"""Automated downloader for Spider 2.0-DBT DuckDB enterprise database assets.

Downloads the enterprise DuckDB databases from Hugging Face Hub (harborframework/harbor-datasets).
Supports both YAML config defaults and CLI argument overrides.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

# Attempt to import OmegaConf for Hydra YAML defaults
try:
    from omegaconf import OmegaConf

    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_spider2_dbt")

DEFAULT_CONFIG_PATH = Path("configs/data/spider2_dbt.yaml")
DEFAULT_REPO_ID = "harborframework/harbor-datasets"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_REMOTE_DIR = "datasets/spider2-dbt"
DEFAULT_TARGET_DIR = Path("data/raw/spider2_dbt")


def load_yaml_defaults(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load default parameters from Hydra YAML configuration if present."""
    if HAS_OMEGACONF and config_path.exists():
        try:
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        except Exception as e:
            logger.warning("Could not load YAML config from %s: %s", config_path, e)
    return {}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with fallback to YAML configuration defaults."""
    yaml_cfg = load_yaml_defaults()

    parser = argparse.ArgumentParser(
        description="Download Spider 2.0-DBT enterprise DuckDB databases from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=yaml_cfg.get("hf_repo_id", DEFAULT_REPO_ID),
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default=yaml_cfg.get("repo_type", DEFAULT_REPO_TYPE),
        help=f"Hugging Face repository type (default: {DEFAULT_REPO_TYPE}).",
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default=yaml_cfg.get("remote_dir", DEFAULT_REMOTE_DIR),
        help=f"Directory path within the Hugging Face repository (default: {DEFAULT_REMOTE_DIR}).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(yaml_cfg.get("target_raw_dir", DEFAULT_TARGET_DIR)),
        help=f"Local target directory where DuckDB files are saved (default: {DEFAULT_TARGET_DIR}).",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=yaml_cfg.get("file_pattern", "*.duckdb"),
        help="Glob pattern to filter remote files (default: '*.duckdb').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to download (useful for quick verification).",
    )
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="Filter to only download environment database instances (skipping solution/gold duckdb files).",
    )
    parser.add_argument(
        "--strip-prefix",
        action="store_true",
        default=yaml_cfg.get("strip_prefix", True),
        help="Strip the remote directory prefix so files are stored relative to remote-dir.",
    )
    parser.add_argument(
        "--no-strip-prefix",
        dest="strip_prefix",
        action="store_false",
        help="Retain the full remote repository directory path locally.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist locally with matching size.",
    )
    return parser.parse_args()


def download_spider2_dbt_assets(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    target_dir: Path = DEFAULT_TARGET_DIR,
    file_pattern: str = "*.duckdb",
    limit: int | None = None,
    environment_only: bool = False,
    strip_prefix: bool = True,
    force: bool = False,
) -> list[Path]:
    """Discover and download DuckDB databases from Hugging Face Hub."""
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing DuckDB asset discovery from %s (%s)...", repo_id, remote_dir)
    api = HfApi()

    files = list(
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=remote_dir,
            recursive=True,
        )
    )

    # Filter files by pattern and size attribute
    duckdb_files: list[Any] = [
        f
        for f in files
        if hasattr(f, "size") and fnmatch.fnmatch(f.path, f"*{file_pattern.lstrip('*')}")
    ]

    if environment_only:
        duckdb_files = [f for f in duckdb_files if "environment" in f.path]

    if not duckdb_files:
        raise RuntimeError(
            f"No DuckDB files matching '{file_pattern}' found under '{remote_dir}' in repo '{repo_id}'."
        )

    logger.info("Discovered %d DuckDB files in remote repository.", len(duckdb_files))
    if limit is not None and limit > 0:
        duckdb_files = duckdb_files[:limit]
        logger.info("Applying download limit: downloading first %d files.", limit)

    downloaded_paths: list[Path] = []
    skipped_count = 0
    download_count = 0

    norm_remote_dir = remote_dir.strip("/")

    for i, file_info in enumerate(duckdb_files, 1):
        remote_path = file_info.path
        remote_size = getattr(file_info, "size", None)

        if strip_prefix and remote_path.startswith(norm_remote_dir):
            rel_path = remote_path[len(norm_remote_dir) :].lstrip("/")
        else:
            rel_path = remote_path

        local_target_file = target_dir / rel_path
        local_target_file.parent.mkdir(parents=True, exist_ok=True)

        if (
            not force
            and local_target_file.exists()
            and (remote_size is None or local_target_file.stat().st_size == remote_size)
        ):
            logger.info(
                "[%d/%d] Skipping existing file: %s (%s)",
                i,
                len(duckdb_files),
                rel_path,
                f"{local_target_file.stat().st_size / (1024 * 1024):.1f} MB",
            )
            downloaded_paths.append(local_target_file)
            skipped_count += 1
            continue

        logger.info(
            "[%d/%d] Downloading %s (%.1f MB)...",
            i,
            len(duckdb_files),
            remote_path,
            (remote_size or 0) / (1024 * 1024),
        )

        cached_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=remote_path,
        )

        shutil.copy2(cached_file, local_target_file)
        downloaded_paths.append(local_target_file)
        download_count += 1

    logger.info("============================================================")
    logger.info("SPIDER 2.0-DBT DUCKDB ASSET DOWNLOAD SUMMARY")
    logger.info("============================================================")
    logger.info("Target directory : %s", target_dir)
    logger.info("Total discovered : %d", len(duckdb_files))
    logger.info("Total downloaded : %d", download_count)
    logger.info("Total skipped    : %d (already cached locally)", skipped_count)
    logger.info("============================================================")

    return downloaded_paths


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    download_spider2_dbt_assets(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        remote_dir=args.remote_dir,
        target_dir=args.target_dir,
        file_pattern=args.file_pattern,
        limit=args.limit,
        environment_only=args.environment_only,
        strip_prefix=args.strip_prefix,
        force=args.force,
    )


if __name__ == "__main__":
    main()
