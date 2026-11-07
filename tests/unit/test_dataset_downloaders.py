"""Unit tests for dataset and database downloader scripts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.download_spider1 import parse_args as parse_args_spider1
from scripts.download_spider2_dbt import (
    download_spider2_dbt_assets,
    load_yaml_defaults,
)
from scripts.download_spider2_dbt import (
    parse_args as parse_args_spider2_dbt,
)


class MockRepoFile:
    """Mock file representation returned by HfApi.list_repo_tree."""

    def __init__(self, path: str, size: int = 1024) -> None:
        self.path = path
        self.size = size


def test_spider1_parse_args() -> None:
    """Test argument parsing for download_spider1."""
    with patch(
        "sys.argv", ["download_spider1.py", "--raw-dir", "/tmp/raw", "--databases-dir", "/tmp/dbs"]
    ):
        args = parse_args_spider1()
        assert str(args.raw_dir) == "/tmp/raw"
        assert str(args.databases_dir) == "/tmp/dbs"
        assert args.force is False


def test_spider2_dbt_parse_args_defaults() -> None:
    """Test argument parsing for download_spider2_dbt."""
    with patch("sys.argv", ["download_spider2_dbt.py"]):
        args = parse_args_spider2_dbt()
        assert args.repo_id == "harborframework/harbor-datasets"
        assert args.remote_dir == "datasets/spider2-dbt"
        assert args.file_pattern == "*.duckdb"
        assert args.strip_prefix is True
        assert args.force is False


def test_spider2_dbt_load_yaml_defaults(tmp_path: Path) -> None:
    """Test loading configuration from YAML file."""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text("hf_repo_id: 'custom/repo'\nremote_dir: 'custom/dir'\n")
    defaults = load_yaml_defaults(yaml_file)
    assert defaults.get("hf_repo_id") == "custom/repo"
    assert defaults.get("remote_dir") == "custom/dir"


def test_download_spider2_dbt_assets_success(tmp_path: Path) -> None:
    """Test successful discovery and downloading of DuckDB files with mocks."""
    mock_files = [
        MockRepoFile("datasets/spider2-dbt/task1/environment/dbt/test1.duckdb", size=500),
        MockRepoFile("datasets/spider2-dbt/task2/environment/dbt/test2.duckdb", size=600),
        MockRepoFile("datasets/spider2-dbt/task1/solution/gold.duckdb", size=700),
        MockRepoFile("datasets/spider2-dbt/task1/readme.md", size=100),  # Not a duckdb file
    ]

    target_dir = tmp_path / "raw_dbt"

    fake_cached = tmp_path / "cached.duckdb"
    fake_cached.write_bytes(b"dummy duckdb content")

    with (
        patch("scripts.download_spider2_dbt.HfApi") as mock_hf_api,
        patch("scripts.download_spider2_dbt.hf_hub_download", return_value=str(fake_cached)),
    ):
        api_instance = MagicMock()
        api_instance.list_repo_tree.return_value = mock_files
        mock_hf_api.return_value = api_instance

        # Test download all duckdb files
        downloaded = download_spider2_dbt_assets(
            repo_id="test/repo",
            remote_dir="datasets/spider2-dbt",
            target_dir=target_dir,
            strip_prefix=True,
            force=False,
        )

        assert len(downloaded) == 3
        expected_file1 = target_dir / "task1" / "environment" / "dbt" / "test1.duckdb"
        assert expected_file1.exists()

        # Test idempotence: running again skips existing files
        downloaded_second = download_spider2_dbt_assets(
            repo_id="test/repo",
            remote_dir="datasets/spider2-dbt",
            target_dir=target_dir,
            strip_prefix=True,
            force=False,
        )
        assert len(downloaded_second) == 3


def test_download_spider2_dbt_assets_environment_only(tmp_path: Path) -> None:
    """Test environment-only filter."""
    mock_files = [
        MockRepoFile("datasets/spider2-dbt/task1/environment/dbt/test1.duckdb", size=500),
        MockRepoFile("datasets/spider2-dbt/task1/solution/gold.duckdb", size=700),
    ]

    target_dir = tmp_path / "raw_dbt_env"
    fake_cached = tmp_path / "cached.duckdb"
    fake_cached.write_bytes(b"dummy")

    with (
        patch("scripts.download_spider2_dbt.HfApi") as mock_hf_api,
        patch("scripts.download_spider2_dbt.hf_hub_download", return_value=str(fake_cached)),
    ):
        api_instance = MagicMock()
        api_instance.list_repo_tree.return_value = mock_files
        mock_hf_api.return_value = api_instance

        downloaded = download_spider2_dbt_assets(
            repo_id="test/repo",
            remote_dir="datasets/spider2-dbt",
            target_dir=target_dir,
            environment_only=True,
        )

        assert len(downloaded) == 1
        assert "environment" in str(downloaded[0])


def test_download_spider2_dbt_no_matching_files(tmp_path: Path) -> None:
    """Test RuntimeError when no files match pattern."""
    mock_files = [MockRepoFile("datasets/spider2-dbt/readme.txt", size=100)]

    with patch("scripts.download_spider2_dbt.HfApi") as mock_hf_api:
        api_instance = MagicMock()
        api_instance.list_repo_tree.return_value = mock_files
        mock_hf_api.return_value = api_instance

        with pytest.raises(RuntimeError, match="No DuckDB files"):
            download_spider2_dbt_assets(
                repo_id="test/repo",
                remote_dir="datasets/spider2-dbt",
                target_dir=tmp_path / "empty",
            )
