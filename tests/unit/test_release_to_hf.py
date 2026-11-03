"""Unit tests for Hugging Face card generator script."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.release_to_hf import (
    cli_generate_cards,
    generate_dataset_card,
    generate_model_card,
)


def test_generate_dataset_card() -> None:
    card = generate_dataset_card("test-user/test-dataset")
    assert "test-user/test-dataset" in card
    assert "reviewed_trajectories" in card
    assert "filtered_rollouts" in card
    assert "spider" in card
    assert "K_SQL" in card


def test_generate_model_card() -> None:
    card = generate_model_card(
        model_repo="test-user/test-model",
        model_variant="harness_zero",
        dataset_repo="test-user/test-dataset",
        release_tag="v1.0.0",
    )
    assert "test-user/test-model" in card
    assert "harness_zero" in card
    assert "v1.0.0" in card
    assert "MVE ADJUDICATION DECISION: GO" in card
    assert "text-to-sql" in card
    assert "\\schema" in card
    assert "0.1268 s (126.80 ms)" in card
    assert "Spearman ρ = 1.0000" in card
    assert "10 → 0 misses" in card
    assert "< 2.0 s" in card


def test_cli_generate_cards(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        card_type="all",
        summary_json="",
        dataset_repo="test-user/test-dataset",
        model_repo="test-user/test-model",
        model_variant="harness_zero",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        release_tag="v1.0.0",
    )
    cli_generate_cards(args)

    dataset_card_path = tmp_path / "DATASET_README.md"
    model_card_path = tmp_path / "MODEL_README.md"

    assert dataset_card_path.is_file()
    assert model_card_path.is_file()
    assert "test-user/test-dataset" in dataset_card_path.read_text(encoding="utf-8")
    assert "test-user/test-model" in model_card_path.read_text(encoding="utf-8")


def test_run_deployment_pipeline_checkpoint_filtering(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from scripts.release_to_hf import run_deployment_pipeline

    monkeypatch.setenv("HF_TOKEN", "dummy_token")
    mock_hf_api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda **kw: mock_hf_api)

    weights_dir = tmp_path / "models"
    weights_dir.mkdir()

    # Default: checkpoint-* is ignored
    args_default = argparse.Namespace(
        dataset_repo="test-user/test-dataset",
        model_repo="test-user/test-model",
        model_variant="harness_zero",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        release_tag="",
        summary_json="",
        weights_dir=str(weights_dir),
        dataset_dir=None,
        include_checkpoints=False,
    )
    run_deployment_pipeline(args_default)

    assert mock_hf_api.upload_folder.call_count == 1
    _, kwargs = mock_hf_api.upload_folder.call_args
    assert "checkpoint-*" in kwargs["ignore_patterns"]

    # When include_checkpoints=True: checkpoint-* is NOT ignored
    mock_hf_api.reset_mock()
    args_with_checkpoints = argparse.Namespace(
        dataset_repo="test-user/test-dataset",
        model_repo="test-user/test-model",
        model_variant="harness_zero",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        release_tag="",
        summary_json="",
        weights_dir=str(weights_dir),
        dataset_dir=None,
        include_checkpoints=True,
    )
    run_deployment_pipeline(args_with_checkpoints)

    assert mock_hf_api.upload_folder.call_count == 1
    _, kwargs_included = mock_hf_api.upload_folder.call_args
    assert "checkpoint-*" not in kwargs_included["ignore_patterns"]
