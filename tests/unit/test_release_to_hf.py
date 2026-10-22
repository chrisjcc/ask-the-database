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
