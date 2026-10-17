"""Unit tests for SFT dataset formatting, LoRA configuration, and training pipeline."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip(
    "torch",
    reason="PyTorch is required for training and adapter tests. Install via 'make install' or 'pip install -e .[training]'.",
)

from ask_the_database.training.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    extract_sql_from_completion,
    format_chatml_messages,
    format_prompt_for_generation,
    load_sft_dataset,
)
from ask_the_database.training.sft_trainer import SFTWarmupPipeline


def test_format_chatml_messages_with_gold() -> None:
    """Verify ChatML conversation messages formatting with gold SQL."""
    messages = format_chatml_messages(
        question="What is the total count?",
        schema_ddl="CREATE TABLE t (id INT);",
        gold_sql="SELECT COUNT(*) FROM t;",
    )
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert "CREATE TABLE t" in messages[0]["content"]
    assert DEFAULT_SYSTEM_PROMPT in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the total count?"

    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "```sql\nSELECT COUNT(*) FROM t;\n```"


def test_format_chatml_messages_without_gold() -> None:
    """Verify ChatML messages formatting for generation without gold SQL."""
    messages = format_chatml_messages(
        question="Show me names",
        schema_ddl="CREATE TABLE users (name TEXT);",
        gold_sql=None,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_format_prompt_for_generation() -> None:
    """Verify raw ChatML generation prompt formatting."""
    prompt = format_prompt_for_generation(
        question="How many users?",
        schema_ddl="CREATE TABLE users (id INT);",
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "How many users?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_extract_sql_from_completion() -> None:
    """Verify SQL extraction from code blocks and raw text."""
    # Standard markdown block
    text1 = "Here is the SQL:\n```sql\nSELECT id FROM users WHERE age > 20;\n```"
    assert extract_sql_from_completion(text1) == "SELECT id FROM users WHERE age > 20;"

    # Case-insensitive block
    text2 = "```SQL\nSELECT 1;\n```"
    assert extract_sql_from_completion(text2) == "SELECT 1;"

    # Generic code block
    text3 = "```\nSELECT * FROM t\n```"
    assert extract_sql_from_completion(text3) == "SELECT * FROM t"

    # Raw text without block
    text4 = "SELECT count(*) FROM head;"
    assert extract_sql_from_completion(text4) == "SELECT count(*) FROM head;"


def test_load_sft_dataset_file_not_found() -> None:
    """Verify load_sft_dataset raises FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        load_sft_dataset("nonexistent_path_to_data.jsonl")


def test_load_sft_dataset_success(tmp_path: Path) -> None:
    """Verify loading from JSONL file with max_samples limitation."""
    jsonl_file = tmp_path / "test_data.jsonl"
    samples = [
        {
            "instance_id": f"sample_{i}",
            "db_id": "test_db",
            "question": f"Question {i}",
            "gold_sql": f"SELECT {i}",
            "schema_ddl": "CREATE TABLE t (id INT);",
            "db_path": "/path/to/db.sqlite",
        }
        for i in range(5)
    ]
    with jsonl_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    dataset = load_sft_dataset(str(jsonl_file), max_samples=3)
    assert len(dataset) == 3
    assert dataset[0]["instance_id"] == "sample_0"
    assert len(dataset[0]["messages"]) == 3
    assert dataset[0]["messages"][2]["content"] == "```sql\nSELECT 0\n```"


def test_sft_pipeline_config_and_args(tmp_path: Path) -> None:
    """Verify SFTWarmupPipeline builds valid LoRA config and SFTConfig."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "trust_remote_code": True,
            },
            "training": {
                "output_dir": str(tmp_path / "sft_out"),
                "learning_rate": 3e-5,
                "num_train_epochs": 2,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 5,
                "weight_decay": 0.02,
                "bf16": False,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
                "max_steps": 10,
                "max_seq_length": 1024,
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "paths": {
                "processed_dir": str(tmp_path),
            },
        }
    )

    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg.r == 8
    assert lora_cfg.lora_alpha == 16
    assert lora_cfg.target_modules == {"q_proj", "v_proj"}

    training_args = pipeline.build_training_args()
    assert training_args.output_dir == str(tmp_path / "sft_out")
    assert training_args.learning_rate == 3e-5
    assert training_args.max_steps == 10
    assert training_args.max_length == 1024


def test_dry_run_artifacts(tmp_path: Path) -> None:
    """Verify that dry-run mode produces valid adapter checkpoint files."""
    from scripts.train_sft import main

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "trust_remote_code": True,
            },
            "training": {
                "output_dir": str(tmp_path / "models" / "sft" / "final"),
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 1,
                "weight_decay": 0.01,
                "bf16": False,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
                "max_steps": 1,
                "max_seq_length": 512,
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "paths": {
                "processed_dir": str(tmp_path / "data"),
            },
            "dry_run": True,
        }
    )

    # Create dummy data files
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"question": "Q", "gold_sql": "SELECT 1", "schema_ddl": "CREATE TABLE t (x INT);"}
            )
            + "\n"
        )
    with (data_dir / "val.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"question": "Q", "gold_sql": "SELECT 1", "schema_ddl": "CREATE TABLE t (x INT);"}
            )
            + "\n"
        )

    # Call main with cfg
    main(cfg)

    out_dir = tmp_path / "models" / "sft" / "final"
    assert (out_dir / "adapter_config.json").is_file()
    assert (out_dir / "adapter_model.safetensors").is_file()
