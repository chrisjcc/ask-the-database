"""Unit tests for SFT dataset formatting, LoRA configuration, and training pipeline."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip(
    "torch",
    reason="PyTorch is required for training and adapter tests. Install via 'make install' or 'pip install -e .[training]'.",
)

from typing import Any
from unittest.mock import MagicMock, patch

import torch

from ask_the_database.training.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    extract_sql_from_completion,
    format_chatml_messages,
    format_prompt_for_generation,
    load_sft_dataset,
)
from ask_the_database.training.sft_trainer import (
    RobustSFTTrainer,
    SFTWarmupPipeline,
    validate_lora_config,
    validate_quantization_config,
)


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
                {
                    "question": "Q",
                    "gold_sql": "SELECT 1",
                    "schema_ddl": "CREATE TABLE t (x INT);",
                }
            )
            + "\n"
        )
    with (data_dir / "val.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "question": "Q",
                    "gold_sql": "SELECT 1",
                    "schema_ddl": "CREATE TABLE t (x INT);",
                }
            )
            + "\n"
        )

    # Call main with cfg
    main(cfg)

    out_dir = tmp_path / "models" / "sft" / "final"
    assert (out_dir / "adapter_config.json").is_file()
    assert (out_dir / "adapter_model.safetensors").is_file()


def test_standard_lora_config_loaded(tmp_path: Path) -> None:
    """Verify standard LoRA configuration is loaded correctly with scaling alpha / r."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {
                    "enabled": True,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "use_rslora": False,
                    "target_modules": "all-linear",
                    "bias": "none",
                },
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is not None
    assert lora_cfg.r == 16
    assert lora_cfg.lora_alpha == 32
    assert lora_cfg.use_rslora is False
    assert lora_cfg.target_modules == "all-linear"

    metadata = pipeline.get_lora_metadata(lora_cfg)
    assert metadata["lora_enabled"] is True
    assert metadata["lora_type"] == "standard_lora"
    assert metadata["use_rslora"] is False
    assert metadata["scaling_factor"] == 2.0


def test_rslora_config_loaded(tmp_path: Path) -> None:
    """Verify rsLoRA configuration is loaded correctly with scaling alpha / sqrt(r)."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {
                    "enabled": True,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "use_rslora": True,
                    "target_modules": ["q_proj", "v_proj"],
                    "bias": "none",
                },
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is not None
    assert lora_cfg.r == 16
    assert lora_cfg.lora_alpha == 32
    assert lora_cfg.use_rslora is True
    assert lora_cfg.target_modules == {"q_proj", "v_proj"}

    metadata = pipeline.get_lora_metadata(lora_cfg)
    assert metadata["lora_enabled"] is True
    assert metadata["lora_type"] == "rslora"
    assert metadata["use_rslora"] is True
    assert metadata["scaling_factor"] == 8.0


def test_lora_disabled_handled(tmp_path: Path) -> None:
    """Verify that disabling LoRA returns None and sets metadata to full_finetuning."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {
                    "enabled": False,
                },
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is None

    metadata = pipeline.get_lora_metadata(lora_cfg)
    assert metadata["lora_enabled"] is False
    assert metadata["lora_type"] == "full_finetuning"
    assert metadata["r"] is None
    assert metadata["use_rslora"] is False


def test_missing_optional_parameters_defaults(tmp_path: Path) -> None:
    """Verify omitting optional parameters receives intended defaults."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {},
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is not None
    assert lora_cfg.r == 16
    assert lora_cfg.lora_alpha == 32
    assert lora_cfg.use_rslora is False
    assert lora_cfg.target_modules == "all-linear"
    assert lora_cfg.bias == "none"

    cfg_none = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
            },
        }
    )
    pipeline_none = SFTWarmupPipeline(cfg_none)
    lora_cfg_none = pipeline_none.build_lora_config()
    assert lora_cfg_none is not None
    assert lora_cfg_none.r == 16
    assert lora_cfg_none.use_rslora is False


def test_disabled_lora_ignores_invalid_lora_params(tmp_path: Path) -> None:
    """Verify that when LoRA is disabled, invalid LoRA params do not block full fine-tuning."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {
                    "enabled": False,
                    "r": -99,
                    "lora_alpha": -1,
                    "target_modules": "",
                },
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is None


@pytest.mark.parametrize("invalid_r", [0, -1, -16, True, "16"])
def test_invalid_rank_rejected(invalid_r: Any) -> None:
    """Verify invalid rank values are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": invalid_r,
            "lora_alpha": 32,
        }
    )
    with pytest.raises(ValueError, match="Invalid LoRA rank 'r'"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_alpha", [0, -1, -32, False, "32"])
def test_invalid_alpha_rejected(invalid_alpha: Any) -> None:
    """Verify invalid alpha values are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": invalid_alpha,
        }
    )
    with pytest.raises(ValueError, match="Invalid LoRA alpha 'lora_alpha'"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_dropout", [-0.1, 1.0, 1.5, True, False])
def test_invalid_dropout_rejected(invalid_dropout: Any) -> None:
    """Verify dropout outside [0, 1) is rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": invalid_dropout,
        }
    )
    with pytest.raises(ValueError, match="Invalid LoRA dropout 'lora_dropout'"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_use_rslora", ["true", "false", 1, 0, None, [True]])
def test_invalid_use_rslora_rejected(invalid_use_rslora: Any) -> None:
    """Verify non-boolean use_rslora values are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": 32,
            "use_rslora": invalid_use_rslora,
        }
    )
    with pytest.raises(ValueError, match="Invalid rsLoRA flag 'use_rslora'"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_target_modules", ["", "   ", [], [""], 123, None])
def test_invalid_target_modules_rejected(invalid_target_modules: Any) -> None:
    """Verify invalid target_modules are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": invalid_target_modules,
        }
    )
    with pytest.raises(ValueError, match="Invalid.*target_modules"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_bias", ["invalid_bias", 123, True])
def test_invalid_bias_rejected(invalid_bias: Any) -> None:
    """Verify unsupported bias values are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": 32,
            "bias": invalid_bias,
        }
    )
    with pytest.raises(ValueError, match="Invalid LoRA bias setting 'bias'"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("invalid_modules_to_save", ["lm_head", [123], [""]])
def test_invalid_modules_to_save_rejected(invalid_modules_to_save: Any) -> None:
    """Verify invalid modules_to_save configurations are rejected with ValueError."""
    cfg = OmegaConf.create(
        {
            "r": 16,
            "lora_alpha": 32,
            "modules_to_save": invalid_modules_to_save,
        }
    )
    with pytest.raises(ValueError, match="Invalid.*modules_to_save"):
        validate_lora_config(cfg)


def test_peft_parameter_propagation(tmp_path: Path) -> None:
    """Verify all configured parameters propagate accurately into LoraConfig."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "lora": {
                    "enabled": True,
                    "r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.1,
                    "use_rslora": True,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "bias": "lora_only",
                    "modules_to_save": ["lm_head"],
                    "init_lora_weights": True,
                },
            },
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    lora_cfg = pipeline.build_lora_config()
    assert lora_cfg is not None
    assert lora_cfg.r == 32
    assert lora_cfg.lora_alpha == 64
    assert lora_cfg.lora_dropout == 0.1
    assert lora_cfg.use_rslora is True
    assert lora_cfg.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert lora_cfg.bias == "lora_only"
    assert lora_cfg.modules_to_save == ["lm_head"]
    assert lora_cfg.init_lora_weights is True


def test_sft_trainer_receives_peft_config(tmp_path: Path) -> None:
    """Verify SFTTrainer receives peft_config for rsLoRA training."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
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
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "use_rslora": True,
                    "target_modules": ["q_proj"],
                    "bias": "none",
                },
            },
            "model": {"name": "dummy"},
            "paths": {"processed_dir": str(tmp_path)},
        }
    )

    with patch("ask_the_database.training.sft_trainer.SFTTrainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        pipeline = SFTWarmupPipeline(cfg)
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_ds = MagicMock()

        pipeline.train(model=mock_model, tokenizer=mock_tok, train_dataset=mock_ds)

        mock_trainer_cls.assert_called_once()
        _, kwargs = mock_trainer_cls.call_args
        peft_cfg = kwargs.get("peft_config")
        assert peft_cfg is not None
        assert peft_cfg.use_rslora is True
        assert peft_cfg.r == 8
        assert peft_cfg.lora_alpha == 16

        metadata_file = tmp_path / "out" / "lora_metadata.json"
        assert metadata_file.is_file()
        metadata = json.loads(metadata_file.read_text())
        assert metadata["lora_type"] == "rslora"
        assert metadata["use_rslora"] is True


def test_sft_trainer_full_finetuning_receives_none(tmp_path: Path) -> None:
    """Verify SFTTrainer receives peft_config=None when LoRA is disabled."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out_full"),
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
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
                    "enabled": False,
                },
            },
            "model": {"name": "dummy"},
            "paths": {"processed_dir": str(tmp_path)},
        }
    )

    with patch("ask_the_database.training.sft_trainer.SFTTrainer") as mock_trainer_cls:
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        pipeline = SFTWarmupPipeline(cfg)
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_ds = MagicMock()

        pipeline.train(model=mock_model, tokenizer=mock_tok, train_dataset=mock_ds)

        mock_trainer_cls.assert_called_once()
        _, kwargs = mock_trainer_cls.call_args
        assert kwargs.get("peft_config") is None

        metadata_file = tmp_path / "out_full" / "lora_metadata.json"
        assert metadata_file.is_file()
        metadata = json.loads(metadata_file.read_text())
        assert metadata["lora_enabled"] is False
        assert metadata["lora_type"] == "full_finetuning"


def test_dry_run_artifacts_rslora(tmp_path: Path) -> None:
    """Verify dry-run mode produces valid adapter checkpoint files for rsLoRA."""
    from scripts.train_sft import main

    cfg = OmegaConf.create(
        {
            "model": {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "trust_remote_code": True},
            "training": {
                "output_dir": str(tmp_path / "models" / "sft" / "rslora"),
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
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
                    "enabled": True,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "use_rslora": True,
                    "target_modules": "all-linear",
                    "bias": "none",
                },
            },
            "paths": {"processed_dir": str(tmp_path / "data")},
            "dry_run": True,
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"question": "Q", "gold_sql": "SELECT 1", "schema_ddl": "CREATE TABLE t (x INT);"}
            )
            + "\n"
        )

    main(cfg)

    out_dir = tmp_path / "models" / "sft" / "rslora"
    adapter_cfg_file = out_dir / "adapter_config.json"
    assert adapter_cfg_file.is_file()
    adapter_data = json.loads(adapter_cfg_file.read_text())
    assert adapter_data["use_rslora"] is True
    assert adapter_data["r"] == 16

    metadata_file = out_dir / "lora_metadata.json"
    assert metadata_file.is_file()
    meta_data = json.loads(metadata_file.read_text())
    assert meta_data["lora_type"] == "rslora"
    assert meta_data["use_rslora"] is True


def test_dry_run_artifacts_full_finetuning(tmp_path: Path) -> None:
    """Verify dry-run mode produces appropriate metadata when LoRA is disabled."""
    from scripts.train_sft import main

    cfg = OmegaConf.create(
        {
            "model": {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "trust_remote_code": True},
            "training": {
                "output_dir": str(tmp_path / "models" / "sft" / "full_ft"),
                "learning_rate": 2e-5,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
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
                    "enabled": False,
                },
            },
            "paths": {"processed_dir": str(tmp_path / "data")},
            "dry_run": True,
        }
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"question": "Q", "gold_sql": "SELECT 1", "schema_ddl": "CREATE TABLE t (x INT);"}
            )
            + "\n"
        )

    main(cfg)

    out_dir = tmp_path / "models" / "sft" / "full_ft"
    assert (out_dir / "config.json").is_file()

    metadata_file = out_dir / "lora_metadata.json"
    assert metadata_file.is_file()
    meta_data = json.loads(metadata_file.read_text())
    assert meta_data["lora_enabled"] is False
    assert meta_data["lora_type"] == "full_finetuning"


# -----------------------------------------------------------------------------
# Quantization (QLoRA) and Device Map Tests
# -----------------------------------------------------------------------------


def test_validate_quantization_config_disabled() -> None:
    """Verify validate_quantization_config succeeds when disabled or None."""
    validate_quantization_config(None)
    validate_quantization_config({"enabled": False})
    validate_quantization_config({"enabled": "false"})
    validate_quantization_config(OmegaConf.create({"enabled": False, "load_in_4bit": "invalid"}))


def test_validate_quantization_config_missing_bnb() -> None:
    """Verify ImportError is raised when quantization is enabled but bitsandbytes is missing."""
    with (
        patch(
            "ask_the_database.training.sft_trainer.is_bitsandbytes_available", return_value=False
        ),
        pytest.raises(ImportError, match="bitsandbytes' is not installed or available"),
    ):
        validate_quantization_config({"enabled": True})


def test_validate_quantization_config_valid() -> None:
    """Verify validate_quantization_config passes with valid 4-bit and 8-bit configs."""
    with patch(
        "ask_the_database.training.sft_trainer.is_bitsandbytes_available", return_value=True
    ):
        # 4-bit default
        validate_quantization_config(
            {
                "enabled": True,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            }
        )
        # 4-bit fp4 float16
        validate_quantization_config(
            {
                "enabled": "true",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_compute_dtype": "float16",
            }
        )
        # 8-bit
        validate_quantization_config(
            {
                "enabled": True,
                "load_in_4bit": False,
                "load_in_8bit": True,
            }
        )


def test_validate_quantization_config_invalid_params() -> None:
    """Verify validation errors for conflicting or invalid quantization settings."""
    with patch(
        "ask_the_database.training.sft_trainer.is_bitsandbytes_available", return_value=True
    ):
        # Invalid enabled flag
        with pytest.raises(ValueError, match="Invalid quantization 'enabled' flag"):
            validate_quantization_config({"enabled": "maybe"})

        # Conflicting 4-bit and 8-bit
        with pytest.raises(ValueError, match="Conflicting quantization settings"):
            validate_quantization_config(
                {"enabled": True, "load_in_4bit": True, "load_in_8bit": True}
            )

        # Neither 4-bit nor 8-bit
        with pytest.raises(ValueError, match="neither 'load_in_4bit' nor 'load_in_8bit' is True"):
            validate_quantization_config(
                {"enabled": True, "load_in_4bit": False, "load_in_8bit": False}
            )

        # Invalid quant type
        with pytest.raises(ValueError, match="Invalid 'bnb_4bit_quant_type'"):
            validate_quantization_config(
                {"enabled": True, "load_in_4bit": True, "bnb_4bit_quant_type": "int4"}
            )

        # Invalid compute dtype
        with pytest.raises(ValueError, match="Invalid 'bnb_4bit_compute_dtype'"):
            validate_quantization_config(
                {"enabled": True, "load_in_4bit": True, "bnb_4bit_compute_dtype": "int8"}
            )

        # Non-boolean load_in_4bit
        with pytest.raises(ValueError, match="Invalid 'load_in_4bit' flag"):
            validate_quantization_config({"enabled": True, "load_in_4bit": "yes"})

        # Non-boolean double quant
        with pytest.raises(ValueError, match="Invalid 'bnb_4bit_use_double_quant'"):
            validate_quantization_config(
                {"enabled": True, "load_in_4bit": True, "bnb_4bit_use_double_quant": 123}
            )


def test_build_quantization_config() -> None:
    """Verify build_quantization_config creates appropriate BitsAndBytesConfig."""
    # Disabled or missing
    pipeline_none = SFTWarmupPipeline(OmegaConf.create({"training": {}, "model": {}}))
    assert pipeline_none.build_quantization_config() is None

    pipeline_disabled = SFTWarmupPipeline(
        OmegaConf.create({"training": {"quantization": {"enabled": False}}, "model": {}})
    )
    assert pipeline_disabled.build_quantization_config() is None

    with patch(
        "ask_the_database.training.sft_trainer.is_bitsandbytes_available", return_value=True
    ):
        # 8-bit
        pipeline_8bit = SFTWarmupPipeline(
            OmegaConf.create(
                {
                    "training": {
                        "quantization": {
                            "enabled": True,
                            "load_in_4bit": False,
                            "load_in_8bit": True,
                        }
                    },
                    "model": {},
                }
            )
        )
        bnb_8bit = pipeline_8bit.build_quantization_config()
        assert bnb_8bit is not None
        assert bnb_8bit.load_in_8bit is True
        assert bnb_8bit.load_in_4bit is False

        # 4-bit nf4
        pipeline_4bit = SFTWarmupPipeline(
            OmegaConf.create(
                {
                    "training": {
                        "quantization": {
                            "enabled": True,
                            "load_in_4bit": True,
                            "load_in_8bit": False,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_compute_dtype": "bfloat16",
                        }
                    },
                    "model": {},
                }
            )
        )
        bnb_4bit = pipeline_4bit.build_quantization_config()
        assert bnb_4bit is not None
        assert bnb_4bit.load_in_4bit is True
        assert bnb_4bit.load_in_8bit is False
        assert bnb_4bit.bnb_4bit_quant_type == "nf4"
        assert bnb_4bit.bnb_4bit_use_double_quant is True


def test_qlora_metadata_tracking() -> None:
    """Verify metadata captures QLoRA configurations for both Standard LoRA and rsLoRA."""
    quant_cfg = OmegaConf.create(
        {
            "enabled": True,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    )

    from peft import LoraConfig, TaskType

    peft_lora = LoraConfig(r=16, lora_alpha=32, use_rslora=True, task_type=TaskType.CAUSAL_LM)
    meta = SFTWarmupPipeline.get_lora_metadata(peft_lora, quant_cfg)
    assert meta["lora_type"] == "qlora_rslora"
    assert meta["quantization_enabled"] is True
    assert meta["quantization_type"] == "4bit_nf4"
    assert meta["quantization_compute_dtype"] == "bfloat16"

    # Standard LoRA + 4-bit
    peft_standard = LoraConfig(r=16, lora_alpha=32, use_rslora=False, task_type=TaskType.CAUSAL_LM)
    meta_std = SFTWarmupPipeline.get_lora_metadata(peft_standard, quant_cfg)
    assert meta_std["lora_type"] == "qlora_standard"

    # Full fine-tuning + 8-bit
    quant_8bit = OmegaConf.create({"enabled": True, "load_in_8bit": True, "load_in_4bit": False})
    meta_ft = SFTWarmupPipeline.get_lora_metadata(None, quant_8bit)
    assert meta_ft["lora_type"] == "quantized_full_finetuning"
    assert meta_ft["quantization_type"] == "8bit"


def test_load_model_device_map_and_quantization() -> None:
    """Verify load_model sets single-device map to prevent multi-GPU layer splitting and invokes kbit preparation."""
    cfg = OmegaConf.create(
        {
            "model": {"name": "dummy_model", "trust_remote_code": True},
            "training": {
                "quantization": {
                    "enabled": True,
                    "load_in_4bit": True,
                    "load_in_8bit": False,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                },
                "gradient_checkpointing": True,
            },
        }
    )

    pipeline = SFTWarmupPipeline(cfg)

    with (
        patch("ask_the_database.training.sft_trainer.is_bitsandbytes_available", return_value=True),
        patch("ask_the_database.training.sft_trainer.torch.cuda.is_available", return_value=True),
        patch("ask_the_database.training.sft_trainer.torch.cuda.device_count", return_value=2),
        patch("ask_the_database.training.sft_trainer.torch.cuda.current_device", return_value=0),
        patch(
            "ask_the_database.training.sft_trainer.AutoModelForCausalLM.from_pretrained"
        ) as mock_from_pretrained,
        patch(
            "ask_the_database.training.sft_trainer.prepare_model_for_kbit_training"
        ) as mock_prep_kbit,
    ):
        mock_raw_model = MagicMock()
        mock_from_pretrained.return_value = mock_raw_model
        mock_prep_kbit.return_value = mock_raw_model

        loaded_model = pipeline.load_model("dummy_model")
        assert loaded_model is mock_raw_model

        mock_from_pretrained.assert_called_once()
        _, kwargs = mock_from_pretrained.call_args
        # Crucial check: device_map must be single device {"": 0}, NOT "auto"
        assert kwargs["device_map"] == {"": 0}
        assert kwargs.get("quantization_config") is not None

        mock_prep_kbit.assert_called_once_with(mock_raw_model, use_gradient_checkpointing=True)


def test_robust_sft_trainer_cross_device_recovery() -> None:
    """Verify RobustSFTTrainer catches multi-GPU cross-device tensor mismatch and retries cleanly."""
    mock_model = MagicMock()
    mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    trainer = RobustSFTTrainer.__new__(RobustSFTTrainer)

    call_count = 0

    def mock_super_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        nonlocal call_count
        call_count += 1
        if num_items_in_batch is not None:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!"
            )
        return torch.tensor(0.42)

    with patch.object(
        RobustSFTTrainer.__bases__[0], "compute_loss", side_effect=mock_super_compute_loss
    ):
        loss = trainer.compute_loss(mock_model, mock_inputs, num_items_in_batch=16)
        assert call_count == 2
        assert torch.isclose(loss, torch.tensor(0.42))


def test_robust_sft_trainer_reraises_other_errors() -> None:
    """Verify RobustSFTTrainer re-raises unrelated runtime errors without suppressing them."""
    mock_model = MagicMock()
    mock_inputs = {}
    trainer = RobustSFTTrainer.__new__(RobustSFTTrainer)

    with (
        patch.object(
            RobustSFTTrainer.__bases__[0],
            "compute_loss",
            side_effect=RuntimeError("CUDA out of memory"),
        ),
        pytest.raises(RuntimeError, match="CUDA out of memory"),
    ):
        trainer.compute_loss(mock_model, mock_inputs, num_items_in_batch=16)


def test_robust_sft_trainer_clamps_n_gpu_in_single_process(tmp_path: Path) -> None:
    """Verify RobustSFTTrainer clamps n_gpu to 1 in single process to prevent nn.DataParallel error in TRL."""
    from trl.trainer.sft_config import SFTConfig

    training_args = SFTConfig(
        output_dir=str(tmp_path / "out"), bf16=False, fp16=False, use_cpu=True
    )
    training_args._n_gpu = 2

    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.parameters.return_value = [torch.tensor([1.0])]

    with (
        patch.object(
            RobustSFTTrainer.__bases__[0], "__init__", return_value=None
        ) as mock_super_init,
        patch.dict("os.environ", {"WORLD_SIZE": "1", "LOCAL_RANK": "-1"}, clear=False),
    ):
        RobustSFTTrainer(model=mock_model, args=training_args)
        assert training_args.n_gpu == 1
        mock_super_init.assert_called_once()


def test_build_training_args_clamps_n_gpu(tmp_path: Path) -> None:
    """Verify build_training_args constructs SFTConfig with clamped n_gpu in single process mode."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "learning_rate": 2e-5,
            },
            "model": {},
        }
    )
    pipeline = SFTWarmupPipeline(cfg)
    with patch.dict("os.environ", {"WORLD_SIZE": "1", "LOCAL_RANK": "-1"}, clear=False):
        args = pipeline.build_training_args()
        assert args.n_gpu == 1


def test_build_training_args_precision_resolution(tmp_path: Path) -> None:
    """Verify build_training_args prefers bf16 on Ampere+ and avoids GradScaler conflict."""
    cfg = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "learning_rate": 2e-5,
                "bf16": True,
                "fp16": False,
            },
            "model": {},
        }
    )
    pipeline = SFTWarmupPipeline(cfg)

    # When bf16 is supported on CUDA, bf16 must be True and fp16 must be False
    with (
        patch("ask_the_database.training.sft_trainer._is_bf16_supported", return_value=True),
        patch("ask_the_database.training.sft_trainer.torch.cuda.is_available", return_value=True),
        patch("ask_the_database.training.sft_trainer.SFTConfig") as mock_sft_config,
    ):
        mock_instance = MagicMock()
        mock_sft_config.return_value = mock_instance
        pipeline.build_training_args()
        _, kwargs = mock_sft_config.call_args
        assert kwargs["bf16"] is True
        assert kwargs["fp16"] is False

    # When on CPU (no bf16 supported), both must be False
    with (
        patch("ask_the_database.training.sft_trainer._is_bf16_supported", return_value=False),
        patch("ask_the_database.training.sft_trainer.torch.cuda.is_available", return_value=False),
        patch("ask_the_database.training.sft_trainer.SFTConfig") as mock_sft_config,
    ):
        mock_instance = MagicMock()
        mock_sft_config.return_value = mock_instance
        pipeline.build_training_args()
        _, kwargs = mock_sft_config.call_args
        assert kwargs["bf16"] is False
        assert kwargs["fp16"] is False

    # When fp16 is explicitly requested, fp16 must be True and bf16 False
    cfg_fp16 = OmegaConf.create(
        {
            "training": {
                "output_dir": str(tmp_path / "out"),
                "learning_rate": 2e-5,
                "bf16": False,
                "fp16": True,
            },
            "model": {},
        }
    )
    pipeline_fp16 = SFTWarmupPipeline(cfg_fp16)
    with (
        patch("ask_the_database.training.sft_trainer._is_bf16_supported", return_value=True),
        patch("ask_the_database.training.sft_trainer.torch.cuda.is_available", return_value=True),
        patch("ask_the_database.training.sft_trainer.SFTConfig") as mock_sft_config,
    ):
        mock_instance = MagicMock()
        mock_sft_config.return_value = mock_instance
        pipeline_fp16.build_training_args()
        _, kwargs = mock_sft_config.call_args
        assert kwargs["bf16"] is False
        assert kwargs["fp16"] is True
