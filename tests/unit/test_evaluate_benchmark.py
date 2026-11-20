"""Unit tests for NL2SQLPipeline and comprehensive benchmark evaluation harness."""

from pathlib import Path
from typing import Any

import pytest

from ask_the_database.inference.pipeline import NL2SQLPipeline
from scripts.evaluate_benchmark import run_benchmark_evaluation


def test_nl2sql_pipeline_interactive_rollout(sample_sqlite_db: Path) -> None:
    """Test NL2SQLPipeline runs multi-turn interactive trajectory and captures query."""
    pipeline = NL2SQLPipeline(max_turns=4)

    gold_sql = "SELECT name FROM users ORDER BY age DESC LIMIT 1;"
    traj = pipeline.run_interactive(
        question="Who is the oldest user?",
        db_path=str(sample_sqlite_db),
        db_id="company",
        gold_sql=gold_sql,
    )

    assert len(traj.steps) > 0
    assert traj.final_execution_success is True
    assert traj.final_query != ""
    assert traj.question == "Who is the oldest user?"

    dump = traj.to_dict()
    assert dump["db_id"] == "company"
    assert dump["final_execution_success"] is True


def test_run_benchmark_evaluation_with_custom_samples(
    sample_sqlite_db: Path, tmp_path: Path
) -> None:
    """Test evaluation harness calculates TS, EX, VA, MRR across custom samples."""
    out_dir = tmp_path / "eval_results"
    summary_file = out_dir / "evaluation_summary.json"

    gold_sql = "SELECT name, age FROM users WHERE age > 25 ORDER BY age ASC;"
    custom_samples = [
        {
            "question": "Which users are older than 25?",
            "gold_sql": gold_sql,
            "db_id": "company",
            "db_path": str(sample_sqlite_db),
        }
    ]

    cfg = {
        "evaluation": {
            "model_path": None,
            "max_turns": 3,
            "benchmarks": ["spider", "spider_robustness_test"],
            "custom_samples": {
                "spider": custom_samples,
                "spider_robustness_test": custom_samples,
            },
            "output_dir": str(out_dir),
            "summary_file": str(summary_file),
        }
    }

    leaderboard = run_benchmark_evaluation(cfg)

    assert "spider" in leaderboard
    assert "spider_robustness_test" in leaderboard
    assert leaderboard["spider"]["total_samples"] == 1
    assert leaderboard["spider"]["valid_execution_rate_va"] == 100.0
    assert leaderboard["spider"]["execution_accuracy_ex"] == 100.0
    assert leaderboard["spider"]["test_suite_accuracy_ts"] == 100.0

    assert summary_file.exists()


def test_nl2sql_pipeline_resolves_rlvr_policy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test NL2SQLPipeline resolves underlying checkpoint from rlvr_policy_config.json."""
    import json
    from unittest.mock import MagicMock

    import peft
    import transformers

    # Mock an SFT adapter directory
    sft_dir = tmp_path / "models" / "sft" / "harness_zero"
    sft_dir.mkdir(parents=True)
    (sft_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "mock_base", "peft_type": "LORA"})
    )

    # Mock an RLVR directory that only has rlvr_policy_config.json
    rlvr_dir = tmp_path / "models" / "rlvr" / "harness_zero_grpo"
    rlvr_dir.mkdir(parents=True)
    (rlvr_dir / "rlvr_policy_config.json").write_text(
        json.dumps(
            {
                "adapter": "SkyRLSQLEnvironment",
                "model_path": str(sft_dir),
                "max_turns": 5,
                "steps": 250,
            }
        )
    )

    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_peft = MagicMock()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: mock_tok)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: mock_model
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *a, **kw: mock_peft)

    pipe = NL2SQLPipeline(model_path=str(rlvr_dir))

    assert pipe.model_path == str(rlvr_dir)
    assert pipe.tokenizer is mock_tok
    assert pipe.model is mock_peft
    assert pipe.load_error is None


def test_nl2sql_pipeline_resolves_rlvr_hub_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test NL2SQLPipeline resolves underlying checkpoint pointing to a Hugging Face Hub repo ID."""
    import json
    from unittest.mock import MagicMock

    import transformers

    monkeypatch.setenv("FORCE_CPU_MODEL_LOAD", "1")

    # Mock an RLVR directory that only has rlvr_policy_config.json pointing to Hub ID
    rlvr_dir = tmp_path / "models" / "rlvr" / "c1_pure_grpo"
    rlvr_dir.mkdir(parents=True)
    (rlvr_dir / "rlvr_policy_config.json").write_text(
        json.dumps(
            {
                "adapter": "SkyRLSQLEnvironment",
                "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "max_turns": 5,
                "steps": 250,
            }
        )
    )

    mock_tok = MagicMock()
    mock_model = MagicMock()

    loaded_names: list[str] = []

    def mock_causal_lm(model_id: str, *a: Any, **kw: Any) -> MagicMock:
        loaded_names.append(model_id)
        return mock_model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: mock_tok)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", mock_causal_lm)

    pipe = NL2SQLPipeline(model_path=str(rlvr_dir))

    assert pipe.model_path == str(rlvr_dir)
    assert pipe.tokenizer is mock_tok
    assert pipe.model is mock_model
    assert pipe.load_error is None
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in loaded_names


def test_nl2sql_pipeline_resolves_rlvr_fallback_sft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test NL2SQLPipeline falls back to c0_harness_zero when models/sft/final is non-existent."""
    import json
    from unittest.mock import MagicMock

    import peft
    import transformers

    # Mock an SFT adapter directory at c0_harness_zero
    sft_dir = tmp_path / "models" / "sft" / "c0_harness_zero"
    sft_dir.mkdir(parents=True)
    (sft_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "mock_base", "peft_type": "LORA"})
    )

    # Mock an RLVR directory that points to models/sft/final (which doesn't exist)
    rlvr_dir = tmp_path / "models" / "rlvr" / "c1_pure_grpo"
    rlvr_dir.mkdir(parents=True)
    (rlvr_dir / "rlvr_policy_config.json").write_text(
        json.dumps(
            {
                "adapter": "SkyRLSQLEnvironment",
                "model_path": "models/sft/final",
                "max_turns": 5,
                "steps": 250,
            }
        )
    )

    # Monkeypatch the fallback search to use tmp_path
    monkeypatch.chdir(tmp_path)

    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_peft = MagicMock()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: mock_tok)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: mock_model
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *a, **kw: mock_peft)

    pipe = NL2SQLPipeline(model_path=str(rlvr_dir))

    assert pipe.model_path == str(rlvr_dir)
    assert pipe.model is mock_peft
    assert pipe.load_error is None
