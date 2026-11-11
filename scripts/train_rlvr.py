"""CLI entrypoint for Multi-Turn RLVR (Reinforcement Learning with Verifiable Rewards) Agent Training."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ask_the_database.inference.pipeline import NL2SQLPipeline
from ask_the_database.rewards.multi_turn_reward import MultiTurnRewardEvaluator
from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import RolloutBatch
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment
from ask_the_database.training.adapters.trl_grpo import (
    TRLGRPOAdapter,
    compute_group_normalized_rewards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_rlvr_training(cfg: DictConfig) -> dict[str, Any]:
    """Execute multi-turn RLVR training loop or dry-run validation."""
    logger.info("Initializing Multi-Turn RLVR Agent Training Pipeline...")
    training_cfg = getattr(cfg, "training", cfg)
    reward_cfg = getattr(cfg, "reward", None)
    reward_evaluator = MultiTurnRewardEvaluator(config=reward_cfg)
    env = SkyRLSQLEnvironment(config=training_cfg, reward_evaluator=reward_evaluator)
    rubric = SQLRewardRubric(reward_cfg)
    adapter = TRLGRPOAdapter(cfg, rubric)

    dry_run = bool(getattr(cfg, "dry_run", False) or getattr(training_cfg, "dry_run", False))
    num_rollouts = int(getattr(training_cfg, "num_rollouts_per_prompt", 4))
    max_steps = int(getattr(training_cfg, "max_steps", 250))
    if max_steps <= 0:
        max_steps = 250
    output_dir = str(getattr(training_cfg, "output_dir", "models/rlvr/final"))
    if "sft" in output_dir:
        output_dir = "models/rlvr/final"

    telemetry: dict[str, Any] = {
        "status": "initialized",
        "dry_run": dry_run,
        "num_rollouts": num_rollouts,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "episodes_completed": 0,
        "mean_rewards": [],
        "group_advantages": [],
    }

    if dry_run:
        logger.info(
            "Dry-run mode active. Validating multi-turn SkyRL environment rollouts and reward dynamics..."
        )

        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        try:
            conn = sqlite3.connect(tmp_db_path)
            conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);")
            conn.execute("INSERT INTO test_table VALUES (1, 'item_1'), (2, 'item_2');")
            conn.commit()
            conn.close()

            mock_instances = [
                {
                    "question": "What is the total number of records?",
                    "gold_sql": "SELECT COUNT(*) FROM test_table;",
                    "db_id": "test_db",
                    "db_path": tmp_db_path,
                }
            ]

            for step in range(min(max_steps if max_steps > 0 else 5, 5)):
                instance = mock_instances[step % len(mock_instances)]
                rollout_rewards: list[float] = []

                for _ in range(num_rollouts):
                    env.reset(instance)
                    # Turn 1: inspect schema
                    _, r1, done1, _ = env.step('<tool name="inspect_schema">["test_table"]</tool>')
                    assert not done1

                    # Turn 2: final answer
                    _, r2, done2, _ = env.step(
                        "<final_answer>SELECT COUNT(*) FROM test_table;</final_answer>"
                    )
                    assert done2

                    rollout_rewards.append(r2)

                # Compute group-normalized relative advantage
                advantages = compute_group_normalized_rewards(
                    rollout_rewards, group_size=num_rollouts
                )
                mean_reward = sum(rollout_rewards) / len(rollout_rewards)

                telemetry["episodes_completed"] += 1
                telemetry["mean_rewards"].append(mean_reward)
                telemetry["group_advantages"].append(advantages)

                logger.info(
                    "Step %d/%d: Mean Reward=%.4f, Advantages=%s",
                    step + 1,
                    min(max_steps if max_steps > 0 else 5, 5),
                    mean_reward,
                    [round(a, 3) for a in advantages[:4]],
                )

            # Evaluate adapter step on mock batch and persist checkpoint
            mock_batch = RolloutBatch(
                prompts=[instance["question"]] * num_rollouts,
                completions=["SELECT COUNT(*) FROM test_table;"] * num_rollouts,
                gold_sqls=[instance["gold_sql"]] * num_rollouts,
                db_paths=[tmp_db_path] * num_rollouts,
            )
            mock_action_mask = torch.ones((num_rollouts, 16), dtype=torch.float32)
            mock_teacher_logps = torch.zeros((num_rollouts, 16), dtype=torch.float32) + 0.05
            step_metrics = adapter.train_step(
                mock_batch,
                action_mask=mock_action_mask,
                teacher_logps=mock_teacher_logps,
            )
            telemetry["last_step_metrics"] = step_metrics

            # Persist dry-run state artifacts
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            adapter.save_checkpoint(str(out_path))
            with open(out_path / "training_state.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dry_run": True,
                        "episodes": telemetry["episodes_completed"],
                        "final_mean_reward": (
                            telemetry["mean_rewards"][-1] if telemetry["mean_rewards"] else 0.0
                        ),
                        "last_step_metrics": step_metrics,
                    },
                    f,
                    indent=2,
                )

        finally:
            Path(tmp_db_path).unlink(missing_ok=True)

        telemetry["status"] = "dry_run_completed"
        logger.info(
            "Dry-run validation completed successfully. Artifacts saved to %s",
            output_dir,
        )
        return telemetry

    # Full Multi-Turn RLVR agent training loop over preprocessed dataset
    logger.info(
        "Executing Full Multi-Turn RLVR Agent training loop for output_dir=%s",
        output_dir,
    )
    processed_dir = Path(getattr(getattr(cfg, "paths", {}), "processed_dir", "data/processed"))
    train_path = processed_dir / "train.jsonl"

    instances: list[dict[str, Any]] = []
    if train_path.is_file():
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))
        logger.info("Loaded %d training instances from %s", len(instances), train_path)
    else:
        logger.warning(
            "Training split not found at %s. Falling back to synthetic instances.",
            train_path,
        )

    if not instances:
        instances = [
            {
                "question": "What is the total number of records?",
                "gold_sql": "SELECT 1;",
                "db_id": "default_db",
                "db_path": ":memory:",
            }
        ]

    # Initialize agent policy pipeline
    model_path = getattr(training_cfg, "model_path", None)
    if not model_path or not Path(str(model_path)).exists():
        model_name = getattr(getattr(cfg, "model", {}), "name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        model_path = model_name

    logger.info("Initializing Agent Policy with model_path=%s", model_path)
    pipeline = NL2SQLPipeline(
        model_path=model_path,
        max_turns=int(training_cfg.get("max_turns", 5)),
    )

    if adapter.grpo_config.distillation_enabled:
        adapter.initialize_teacher(model=pipeline.model, tokenizer=pipeline.tokenizer)

    from ask_the_database.training.action_masking import ActionMaskGenerator

    action_mask_gen = (
        ActionMaskGenerator(tokenizer=pipeline.tokenizer)
        if pipeline.tokenizer is not None
        else None
    )

    report_to = str(training_cfg.get("report_to", "none")).lower()
    wandb_run = None
    if report_to in ("wandb", "wandb_and_weave", "all"):
        try:
            import wandb

            run_name = Path(output_dir).name
            wandb_run = wandb.init(
                project=str(getattr(cfg, "project_name", "ask-the-database")),
                name=f"rlvr_{run_name}",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info("Initialized Weights & Biases run: rlvr_%s", run_name)
        except Exception as e:
            logger.warning(
                "Failed to initialize Weights & Biases: %s. Continuing without wandb.",
                e,
            )

    if report_to in ("weave", "wandb_and_weave", "all") or bool(
        training_cfg.get("weave", {}).get("enabled", False)
    ):
        try:
            import weave

            project_name = str(
                training_cfg.get("weave", {}).get(
                    "project", getattr(cfg, "project_name", "ask-the-database")
                )
            )
            weave.init(project_name)
            logger.info("Initialized Weave tracking project: %s", project_name)
        except Exception as e:
            logger.warning(
                "Failed to initialize Weave: %s. Continuing without weave.",
                e,
            )

    logging_steps = int(getattr(training_cfg, "logging_steps", 5))
    save_steps = int(getattr(training_cfg, "save_steps", 50))
    total_steps = max_steps if max_steps > 0 else min(len(instances), 250)

    for step in range(total_steps):
        instance = instances[step % len(instances)]
        rollout_rewards: list[float] = []
        completions: list[str] = []
        rollout_histories: list[list[dict[str, Any]]] = []
        total_turns = 0
        total_tool_calls = 0
        failed_tool_calls = 0

        for _ in range(num_rollouts):
            env.reset(instance)
            done = False
            r = 0.0
            last_action = ""
            turn_count = 0
            while not done:
                turn_count += 1
                tables = env.registry.schema_manager.get_tables() if env.registry else None
                sampling_temp = float(getattr(training_cfg, "temperature", 0.7))
                action = pipeline.generate_step(
                    messages=env.history,
                    available_tables=tables,
                    default_query=instance.get("gold_sql"),
                    do_sample=True,
                    temperature=sampling_temp,
                )
                last_action = action
                obs, r, done, _ = env.step(action)
                if "<tool" in action or "<tool_call>" in action:
                    total_tool_calls += 1
                    obs_lower = obs.lower()
                    if "error" in obs_lower or "syntax" in obs_lower or "no such" in obs_lower:
                        failed_tool_calls += 1

            total_turns += turn_count
            rollout_rewards.append(r)
            completions.append(last_action)
            rollout_histories.append(list(env.history))

        avg_turns = total_turns / max(1, num_rollouts)
        err_rate = failed_tool_calls / max(1, total_tool_calls)
        tool_efficiency = float(max(0.0, 1.0 - err_rate) / (1.0 + 0.1 * avg_turns))

        batch = RolloutBatch(
            prompts=[instance.get("question", "")] * num_rollouts,
            completions=completions,
            gold_sqls=[instance.get("gold_sql", "")] * num_rollouts,
            db_paths=[str(instance.get("db_path", ":memory:"))] * num_rollouts,
        )

        # Extract token-level action masks and teacher logps if available
        batch_action_mask = None
        batch_teacher_logps = None

        if action_mask_gen is not None and pipeline.tokenizer is not None and rollout_histories:
            try:
                mask_list = []
                ids_list = []
                for hist in rollout_histories:
                    inp_ids, a_mask = action_mask_gen.generate_from_messages(hist)
                    ids_list.append(inp_ids.squeeze(0))
                    mask_list.append(a_mask.squeeze(0))

                max_len = max(t.size(0) for t in ids_list)
                padded_ids = torch.zeros((len(ids_list), max_len), dtype=torch.long)
                padded_mask = torch.zeros((len(mask_list), max_len), dtype=torch.float32)
                for i, (ids, m) in enumerate(zip(ids_list, mask_list, strict=True)):
                    padded_ids[i, : ids.size(0)] = ids
                    padded_mask[i, : m.size(0)] = m

                batch_action_mask = padded_mask

                if adapter.teacher is not None and pipeline.model is not None:
                    batch_teacher_logps = adapter.teacher.compute_action_logprobs(
                        input_ids=padded_ids,
                    )
            except Exception as exc:
                logger.debug("Failed to compute dynamic action masks / teacher logps: %s", exc)

        step_metrics = adapter.train_step(
            batch,
            action_mask=batch_action_mask,
            teacher_logps=batch_teacher_logps,
            tool_interaction_efficiency=tool_efficiency,
        )

        advantages = compute_group_normalized_rewards(rollout_rewards, group_size=num_rollouts)
        mean_reward = sum(rollout_rewards) / len(rollout_rewards) if rollout_rewards else 0.0

        telemetry["episodes_completed"] += 1
        telemetry["mean_rewards"].append(mean_reward)
        telemetry["group_advantages"].append(advantages)
        telemetry["last_step_metrics"] = step_metrics

        if wandb_run is not None:
            wandb_payload = {
                "step": step + 1,
                "mean_reward": mean_reward,
                "min_reward": min(rollout_rewards) if rollout_rewards else 0.0,
                "max_reward": max(rollout_rewards) if rollout_rewards else 0.0,
                "advantage_mean": (sum(advantages) / len(advantages) if advantages else 0.0),
            }
            wandb_payload.update(step_metrics)
            wandb.log(wandb_payload, step=step + 1)

        if (step + 1) % logging_steps == 0 or step == 0 or (step + 1) == total_steps:
            if adapter.grpo_config.distillation_enabled:
                logger.info(
                    "Step %d/%d (%.1f%%) [%s] — Mean Reward: %.4f | Policy Loss: %.4f | Filtered: %.1f%% | Effective Tokens: %d | R0: %.2f | T1 Conf: %.2f | T2 Conf: %.2f | Grad Norm: %.3f | Tool Eff: %.2f | Advantages: %s",
                    step + 1,
                    total_steps,
                    ((step + 1) / total_steps) * 100,
                    adapter.grpo_config.algorithm.upper(),
                    mean_reward,
                    step_metrics.get("policy_loss", 0.0),
                    step_metrics.get("filtered_groups_ratio", 0.0) * 100.0,
                    int(step_metrics.get("effective_tokens", 0)),
                    step_metrics.get("r_zero_ratio", 0.0),
                    step_metrics.get("type1_conflict_ratio", 0.0),
                    step_metrics.get("type2_conflict_ratio", 0.0),
                    step_metrics.get("telemetry/grad_norm", 0.0),
                    step_metrics.get("telemetry/tool_interaction_efficiency", 1.0),
                    [round(a, 3) for a in advantages[:4]],
                )
            else:
                logger.info(
                    "Step %d/%d (%.1f%%) [%s] — Mean Reward: %.4f | Policy Loss: %.4f | Filtered: %.1f%% | Effective Tokens: %d | Advantages: %s",
                    step + 1,
                    total_steps,
                    ((step + 1) / total_steps) * 100,
                    adapter.grpo_config.algorithm.upper(),
                    mean_reward,
                    step_metrics.get("policy_loss", 0.0),
                    step_metrics.get("filtered_groups_ratio", 0.0) * 100.0,
                    int(step_metrics.get("effective_tokens", 0)),
                    [round(a, 3) for a in advantages[:4]],
                )

        if (step + 1) % save_steps == 0 or (step + 1) == total_steps:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            adapter.save_checkpoint(str(out_path))
            with open(out_path / "training_state.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dry_run": False,
                        "episodes": telemetry["episodes_completed"],
                        "current_step": step + 1,
                        "total_steps": total_steps,
                        "final_mean_reward": mean_reward,
                        "reward_history": telemetry["mean_rewards"],
                        "last_step_metrics": step_metrics,
                    },
                    f,
                    indent=2,
                )
            (out_path / "rlvr_policy_config.json").write_text(
                json.dumps(
                    {
                        "adapter": "SkyRLSQLEnvironment",
                        "model_path": str(model_path),
                        "max_turns": int(training_cfg.get("max_turns", 5)),
                        "steps": step + 1,
                    },
                    indent=2,
                )
            )

            # Persist model adapter weights and tokenizer if pipeline has them loaded
            if pipeline.model is not None:
                try:
                    pipeline.model.save_pretrained(out_path)
                except Exception as e:
                    logger.warning("Could not save model to %s: %s", out_path, e)
            elif Path(str(model_path)).is_dir():
                import shutil

                for f_name in [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                ]:
                    src_f = Path(str(model_path)) / f_name
                    dst_f = out_path / f_name
                    if src_f.exists() and not dst_f.exists():
                        shutil.copy2(src_f, dst_f)
            if pipeline.tokenizer is not None:
                try:
                    pipeline.tokenizer.save_pretrained(out_path)
                except Exception as e:
                    logger.warning("Could not save tokenizer to %s: %s", out_path, e)

    if wandb_run is not None:
        wandb_run.finish()

    telemetry["status"] = "training_completed"
    logger.info(
        "Multi-Turn RLVR Agent Training Completed successfully. Artifacts saved to %s",
        output_dir,
    )
    return telemetry


def _normalize_cli_args() -> None:
    """Normalize CLI arguments so common training parameters work with or without 'training.' prefix."""
    import sys

    top_level_groups = {
        "environment",
        "data",
        "model",
        "training",
        "reward",
        "evaluation",
        "project_name",
        "seed",
        "dry_run",
        "paths",
        "sweeps",
        "verification",
        "hydra",
    }

    normalized_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith(("-", "@")):
            prefix = "+" if arg.startswith("+") else ""
            clean_arg = arg[1:] if arg.startswith("+") else arg
            key = clean_arg.split("=")[0].strip()
            root_key = key.split(".")[0].strip()
            if root_key not in top_level_groups:
                normalized_argv.append(f"{prefix}training.{clean_arg}")
                continue
        normalized_argv.append(arg)
    sys.argv = normalized_argv


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI main entrypoint."""
    logger.info("Starting RLVR Agent training with config:\n%s", OmegaConf.to_yaml(cfg))
    run_rlvr_training(cfg)


if __name__ == "__main__":
    _normalize_cli_args()
    main()
