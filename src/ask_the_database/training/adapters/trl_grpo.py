"""Hugging Face TRL GRPO Adapter for Group Relative Policy Optimization.

Bridges between TRL GRPOTrainer policy rollouts and the SQLRewardRubric
verifiable database verification engine.
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from trl import GRPOConfig  # type: ignore[attr-defined]

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.dataset import extract_sql_from_completion

logger = logging.getLogger(__name__)


def compute_group_normalized_rewards(
    rewards: list[float],
    group_size: int = 8,
    epsilon: float = 1e-8,
) -> list[float]:
    """Normalize rewards within prompt groups to have zero mean and unit variance.

    For each group of size G, computes:
        A_i = (R_i - mean(R_g)) / (std(R_g) + epsilon)
    If standard deviation within group is zero, returns zero advantages.

    Args:
        rewards: Flattened list of raw rewards for all rollouts.
        group_size: Number of generations G per prompt.
        epsilon: Small numerical stabilizer to avoid division by zero.

    Returns:
        Flattened list of group-normalized advantages.
    """
    if not rewards:
        return []

    normalized: list[float] = []
    num_samples = len(rewards)

    for start_idx in range(0, num_samples, group_size):
        chunk = rewards[start_idx : start_idx + group_size]
        n = len(chunk)
        if n <= 1:
            normalized.extend([0.0] * n)
            continue

        mean = sum(chunk) / n
        variance = sum((r - mean) ** 2 for r in chunk) / n
        std = math.sqrt(variance)

        if std < epsilon:
            normalized.extend([0.0] * n)
        else:
            normalized.extend([(r - mean) / (std + epsilon) for r in chunk])

    return normalized


class TRLGRPOAdapter(BaseRLAdapter):
    """TRL GRPOTrainer implementation of the RL framework adapter."""

    def __init__(self, config: DictConfig, rubric: SQLRewardRubric) -> None:
        """Initialize adapter with Hydra config and verifiable reward rubric."""
        super().__init__(config, rubric)
        self.output_dir = Path(getattr(config.training, "output_dir", "models/grpo_baseline/final"))

    def initialize_models(self) -> None:
        """Initialize actor policy and tokenizer."""
        logger.info("Initializing models for TRL GRPO Adapter...")

    def compute_rewards(self, batch: RolloutBatch) -> list[float]:
        """Compute scalar rewards for each completion in the rollout batch using the rubric.

        Args:
            batch: Standardized RolloutBatch containing completions, gold SQLs, and db paths.

        Returns:
            List of scalar rewards for each completion.
        """
        rewards: list[float] = []
        for completion, gold_sql, db_path in zip(
            batch.completions, batch.gold_sqls, batch.db_paths, strict=True
        ):
            candidate_sql = extract_sql_from_completion(completion)
            out = self.rubric.evaluate(
                candidate_sql=candidate_sql,
                gold_sql=gold_sql,
                db_path=db_path,
            )
            rewards.append(out.total_reward)
        return rewards

    def compute_advantages(self, batch: RolloutBatch, group_size: int | None = None) -> list[float]:
        """Compute group-relative advantages normalized across generation groups.

        Args:
            batch: RolloutBatch.
            group_size: Optional override for group size G (defaults to config).

        Returns:
            List of normalized advantages.
        """
        raw_rewards = self.compute_rewards(batch)
        g = group_size or int(getattr(self.cfg.training, "num_generations", 8))
        return compute_group_normalized_rewards(raw_rewards, group_size=g)

    def make_reward_function(self) -> Callable[..., list[float]]:
        """Construct a reward function callable compatible with TRL GRPOTrainer."""

        def reward_func(
            completions: list[Any],
            gold_sqls: list[str] | None = None,
            gold_sql: list[str] | None = None,
            db_paths: list[str] | None = None,
            db_path: list[str] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            golds = gold_sqls if gold_sqls is not None else (gold_sql or [])
            dbs = db_paths if db_paths is not None else (db_path or [])
            n = len(completions)

            # If dataset provided single item per prompt group, broadcast to completions
            if len(golds) == 1 and n > 1:
                golds = golds * n
            if len(dbs) == 1 and n > 1:
                dbs = dbs * n

            rewards: list[float] = []
            for i, comp in enumerate(completions):
                cand = extract_sql_from_completion(comp)
                g = golds[i] if i < len(golds) else ""
                d = dbs[i] if i < len(dbs) else ""
                if not g or not d:
                    rewards.append(0.0)
                    continue
                out = self.rubric.evaluate(candidate_sql=cand, gold_sql=g, db_path=d)
                rewards.append(out.total_reward)
            return rewards

        return reward_func

    def build_training_args(self, output_dir: str | Path | None = None) -> GRPOConfig:
        """Build GRPOConfig for Hugging Face GRPOTrainer."""
        out_dir = str(output_dir or self.output_dir)
        training_cfg = self.cfg.training

        is_cuda = torch.cuda.is_available()
        bf16_requested = bool(getattr(training_cfg, "bf16", False))
        bf16 = bf16_requested and is_cuda and torch.cuda.is_bf16_supported()
        fp16 = not bf16 and is_cuda

        lr = float(getattr(training_cfg, "learning_rate", 1e-5))
        epochs = float(getattr(training_cfg, "num_train_epochs", 1))
        batch_size = int(getattr(training_cfg, "per_device_train_batch_size", 1))
        num_generations = int(getattr(training_cfg, "num_generations", 8))
        grad_accum = int(getattr(training_cfg, "gradient_accumulation_steps", 4))
        if (batch_size * grad_accum) % num_generations != 0:
            grad_accum = max(1, (num_generations + batch_size - 1) // batch_size)
        max_completion_length = int(getattr(training_cfg, "max_completion_length", 512))
        beta = float(getattr(training_cfg, "beta", 0.04))
        logging_steps = int(getattr(training_cfg, "logging_steps", 1))
        save_strategy = str(getattr(training_cfg, "save_strategy", "epoch"))
        report_to = str(getattr(training_cfg, "report_to", "none"))
        max_steps = int(getattr(training_cfg, "max_steps", -1))

        return GRPOConfig(
            output_dir=out_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            beta=beta,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to=report_to,
            bf16=bf16,
            fp16=fp16,
            max_steps=max_steps,
        )

    def train_step(self, batch: RolloutBatch) -> dict[str, float]:
        """Execute reward scoring and compute telemetry metrics for the rollout batch."""
        rewards = self.compute_rewards(batch)
        advantages = compute_group_normalized_rewards(
            rewards,
            group_size=int(getattr(self.cfg.training, "num_generations", 8)),
        )

        n = len(rewards)
        mean_reward = sum(rewards) / n if n > 0 else 0.0
        min_reward = min(rewards) if n > 0 else 0.0
        max_reward = max(rewards) if n > 0 else 0.0

        return {
            "batch_size": float(n),
            "raw_mean_reward": mean_reward,
            "raw_min_reward": min_reward,
            "raw_max_reward": max_reward,
            "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
        }

    def save_checkpoint(self, output_dir: str) -> None:
        """Persist checkpoint adapter weights or training state."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "grpo_adapter_config.json").write_text('{"adapter": "TRLGRPOAdapter"}\n')
        logger.info("Saved GRPO checkpoint metadata to %s", output_dir)
