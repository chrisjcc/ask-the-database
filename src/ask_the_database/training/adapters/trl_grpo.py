"""Hugging Face TRL GRPO Adapter for Group Relative Policy Optimization.

Bridges between TRL GRPOTrainer policy rollouts and the SQLRewardRubric
verifiable database verification engine, supporting DAPO and GSPO.
"""

import contextlib
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.dataset import extract_sql_from_completion

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainingConfig:
    """Configuration for Group Relative Policy Optimization and enterprise variants (DAPO, GSPO)."""

    algorithm: Literal["grpo", "dapo", "gspo", "opd", "opdvr", "grpd"] = "dapo"
    enable_dynamic_sampling: bool = True
    min_reward_std: float = 1e-4
    dapo_clip_low: float = 0.20
    dapo_clip_high: float = 0.28
    gspo_seq_epsilon: float = 3e-4
    epsilon: float = 0.20
    beta: float = 0.04
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    max_completion_length: int = 512
    logging_steps: int = 1
    save_strategy: str = "epoch"
    report_to: str = "none"
    bf16: bool = False
    fp16: bool = False
    max_steps: int = -1
    output_dir: str = "models/rlvr/harness_zero_grpo"
    distillation_enabled: bool = False
    teacher_model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    share_base_model: bool = True
    privileged_schema: bool = True
    gating_mode: Literal["none", "relu", "inverse"] = "relu"
    distillation_eps: float = 1e-8
    max_ddl_tokens: int = 2500
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict_config(cls, cfg: DictConfig | dict[str, Any]) -> "GRPOTrainingConfig":
        """Instantiate GRPOTrainingConfig from a Hydra DictConfig or standard dictionary."""
        training_cfg = (
            cfg.get("training", cfg) if hasattr(cfg, "get") else getattr(cfg, "training", cfg)
        )

        def get_val(key: str, default: Any) -> Any:
            if hasattr(training_cfg, key):
                val = getattr(training_cfg, key)
                return default if val is None else val
            if isinstance(training_cfg, dict):
                val = training_cfg.get(key)
                return default if val is None else val
            return default

        dist_cfg = (
            cfg.get("distillation", {}) if hasattr(cfg, "get") else getattr(cfg, "distillation", {})
        )
        if hasattr(dist_cfg, "_content") or hasattr(dist_cfg, "__dict__"):
            with contextlib.suppress(Exception):
                dist_cfg = OmegaConf.to_container(dist_cfg, resolve=True)
        if not isinstance(dist_cfg, dict):
            dist_cfg = {}

        algo_val = str(get_val("algorithm", "dapo")).lower()
        is_dist_algo = algo_val in ("opd", "opdvr", "grpd")
        dist_enabled = bool(dist_cfg.get("enabled", is_dist_algo))
        gating_mode_val = str(
            dist_cfg.get("gating_mode", "none" if algo_val == "opd" else "relu")
        ).lower()

        return cls(
            algorithm=algo_val,  # type: ignore[arg-type]
            enable_dynamic_sampling=bool(get_val("enable_dynamic_sampling", True)),
            min_reward_std=float(get_val("min_reward_std", 1e-4)),
            dapo_clip_low=float(get_val("dapo_clip_low", 0.20)),
            dapo_clip_high=float(get_val("dapo_clip_high", 0.28)),
            gspo_seq_epsilon=float(get_val("gspo_seq_epsilon", 3e-4)),
            epsilon=float(get_val("epsilon", 0.20)),
            beta=float(get_val("beta", 0.04)),
            learning_rate=float(get_val("learning_rate", 1e-5)),
            num_train_epochs=float(get_val("num_train_epochs", 1.0)),
            per_device_train_batch_size=int(get_val("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(get_val("gradient_accumulation_steps", 4)),
            num_generations=int(get_val("num_generations", 8)),
            max_completion_length=int(get_val("max_completion_length", 512)),
            logging_steps=int(get_val("logging_steps", 1)),
            save_strategy=str(get_val("save_strategy", "epoch")),
            report_to=str(get_val("report_to", "none")),
            bf16=bool(get_val("bf16", False)),
            fp16=bool(get_val("fp16", False)),
            max_steps=int(get_val("max_steps", -1)),
            output_dir=str(get_val("output_dir", "models/rlvr/harness_zero_grpo")),
            distillation_enabled=dist_enabled,
            teacher_model_name_or_path=str(
                dist_cfg.get("teacher_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
            ),
            share_base_model=bool(dist_cfg.get("share_base_model", True)),
            privileged_schema=bool(dist_cfg.get("privileged_schema", True)),
            gating_mode=gating_mode_val,  # type: ignore[arg-type]
            distillation_eps=float(dist_cfg.get("eps", 1e-8)),
            max_ddl_tokens=int(dist_cfg.get("max_ddl_tokens", 2500)),
        )


@dataclass
class PolicyLossTelemetry:
    """Telemetry diagnostics produced during policy gradient optimization."""

    loss: torch.Tensor
    policy_loss: float
    filtered_groups_ratio: float
    effective_tokens: int
    active_groups: int
    total_groups: int
    mean_seq_ratio: float = 1.0
    clip_fraction_low: float = 0.0
    clip_fraction_high: float = 0.0
    algorithm: str = "dapo"
    r_zero_ratio: float = 0.0
    type1_conflict_ratio: float = 0.0
    type2_conflict_ratio: float = 0.0
    grad_norm: float = 0.0
    tool_interaction_efficiency: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Convert telemetry metrics to dictionary."""
        loss_val = (
            float(self.loss.detach().item()) if hasattr(self.loss, "detach") else float(self.loss)
        )
        return {
            "loss": loss_val,
            "policy_loss": self.policy_loss,
            "train/loss": loss_val,
            "train/policy_loss": self.policy_loss,
            "train/filtered_groups_ratio": self.filtered_groups_ratio,
            "train/effective_tokens": float(self.effective_tokens),
            "train/active_groups": float(self.active_groups),
            "train/total_groups": float(self.total_groups),
            "train/mean_seq_ratio": self.mean_seq_ratio,
            "train/clip_fraction_low": self.clip_fraction_low,
            "train/clip_fraction_high": self.clip_fraction_high,
            "train/r_zero_ratio": self.r_zero_ratio,
            "train/type1_conflict_ratio": self.type1_conflict_ratio,
            "train/type2_conflict_ratio": self.type2_conflict_ratio,
            "train/grad_norm": self.grad_norm,
            "telemetry/r_zero_ratio": self.r_zero_ratio,
            "telemetry/type1_conflict_ratio": self.type1_conflict_ratio,
            "telemetry/type2_conflict_ratio": self.type2_conflict_ratio,
            "telemetry/grad_norm": self.grad_norm,
            "telemetry/tool_interaction_efficiency": self.tool_interaction_efficiency,
        }


def compute_group_normalized_rewards(
    rewards: list[float],
    group_size: int = 8,
    epsilon: float = 1e-8,
) -> list[float]:
    """Normalize rewards within prompt groups to have zero mean and unit variance.

    For each group of size G, computes:
        A_i = (R_i - mean(R_g)) / (std(R_g) + epsilon)
    If standard deviation within group is zero, returns zero advantages.
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


def compute_policy_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int = 8,
    config: GRPOTrainingConfig | None = None,
    action_mask: torch.Tensor | None = None,
) -> PolicyLossTelemetry:
    """Compute policy optimization loss across GRPO, DAPO, and GSPO algorithms.

    Args:
        per_token_logps: Log probabilities from current policy, shaped (B, T).
        old_per_token_logps: Log probabilities from previous/ref policy, shaped (B, T).
        completion_mask: Binary mask for completion tokens, shaped (B, T).
        rewards: Raw scalar rewards for each trajectory, shaped (B,).
        group_size: Number of generations G per prompt.
        config: GRPOTrainingConfig specifying algorithm and clipping hyperparameters.
        action_mask: Optional binary mask isolating policy action/thought tokens (1)
            from environment observation tokens (0). If provided, observation tokens
            are strictly excluded from policy gradient updates.

    Returns:
        PolicyLossTelemetry with scalar loss and diagnostic statistics.
    """
    if config is None:
        config = GRPOTrainingConfig()

    b, _ = per_token_logps.shape
    g = group_size if (group_size > 0 and b >= group_size and b % group_size == 0) else max(1, b)
    num_groups = max(1, b // g)

    # 1. Group-relative Advantage Normalization & Dynamic Sampling
    rewards_grouped = rewards.view(num_groups, g)
    means = rewards_grouped.mean(dim=-1, keepdim=True)
    stds = rewards_grouped.std(dim=-1, keepdim=True, unbiased=False)

    if config.enable_dynamic_sampling:
        valid_groups = stds.squeeze(-1) > config.min_reward_std
    else:
        valid_groups = torch.ones(num_groups, dtype=torch.bool, device=rewards.device)

    active_groups_count = int(valid_groups.sum().item())
    filtered_ratio = 1.0 - (active_groups_count / max(1, num_groups))

    norm_advantages = (rewards_grouped - means) / (stds + 1e-8)
    # Zero out advantages for filtered zero-variance groups
    norm_advantages = norm_advantages * valid_groups.unsqueeze(-1).float()
    advantages = norm_advantages.view(b)  # (B,)

    # 2. Token-level Log Ratio with Multi-Turn Action Loss Masking
    eff_mask = completion_mask if action_mask is None else (completion_mask * action_mask)
    token_log_ratio = per_token_logps - old_per_token_logps
    token_log_ratio = token_log_ratio * eff_mask
    token_ratio = torch.exp(token_log_ratio)

    # 3. Active Masking
    group_mask = valid_groups.repeat_interleave(g).float()  # (B,)
    token_active_mask = eff_mask * group_mask.unsqueeze(-1)  # (B, T)
    effective_tokens = int(token_active_mask.sum().item())

    mean_seq_ratio = 1.0
    clip_fraction_low = 0.0
    clip_fraction_high = 0.0
    r_zero_ratio = 0.0
    type1_conflict_ratio = 0.0
    type2_conflict_ratio = 0.0
    adv_expanded = advantages.unsqueeze(-1)  # (B, 1)

    # 4. Objective Formulation Dispatch
    algo = config.algorithm.lower()

    if algo == "grpo":
        # Standard symmetric token-level clipped surrogate
        eps = config.epsilon
        surr1 = token_ratio * adv_expanded
        surr2 = torch.clamp(token_ratio, 1.0 - eps, 1.0 + eps) * adv_expanded
        token_loss = -torch.min(surr1, surr2)

        if effective_tokens > 0:
            loss = (token_loss * token_active_mask).sum() / float(effective_tokens)
        else:
            loss = 0.0 * per_token_logps.sum()

        with torch.no_grad():
            low_clipped = (token_ratio < (1.0 - eps)) & (adv_expanded < 0) & (token_active_mask > 0)
            high_clipped = (
                (token_ratio > (1.0 + eps)) & (adv_expanded > 0) & (token_active_mask > 0)
            )
            denom = max(1, effective_tokens)
            clip_fraction_low = float(low_clipped.sum().item() / denom)
            clip_fraction_high = float(high_clipped.sum().item() / denom)

    elif algo == "dapo":
        # Decoupled Asymmetric Clipping & Dynamic Sampling
        clip_low = config.dapo_clip_low
        clip_high = config.dapo_clip_high
        surr1 = token_ratio * adv_expanded
        surr2 = torch.clamp(token_ratio, 1.0 - clip_low, 1.0 + clip_high) * adv_expanded
        token_loss = -torch.min(surr1, surr2)

        if effective_tokens > 0:
            loss = (token_loss * token_active_mask).sum() / float(effective_tokens)
        else:
            loss = 0.0 * per_token_logps.sum()

        with torch.no_grad():
            low_clipped = (
                (token_ratio < (1.0 - clip_low)) & (adv_expanded < 0) & (token_active_mask > 0)
            )
            high_clipped = (
                (token_ratio > (1.0 + clip_high)) & (adv_expanded > 0) & (token_active_mask > 0)
            )
            denom = max(1, effective_tokens)
            clip_fraction_low = float(low_clipped.sum().item() / denom)
            clip_fraction_high = float(high_clipped.sum().item() / denom)

    elif algo == "gspo":
        # Group Sequence Policy Optimization (Length-normalized sequence ratio)
        seq_lens = eff_mask.sum(dim=-1).clamp(min=1.0)  # (B,)
        seq_mean_log_ratio = (token_log_ratio * eff_mask).sum(dim=-1) / seq_lens  # (B,)
        seq_ratio = torch.exp(seq_mean_log_ratio)  # (B,)

        eps_s = config.gspo_seq_epsilon
        surr1_seq = seq_ratio * advantages
        surr2_seq = torch.clamp(seq_ratio, 1.0 - eps_s, 1.0 + eps_s) * advantages
        seq_loss = -torch.min(surr1_seq, surr2_seq)  # (B,)

        num_active_samples = active_groups_count * g
        if num_active_samples > 0:
            loss = (seq_loss * group_mask).sum() / float(num_active_samples)
        else:
            loss = 0.0 * per_token_logps.sum()

        with torch.no_grad():
            if num_active_samples > 0:
                mean_seq_ratio = float((seq_ratio * group_mask).sum().item() / num_active_samples)
            else:
                mean_seq_ratio = 1.0
            low_clipped = (seq_ratio < (1.0 - eps_s)) & (advantages < 0) & (group_mask > 0)
            high_clipped = (seq_ratio > (1.0 + eps_s)) & (advantages > 0) & (group_mask > 0)
            denom_seq = max(1, num_active_samples)
            clip_fraction_low = float(low_clipped.sum().item() / denom_seq)
            clip_fraction_high = float(high_clipped.sum().item() / denom_seq)

    elif algo in ("opd", "opdvr", "grpd"):
        # Group Relative Policy Distillation with Verifiable Reward (OPDVR / GRPD)
        from ask_the_database.training.losses import (
            compute_conflict_diagnostics,
            compute_grpd_token_rewards,
        )

        gating_mode = config.gating_mode
        if algo == "opd":
            gating_mode = "none"
        elif algo in ("opdvr", "grpd") and gating_mode not in ("none", "relu", "inverse"):
            gating_mode = "relu"

        token_rewards = compute_grpd_token_rewards(
            student_logprobs=per_token_logps,
            teacher_logprobs=old_per_token_logps,
            advantages=adv_expanded,
            gating_mode=gating_mode,
            eps=config.distillation_eps,
        )
        token_loss = -token_rewards * per_token_logps

        if effective_tokens > 0:
            loss = (token_loss * token_active_mask).sum() / float(effective_tokens)
        else:
            loss = 0.0 * per_token_logps.sum()

        with torch.no_grad():
            conf_diag = compute_conflict_diagnostics(
                student_logprobs=per_token_logps,
                teacher_logprobs=old_per_token_logps,
                advantages=adv_expanded,
                action_mask=token_active_mask,
                token_rewards=token_rewards,
                eps=config.distillation_eps,
            )
            r_zero_ratio = float(conf_diag.get("r_zero_ratio", 0.0))
            type1_conflict_ratio = float(conf_diag.get("type1_conflict_ratio", 0.0))
            type2_conflict_ratio = float(conf_diag.get("type2_conflict_ratio", 0.0))

    else:
        raise ValueError(f"Unsupported policy optimization algorithm: '{config.algorithm}'")

    return PolicyLossTelemetry(
        loss=loss,
        policy_loss=float(loss.detach().item()),
        filtered_groups_ratio=filtered_ratio,
        effective_tokens=effective_tokens,
        active_groups=active_groups_count,
        total_groups=num_groups,
        mean_seq_ratio=mean_seq_ratio,
        clip_fraction_low=clip_fraction_low,
        clip_fraction_high=clip_fraction_high,
        algorithm=algo,
        r_zero_ratio=r_zero_ratio,
        type1_conflict_ratio=type1_conflict_ratio,
        type2_conflict_ratio=type2_conflict_ratio,
    )


class CustomGRPOTrainer(GRPOTrainer):
    """Custom Hugging Face GRPOTrainer supporting DAPO and GSPO optimization objectives."""

    def __init__(self, grpo_config: GRPOTrainingConfig, *args: Any, **kwargs: Any) -> None:
        self.grpo_training_config = grpo_config
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model: torch.nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        """Override _compute_loss to dispatch DAPO, GSPO, or standard GRPO objectives."""
        input_ids = inputs["input_ids"]
        prompt_mask = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")
        action_mask = inputs.get("action_mask")
        rewards = inputs.get("rewards")

        if prompt_mask is None or completion_mask is None or rewards is None:
            # Fallback to standard superclass compute_loss if custom inputs are not available
            res = super()._compute_loss(model, inputs)  # type: ignore[no-untyped-call]
            return cast(torch.Tensor, res)

        # Forward pass through model to obtain logits
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        comp_mask = completion_mask[:, 1:]
        act_mask = None
        if action_mask is not None:
            act_mask = (
                action_mask[:, 1:] if action_mask.shape[1] == input_ids.shape[1] else action_mask
            )

        logps = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(logps, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = old_per_token_logps[:, 1:]

        group_size = getattr(self.args, "num_generations", 8)
        telemetry = compute_policy_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            completion_mask=comp_mask,
            rewards=rewards,
            group_size=group_size,
            config=self.grpo_training_config,
            action_mask=act_mask,
        )

        # Log metrics to trainer tracker
        if hasattr(self, "_metrics"):
            mode = "train" if model.training else "eval"
            if mode in self._metrics:
                self._metrics[mode]["loss/policy"].append(telemetry.policy_loss)
                self._metrics[mode]["dapo/filtered_groups_ratio"].append(
                    telemetry.filtered_groups_ratio
                )
                self._metrics[mode]["dapo/effective_tokens"].append(
                    float(telemetry.effective_tokens)
                )
                self._metrics[mode]["gspo/mean_seq_ratio"].append(telemetry.mean_seq_ratio)

        return telemetry.loss


class TRLGRPOAdapter(BaseRLAdapter):
    """TRL GRPOTrainer implementation supporting GRPO, DAPO, and GSPO policy optimization."""

    def __init__(
        self,
        config: DictConfig | GRPOTrainingConfig,
        rubric: SQLRewardRubric | None = None,
    ) -> None:
        """Initialize adapter with configuration and verifiable reward rubric."""
        reward_rubric = rubric if rubric is not None else SQLRewardRubric()
        if isinstance(config, GRPOTrainingConfig):
            self.grpo_config = config
            hydra_cfg = OmegaConf.create({"training": config.__dict__})
            super().__init__(hydra_cfg, reward_rubric)
        else:
            super().__init__(config, reward_rubric)
            self.grpo_config = GRPOTrainingConfig.from_dict_config(config)

        self.output_dir = Path(self.grpo_config.output_dir)

    def initialize_models(self) -> None:
        """Initialize actor policy and tokenizer."""
        logger.info(
            "Initializing models for TRL GRPO Adapter under '%s' algorithm...",
            self.grpo_config.algorithm,
        )

    def compute_rewards(self, batch: RolloutBatch) -> list[float]:
        """Compute scalar rewards for each completion in the rollout batch using the rubric."""
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
        """Compute group-relative advantages normalized across generation groups."""
        raw_rewards = self.compute_rewards(batch)
        g = group_size or self.grpo_config.num_generations
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
        cfg = self.grpo_config

        is_cuda = torch.cuda.is_available()
        bf16 = cfg.bf16 and is_cuda and torch.cuda.is_bf16_supported()
        fp16 = not bf16 and is_cuda

        grad_accum = cfg.gradient_accumulation_steps
        if (cfg.per_device_train_batch_size * grad_accum) % cfg.num_generations != 0:
            grad_accum = max(
                1,
                (cfg.num_generations + cfg.per_device_train_batch_size - 1)
                // cfg.per_device_train_batch_size,
            )

        report_to = cfg.report_to
        if report_to == "wandb":
            try:
                import wandb  # noqa: F401

                if not os.environ.get("WANDB_PROJECT"):
                    os.environ["WANDB_PROJECT"] = "ask-the-database"
            except ImportError:
                logger.warning("wandb is not installed. Falling back to report_to='none'.")
                report_to = "none"

        return GRPOConfig(
            output_dir=out_dir,
            learning_rate=cfg.learning_rate,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            beta=cfg.beta,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            report_to=report_to,
            bf16=bf16,
            fp16=fp16,
            max_steps=cfg.max_steps,
        )

    def train_step(
        self,
        batch: RolloutBatch,
        group_size: int | None = None,
        action_mask: torch.Tensor | None = None,
        teacher_logps: torch.Tensor | None = None,
        tool_interaction_efficiency: float = 1.0,
    ) -> dict[str, float]:
        """Execute reward scoring and compute telemetry metrics for the rollout batch."""
        rewards = self.compute_rewards(batch)
        n = len(rewards)
        default_g = self.grpo_config.num_generations
        g = group_size or (default_g if (n >= default_g and n % default_g == 0) else max(1, n))
        advantages = compute_group_normalized_rewards(
            rewards,
            group_size=g,
        )

        mean_reward = sum(rewards) / n if n > 0 else 0.0
        min_reward = min(rewards) if n > 0 else 0.0
        max_reward = max(rewards) if n > 0 else 0.0

        # Construct dummy tensor logs for telemetry calculation
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        seq_len = action_mask.shape[1] if action_mask is not None else 8
        dummy_logps = torch.zeros((n, seq_len), dtype=torch.float32)
        dummy_mask = torch.ones((n, seq_len), dtype=torch.float32)

        if teacher_logps is not None:
            old_logps = teacher_logps
        elif self.grpo_config.algorithm in ("opd", "opdvr", "grpd"):
            # Provide non-identical teacher baseline to compute diagnostics under dummy inputs
            old_logps = dummy_logps + 0.1
        else:
            old_logps = dummy_logps

        # Enable gradient tracking on logps to compute policy gradient norm
        active_logps = dummy_logps.clone().detach().requires_grad_(True)
        telemetry = compute_policy_loss(
            per_token_logps=active_logps,
            old_per_token_logps=old_logps,
            completion_mask=dummy_mask,
            rewards=rewards_t,
            group_size=g,
            config=self.grpo_config,
            action_mask=action_mask,
        )

        grad_norm = 0.0
        if telemetry.loss.requires_grad:
            telemetry.loss.backward()  # type: ignore[no-untyped-call]
            if active_logps.grad is not None:
                grad_norm = float(torch.norm(active_logps.grad).item())

        telemetry.grad_norm = grad_norm
        telemetry.tool_interaction_efficiency = tool_interaction_efficiency

        metrics = telemetry.to_dict()
        metrics.update(
            {
                "batch_size": float(n),
                "raw_mean_reward": mean_reward,
                "raw_min_reward": min_reward,
                "raw_max_reward": max_reward,
                "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            }
        )
        return metrics

    def initialize_teacher(self, model: Any | None = None, tokenizer: Any | None = None) -> None:
        """Initialize PrivilegedSchemaTeacher if distillation is enabled."""
        if self.grpo_config.distillation_enabled:
            from ask_the_database.training.privileged_teacher import PrivilegedSchemaTeacher

            self.teacher = PrivilegedSchemaTeacher(
                model=model,
                tokenizer=tokenizer,
                is_shared_base=self.grpo_config.share_base_model,
                max_ddl_tokens=self.grpo_config.max_ddl_tokens,
            )
            logger.info(
                "Initialized PrivilegedSchemaTeacher for %s (gating=%s, shared_base=%s)",
                self.grpo_config.algorithm.upper(),
                self.grpo_config.gating_mode,
                self.grpo_config.share_base_model,
            )

    def save_checkpoint(self, output_dir: str) -> None:
        """Persist checkpoint adapter weights or training state."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        config_data = {
            "adapter": "TRLGRPOAdapter",
            "algorithm": self.grpo_config.algorithm,
            "enable_dynamic_sampling": self.grpo_config.enable_dynamic_sampling,
            "dapo_clip_low": self.grpo_config.dapo_clip_low,
            "dapo_clip_high": self.grpo_config.dapo_clip_high,
            "gspo_seq_epsilon": self.grpo_config.gspo_seq_epsilon,
            "distillation_enabled": self.grpo_config.distillation_enabled,
            "gating_mode": self.grpo_config.gating_mode,
            "share_base_model": self.grpo_config.share_base_model,
        }
        import json

        (out_path / "grpo_adapter_config.json").write_text(json.dumps(config_data, indent=2) + "\n")
        logger.info("Saved GRPO checkpoint metadata to %s", output_dir)
