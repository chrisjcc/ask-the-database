"""Framework adapters for reinforcement learning."""

from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment

try:
    from ask_the_database.training.adapters.trl_grpo import (
        CustomGRPOTrainer,
        GRPOTrainingConfig,
        PolicyLossTelemetry,
        TRLGRPOAdapter,
        compute_group_normalized_rewards,
        compute_policy_loss,
    )
except ImportError:  # pragma: no cover
    TRLGRPOAdapter = None  # type: ignore[assignment, misc]
    CustomGRPOTrainer = None  # type: ignore[assignment, misc]
    GRPOTrainingConfig = None  # type: ignore[assignment, misc]
    PolicyLossTelemetry = None  # type: ignore[assignment, misc]
    compute_group_normalized_rewards = None  # type: ignore[assignment]
    compute_policy_loss = None  # type: ignore[assignment]

__all__ = [
    "BaseRLAdapter",
    "CustomGRPOTrainer",
    "GRPOTrainingConfig",
    "PolicyLossTelemetry",
    "RolloutBatch",
    "SkyRLSQLEnvironment",
    "TRLGRPOAdapter",
    "compute_group_normalized_rewards",
    "compute_policy_loss",
]
