"""Framework adapters for reinforcement learning."""

from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.adapters.skyrl_adapter import SkyRLSQLEnvironment

try:
    from ask_the_database.training.adapters.trl_grpo import (
        TRLGRPOAdapter,
        compute_group_normalized_rewards,
    )
except ImportError:  # pragma: no cover
    TRLGRPOAdapter = None  # type: ignore[assignment, misc]
    compute_group_normalized_rewards = None  # type: ignore[assignment]

__all__ = [
    "BaseRLAdapter",
    "RolloutBatch",
    "SkyRLSQLEnvironment",
    "TRLGRPOAdapter",
    "compute_group_normalized_rewards",
]
