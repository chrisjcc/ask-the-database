"""Framework adapters for reinforcement learning."""

from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.adapters.trl_grpo import (
    TRLGRPOAdapter,
    compute_group_normalized_rewards,
)

__all__ = [
    "BaseRLAdapter",
    "RolloutBatch",
    "TRLGRPOAdapter",
    "compute_group_normalized_rewards",
]
