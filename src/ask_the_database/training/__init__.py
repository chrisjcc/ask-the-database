"""Training and framework adapter modules."""

from ask_the_database.training.adapters.base import BaseRLAdapter, RolloutBatch
from ask_the_database.training.adapters.trl_grpo import (
    TRLGRPOAdapter,
    compute_group_normalized_rewards,
)
from ask_the_database.training.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    extract_sql_from_completion,
    format_chatml_messages,
    format_prompt_for_generation,
    load_sft_dataset,
)
from ask_the_database.training.sft_trainer import SFTWarmupPipeline

__all__ = [
    "BaseRLAdapter",
    "RolloutBatch",
    "TRLGRPOAdapter",
    "compute_group_normalized_rewards",
    "DEFAULT_SYSTEM_PROMPT",
    "extract_sql_from_completion",
    "format_chatml_messages",
    "format_prompt_for_generation",
    "load_sft_dataset",
    "SFTWarmupPipeline",
]
