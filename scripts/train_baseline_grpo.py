"""CLI entrypoint for single-turn GRPO baseline reinforcement learning."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ask_the_database.rewards.rubric import SQLRewardRubric
from ask_the_database.training.adapters.base import RolloutBatch
from ask_the_database.training.adapters.trl_grpo import TRLGRPOAdapter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run single-turn GRPO baseline training using Hydra configuration."""
    logger.info(
        "Initializing GRPO Baseline Training with configuration:\n%s",
        OmegaConf.to_yaml(cfg),
    )

    rubric = SQLRewardRubric(getattr(cfg, "reward", None))
    adapter = TRLGRPOAdapter(cfg, rubric)

    dry_run = bool(getattr(cfg, "dry_run", False))
    if dry_run:
        logger.info(
            "Dry-run mode active. Validating reward functions and adapter configuration..."
        )
        training_args = adapter.build_training_args()
        logger.info(
            "Validated GRPOConfig: output_dir=%s, num_generations=%d, beta=%f",
            training_args.output_dir,
            training_args.num_generations,
            training_args.beta,
        )

        # Test reward calculation on sample batch
        mock_batch = RolloutBatch(
            prompts=["What is the count?"],
            completions=["```sql\nSELECT 1;\n```"],
            gold_sqls=["SELECT 1;"],
            db_paths=[":memory:"],
        )
        rewards = adapter.compute_rewards(mock_batch)
        logger.info("Validated mock batch reward: %s", rewards)

        out_dir = Path(training_args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(str(out_dir))
        logger.info("Dry-run completed successfully.")
        return

    logger.info("Full GRPO training pipeline initialized.")


if __name__ == "__main__":
    main()
