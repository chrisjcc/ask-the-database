"""CLI entrypoint for Supervised Fine-Tuning (SFT) Warmup of text-to-SQL policies."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ask_the_database.training.dataset import load_sft_dataset
from ask_the_database.training.sft_trainer import SFTWarmupPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run SFT warmup training using Hydra configuration."""
    logger.info("Initializing SFT Warmup Pipeline with configuration:\n%s", OmegaConf.to_yaml(cfg))

    pipeline = SFTWarmupPipeline(cfg)

    # Dry-run validation mode for CI and resource-constrained environments
    dry_run = bool(getattr(cfg, "dry_run", False))
    if dry_run:
        logger.info("Dry-run mode active. Validating dataset and configs...")
        custom_dataset = getattr(cfg.training, "dataset_path", None)
        train_path = (
            Path(custom_dataset)
            if custom_dataset
            else Path(cfg.paths.processed_dir) / "train.jsonl"
        )
        custom_val = getattr(cfg.training, "val_dataset_path", None)
        val_path = Path(custom_val) if custom_val else Path(cfg.paths.processed_dir) / "val.jsonl"

        train_ds = load_sft_dataset(str(train_path), max_samples=5)
        logger.info("Loaded dry-run training samples: %d", len(train_ds))
        if val_path.is_file():
            val_ds = load_sft_dataset(str(val_path), max_samples=5)
            logger.info("Loaded dry-run validation samples: %d", len(val_ds))

        lora_cfg = pipeline.build_lora_config()
        logger.info(
            "Validated LoRA config: target_modules=%s, r=%d, alpha=%d",
            lora_cfg.target_modules,
            lora_cfg.r,
            lora_cfg.lora_alpha,
        )

        training_args = pipeline.build_training_args()
        logger.info("Validated SFTConfig output_dir: %s", training_args.output_dir)

        # Ensure output directory exists for DVC / workflow tracking
        out_dir = Path(training_args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter_config.json").write_text('{"peft_type": "LORA", "dry_run": true}\n')
        (out_dir / "adapter_model.safetensors").touch()
        logger.info("Dry-run completed successfully.")
        return

    # Full training execution
    output_dir = pipeline.train()
    logger.info("SFT Warmup completed successfully. Checkpoint saved to: %s", output_dir)


if __name__ == "__main__":
    main()
