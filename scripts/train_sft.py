"""CLI entrypoint for Supervised Fine-Tuning (SFT) Warmup of text-to-SQL policies."""

import json
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
    logger.info(
        "Initializing SFT Warmup Pipeline with configuration:\n%s",
        OmegaConf.to_yaml(cfg),
    )

    pipeline = SFTWarmupPipeline(cfg)

    # Dry-run validation mode for CI and resource-constrained environments
    dry_run = bool(getattr(cfg, "dry_run", False))
    if dry_run:
        logger.info("Dry-run mode active. Validating dataset and configs...")
        custom_dataset = getattr(cfg.training, "dataset_path", None) or getattr(
            cfg.training, "train_data_path", None
        )
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
        lora_meta = pipeline.log_effective_lora_config(lora_cfg)
        if lora_cfg is not None:
            logger.info(
                "Validated LoRA config: architecture=%s, target_modules=%s, r=%d, alpha=%s, use_rslora=%s",
                lora_meta["lora_type"],
                lora_cfg.target_modules,
                lora_cfg.r,
                lora_cfg.lora_alpha,
                getattr(lora_cfg, "use_rslora", False),
            )
        else:
            logger.info("Validated configuration: LoRA disabled (full fine-tuning mode).")

        training_args = pipeline.build_training_args()
        logger.info("Validated SFTConfig output_dir: %s", training_args.output_dir)

        # Ensure output directory exists for DVC / workflow tracking
        out_dir = Path(training_args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_lora_metadata(out_dir, lora_meta)

        if lora_cfg is not None:
            target_mods = (
                list(lora_cfg.target_modules)
                if isinstance(lora_cfg.target_modules, (set, list))
                else lora_cfg.target_modules
            )
            adapter_info = {
                "peft_type": "LORA",
                "r": lora_cfg.r,
                "lora_alpha": lora_cfg.lora_alpha,
                "lora_dropout": lora_cfg.lora_dropout,
                "target_modules": target_mods,
                "bias": lora_cfg.bias,
                "use_rslora": getattr(lora_cfg, "use_rslora", False),
                "dry_run": True,
            }
            (out_dir / "adapter_config.json").write_text(json.dumps(adapter_info, indent=2) + "\n")
            (out_dir / "adapter_model.safetensors").touch()
        else:
            (out_dir / "config.json").write_text('{"full_finetuning": true, "dry_run": true}\n')
        logger.info("Dry-run completed successfully.")
        return

    # Full training execution
    output_dir = pipeline.train()
    logger.info("SFT Warmup completed successfully. Checkpoint saved to: %s", output_dir)


if __name__ == "__main__":
    main()
