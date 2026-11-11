"""CLI entrypoint for Supervised Fine-Tuning (SFT) Warmup of text-to-SQL policies."""

import json
import logging
import os
from pathlib import Path

# Auto-configure CUDA_VISIBLE_DEVICES for single-process training on multi-GPU servers:
# When executing directly without a distributed runner (accelerate/torchrun), default to GPU 0
# if CUDA_VISIBLE_DEVICES is not explicitly defined in the environment. This prevents Hugging Face
# Trainer from detecting multiple GPUs and attempting unsupported torch.nn.DataParallel.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size <= 1 and local_rank == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        out_dir = Path(training_args.output_dir or "models/sft/final")
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


if __name__ == "__main__":
    _normalize_cli_args()
    main()
