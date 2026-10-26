"""Supervised Fine-Tuning (SFT) Warmup Trainer for SQL generation.

Configures LoRA / QLoRA adapters on backbone language models and orchestrates
training using Hugging Face TRL SFTTrainer.
"""

import logging
from pathlib import Path
from typing import Any, Literal, cast

import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from ask_the_database.training.dataset import load_sft_dataset

logger = logging.getLogger(__name__)


class SFTWarmupPipeline:
    """Orchestrates Supervised Fine-Tuning with LoRA for SQL translation policies."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize SFT pipeline with Hydra configuration.

        Args:
            config: Full Hydra configuration including model, training, and paths.
        """
        self.cfg = config
        self.output_dir = Path(getattr(config.training, "output_dir", "models/sft/final"))

    def build_lora_config(self) -> LoraConfig:
        """Build PEFT LoRA configuration targeting linear projection layers."""
        lora_cfg = getattr(self.cfg.training, "lora", None)
        r = int(getattr(lora_cfg, "r", 16)) if lora_cfg else 16
        lora_alpha = int(getattr(lora_cfg, "lora_alpha", 32)) if lora_cfg else 32
        lora_dropout = float(getattr(lora_cfg, "lora_dropout", 0.05)) if lora_cfg else 0.05
        bias_val = str(getattr(lora_cfg, "bias", "none")) if lora_cfg else "none"
        if bias_val not in ("none", "all", "lora_only"):
            bias_val = "none"
        bias: Literal["none", "all", "lora_only"] = cast(
            Literal["none", "all", "lora_only"], bias_val
        )
        target_modules = list(
            getattr(
                lora_cfg,
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        )

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

    def build_training_args(self, output_dir: str | Path | None = None) -> SFTConfig:
        """Construct Hugging Face SFTConfig / TrainingArguments."""
        out_dir = str(output_dir or self.output_dir)
        training_cfg = self.cfg.training

        # Detect compute capabilities defensively
        is_cuda = torch.cuda.is_available()
        bf16_requested = bool(getattr(training_cfg, "bf16", False))
        bf16 = bf16_requested and is_cuda and torch.cuda.is_bf16_supported()
        fp16 = not bf16 and is_cuda

        max_steps = int(getattr(training_cfg, "max_steps", -1))
        epochs = float(getattr(training_cfg, "num_train_epochs", 3))
        batch_size = int(getattr(training_cfg, "per_device_train_batch_size", 2))
        grad_accum = int(getattr(training_cfg, "gradient_accumulation_steps", 8))
        lr = float(getattr(training_cfg, "learning_rate", 2e-5))
        warmup_steps = int(getattr(training_cfg, "warmup_steps", 10))
        weight_decay = float(getattr(training_cfg, "weight_decay", 0.01))
        logging_steps = int(getattr(training_cfg, "logging_steps", 5))
        save_strategy = str(getattr(training_cfg, "save_strategy", "epoch"))
        report_to = str(getattr(training_cfg, "report_to", "wandb"))
        wandb_cfg = getattr(training_cfg, "wandb", None)
        if wandb_cfg is not None and hasattr(wandb_cfg, "enabled"):
            if not wandb_cfg.enabled:
                report_to = "none"
            elif report_to == "none":
                report_to = "wandb"
        max_length = int(getattr(training_cfg, "max_seq_length", 2048))

        return SFTConfig(
            output_dir=out_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            bf16=bf16,
            fp16=fp16,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to=report_to,
            max_steps=max_steps,
            max_length=max_length,
        )

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        """Load tokenizer and set padding token."""
        trust_remote = bool(getattr(self.cfg.model, "trust_remote_code", True))
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, model_name: str) -> PreTrainedModel:
        """Load base causal language model with appropriate precision."""
        trust_remote = bool(getattr(self.cfg.model, "trust_remote_code", True))
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote,
        )
        return model

    def train(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> Path:
        """Execute SFT training and save LoRA checkpoint.

        Args:
            model: Optional preloaded model (e.g. for testing).
            tokenizer: Optional preloaded tokenizer.
            train_dataset: Optional training Dataset.
            eval_dataset: Optional evaluation Dataset.

        Returns:
            Path to saved output checkpoint directory.
        """
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(self.cfg.model, "name", "Qwen/Qwen2.5-Coder-7B-Instruct")

        if tokenizer is None:
            tokenizer = self.load_tokenizer(model_name)

        if train_dataset is None:
            custom_dataset = getattr(self.cfg.training, "dataset_path", None) or getattr(
                self.cfg.training, "train_data_path", None
            )
            train_path = (
                Path(custom_dataset)
                if custom_dataset
                else Path(self.cfg.paths.processed_dir) / "train.jsonl"
            )
            train_dataset = load_sft_dataset(str(train_path))

        if eval_dataset is None:
            custom_val = getattr(self.cfg.training, "val_dataset_path", None)
            val_path = (
                Path(custom_val) if custom_val else Path(self.cfg.paths.processed_dir) / "val.jsonl"
            )
            if val_path.is_file():
                eval_dataset = load_sft_dataset(str(val_path))

        if model is None:
            model = self.load_model(model_name)

        lora_config = self.build_lora_config()
        peft_model: Any = get_peft_model(model, lora_config)
        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()

        training_args = self.build_training_args(output_dir)

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting SFT training run...")
        trainer.train()

        logger.info("Saving SFT adapter checkpoint to %s", output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        return output_dir
