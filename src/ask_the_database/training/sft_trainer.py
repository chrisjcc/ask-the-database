"""Supervised Fine-Tuning (SFT) Warmup Trainer for SQL generation.

Configures LoRA / rsLoRA adapters on backbone language models and orchestrates
training using Hugging Face TRL SFTTrainer and PEFT.
"""

import inspect
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import is_bitsandbytes_available  # type: ignore[attr-defined]
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from ask_the_database.training.dataset import load_sft_dataset

logger = logging.getLogger(__name__)


def validate_lora_config(lora_cfg: Any) -> None:
    """Validate LoRA / rsLoRA configuration parameters.

    Ensures configuration compliance with PEFT specifications prior to model initialization.
    If LoRA is disabled (enabled=False), parameter-level validation is skipped to permit
    full fine-tuning without obstruction.

    Args:
        lora_cfg: DictConfig or dictionary containing LoRA configuration parameters.

    Raises:
        ValueError: If any parameter value violates bounds or expected types.
        RuntimeError: If the installed PEFT library does not support requested features (e.g. use_rslora).
    """
    if lora_cfg is None:
        return

    # 1. enabled flag validation
    enabled = getattr(lora_cfg, "enabled", True)
    if isinstance(enabled, str):
        if enabled.lower() in ("true", "1", "yes"):
            enabled = True
        elif enabled.lower() in ("false", "0", "no"):
            enabled = False
        else:
            raise ValueError(f"Invalid LoRA 'enabled' flag: must be a boolean, got {enabled!r}")
    elif not isinstance(enabled, bool):
        raise ValueError(
            f"Invalid LoRA 'enabled' flag: must be a boolean, got {enabled!r} ({type(enabled).__name__})"
        )

    # If LoRA is explicitly disabled, skip further LoRA parameter validation
    if not enabled:
        return

    # 2. Check installed PEFT version supports use_rslora
    init_params = inspect.signature(LoraConfig.__init__).parameters
    if "use_rslora" not in init_params:
        raise RuntimeError(
            "The installed PEFT version does not support 'use_rslora'. "
            "Please upgrade peft to version >= 0.9.0."
        )

    # 3. LoRA rank r (must be positive integer, not boolean)
    r = getattr(lora_cfg, "r", 16)
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError(f"Invalid LoRA rank 'r': must be a positive integer, got {r!r}")

    # 4. LoRA alpha lora_alpha (must be positive number, not boolean)
    lora_alpha = getattr(lora_cfg, "lora_alpha", 32)
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, (int, float)) or lora_alpha <= 0:
        raise ValueError(
            f"Invalid LoRA alpha 'lora_alpha': must be a positive number, got {lora_alpha!r}"
        )

    # 5. LoRA dropout lora_dropout (must be in interval [0, 1), not boolean)
    lora_dropout = getattr(lora_cfg, "lora_dropout", 0.05)
    if (
        isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (int, float))
        or not (0.0 <= lora_dropout < 1.0)
    ):
        raise ValueError(
            f"Invalid LoRA dropout 'lora_dropout': must be in the interval [0, 1), got {lora_dropout!r}"
        )

    # 6. use_rslora (must be boolean)
    use_rslora = getattr(lora_cfg, "use_rslora", False)
    if not isinstance(use_rslora, bool):
        raise ValueError(
            f"Invalid rsLoRA flag 'use_rslora': must be a boolean (True or False), got {use_rslora!r} ({type(use_rslora).__name__})"
        )

    # 7. target_modules (must be non-empty string or non-empty sequence of non-empty strings)
    target_modules = getattr(lora_cfg, "target_modules", "all-linear")
    if target_modules is None:
        raise ValueError(
            "Invalid LoRA target modules 'target_modules': cannot be None. Must be a non-empty string or non-empty list of module names."
        )
    if isinstance(target_modules, str):
        if not target_modules.strip():
            raise ValueError(
                "Invalid LoRA target modules 'target_modules': string cannot be empty."
            )
    elif isinstance(target_modules, (list, tuple, Sequence)):
        if len(target_modules) == 0:
            raise ValueError("Invalid LoRA target modules 'target_modules': list cannot be empty.")
        for item in target_modules:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Invalid module name in target_modules: expected non-empty string, got {item!r}"
                )
    else:
        raise ValueError(
            f"Invalid LoRA target modules 'target_modules': must be a string or list of strings, got {type(target_modules).__name__}"
        )

    # 8. bias (must be one of 'none', 'all', 'lora_only')
    bias = getattr(lora_cfg, "bias", "none")
    if bias not in ("none", "all", "lora_only"):
        raise ValueError(
            f"Invalid LoRA bias setting 'bias': must be one of ('none', 'all', 'lora_only'), got {bias!r}"
        )

    # 9. modules_to_save (null or sequence of non-empty strings)
    modules_to_save = getattr(lora_cfg, "modules_to_save", None)
    if modules_to_save is not None:
        if not isinstance(modules_to_save, (list, tuple, Sequence)) or isinstance(
            modules_to_save, str
        ):
            raise ValueError(
                f"Invalid modules_to_save: must be None or a list of module names, got {type(modules_to_save).__name__}"
            )
        for item in modules_to_save:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Invalid module name in modules_to_save: expected non-empty string, got {item!r}"
                )


def _get_cfg_val(cfg: Any, key: str, default: Any = None) -> Any:
    """Retrieve configuration value supporting both dict and object/DictConfig access."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _is_bf16_supported() -> bool:
    """Safely check if CUDA bf16 is supported without crashing if CUDA is not initialized."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def validate_quantization_config(quant_cfg: Any) -> None:
    """Validate 4-bit / 8-bit QLoRA configuration parameters.

    Ensures configuration compliance with BitsAndBytesConfig specifications prior to model loading.
    If quantization is disabled (enabled=False), validation is skipped.

    Args:
        quant_cfg: DictConfig or dictionary containing quantization parameters.

    Raises:
        ValueError: If parameter values violate bounds, types, or mutual exclusivity.
        ImportError: If quantization is enabled but bitsandbytes is not installed.
    """
    if quant_cfg is None:
        return

    enabled = _get_cfg_val(quant_cfg, "enabled", False)
    if isinstance(enabled, str):
        if enabled.lower() in ("true", "1", "yes"):
            enabled = True
        elif enabled.lower() in ("false", "0", "no"):
            enabled = False
        else:
            raise ValueError(
                f"Invalid quantization 'enabled' flag: must be a boolean, got {enabled!r}"
            )
    elif not isinstance(enabled, bool):
        raise ValueError(
            f"Invalid quantization 'enabled' flag: must be a boolean, got {enabled!r} ({type(enabled).__name__})"
        )

    if not enabled:
        return

    # Check bitsandbytes installation
    if not is_bitsandbytes_available():
        raise ImportError(
            "Quantization is enabled (QLoRA) but 'bitsandbytes' is not installed or available in the environment. "
            "Please install it using: pip install 'bitsandbytes>=0.43.0'."
        )

    load_in_4bit = _get_cfg_val(quant_cfg, "load_in_4bit", True)
    if not isinstance(load_in_4bit, bool):
        raise ValueError(f"Invalid 'load_in_4bit' flag: must be a boolean, got {load_in_4bit!r}")

    load_in_8bit = _get_cfg_val(quant_cfg, "load_in_8bit", False)
    if not isinstance(load_in_8bit, bool):
        raise ValueError(f"Invalid 'load_in_8bit' flag: must be a boolean, got {load_in_8bit!r}")

    if load_in_4bit and load_in_8bit:
        raise ValueError(
            "Conflicting quantization settings: cannot set both 'load_in_4bit=True' and 'load_in_8bit=True'. "
            "Choose either 4-bit (QLoRA) or 8-bit quantization."
        )

    if not load_in_4bit and not load_in_8bit:
        raise ValueError(
            "Quantization is enabled, but neither 'load_in_4bit' nor 'load_in_8bit' is True. "
            "Set either 'load_in_4bit: true' or 'load_in_8bit: true'."
        )

    if load_in_4bit:
        quant_type = str(_get_cfg_val(quant_cfg, "bnb_4bit_quant_type", "nf4")).lower()
        if quant_type not in ("nf4", "fp4"):
            raise ValueError(
                f"Invalid 'bnb_4bit_quant_type': must be 'nf4' or 'fp4', got {quant_type!r}"
            )

        use_double_quant = _get_cfg_val(quant_cfg, "bnb_4bit_use_double_quant", True)
        if not isinstance(use_double_quant, bool):
            raise ValueError(
                f"Invalid 'bnb_4bit_use_double_quant': must be a boolean, got {use_double_quant!r}"
            )

        compute_dtype = str(_get_cfg_val(quant_cfg, "bnb_4bit_compute_dtype", "bfloat16")).lower()
        if compute_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"Invalid 'bnb_4bit_compute_dtype': must be one of ('bfloat16', 'float16', 'float32'), got {compute_dtype!r}"
            )


class RobustSFTTrainer(SFTTrainer):
    """SFTTrainer subclass ensuring robust execution and loss computation across devices.

    Addresses multi-GPU edge cases in Hugging Face TRL:
    1. Prevents TRL SFTTrainer from raising ValueError('...does not support nn.DataParallel...')
       when multiple GPUs are visible in single-process execution, by ensuring n_gpu is clamped to 1.
    2. Defensively handles cross-device tensor mismatch in compute_loss if model outputs reside
       on another GPU from the default token accumulation tensor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize RobustSFTTrainer with single-process multi-GPU safeguards."""
        training_args = kwargs.get("args")
        if training_args is None and len(args) > 1:
            training_args = args[1]

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if world_size <= 1 and local_rank == -1 and training_args is not None:
            if getattr(training_args, "n_gpu", 1) > 1 or getattr(training_args, "_n_gpu", 1) > 1:
                logger.info(
                    "Detected multiple visible GPUs in single-process mode (n_gpu=%s). Clamping n_gpu to 1 "
                    "to prevent unsupported nn.DataParallel in TRL.",
                    getattr(training_args, "n_gpu", "unknown"),
                )
            try:
                training_args.__class__ = type(
                    "SingleDeviceSFTConfig",
                    (training_args.__class__,),
                    {"n_gpu": property(lambda self: 1)},
                )
            except Exception as exc:
                logger.debug("Could not rebind __class__ for n_gpu property: %s", exc)
            training_args._n_gpu = 1

        super().__init__(*args, **kwargs)

        if world_size <= 1 and local_rank == -1 and hasattr(self, "args") and self.args is not None:
            self.args._n_gpu = 1

    def compute_loss(
        self,
        model: PreTrainedModel | torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Compute training loss with defensive handling of cross-device tensor splits in TRL."""
        try:
            return super().compute_loss(  # type: ignore[no-untyped-call]
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        except RuntimeError as err:
            if "same device" in str(err) and num_items_in_batch is not None:
                logger.warning(
                    "Detected cross-device tensor mismatch in TRL compute_loss (%s). "
                    "Retrying loss computation with local per-device loss normalization.",
                    err,
                )
                return super().compute_loss(  # type: ignore[no-untyped-call]
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=None
                )
            raise


class SFTWarmupPipeline:
    """Orchestrates Supervised Fine-Tuning with LoRA / rsLoRA for SQL translation policies."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize SFT pipeline with Hydra configuration.

        Args:
            config: Full Hydra configuration including model, training, and paths.
        """
        self.cfg = config
        self.output_dir = Path(getattr(config.training, "output_dir", "models/sft/final"))

    def build_lora_config(self) -> LoraConfig | None:
        """Build PEFT LoRA configuration dynamically from configuration.

        Constructs either Standard LoRA (use_rslora=False) or Rank-Stabilized LoRA
        (use_rslora=True) based on Hydra configuration. Returns None if LoRA is disabled.

        Returns:
            LoraConfig if LoRA is enabled, or None if LoRA is disabled (full fine-tuning).
        """
        lora_cfg = getattr(self.cfg.training, "lora", None)
        if lora_cfg is None:
            logger.info(
                "No explicit 'lora' section found in config; defaulting to Standard LoRA (r=16, alpha=32)."
            )
            return LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_rslora=False,
                task_type=TaskType.CAUSAL_LM,
            )

        validate_lora_config(lora_cfg)

        enabled = bool(getattr(lora_cfg, "enabled", True))
        if not enabled:
            logger.info(
                "LoRA adapter training is disabled (enabled=False). Running full fine-tuning."
            )
            return None

        r = int(getattr(lora_cfg, "r", 16))
        lora_alpha: int = int(getattr(lora_cfg, "lora_alpha", 32))
        lora_dropout = float(getattr(lora_cfg, "lora_dropout", 0.05))
        use_rslora = bool(getattr(lora_cfg, "use_rslora", False))

        target_modules_raw = getattr(lora_cfg, "target_modules", "all-linear")
        if isinstance(target_modules_raw, str):
            target_modules: str | list[str] = target_modules_raw
        else:
            target_modules = list(target_modules_raw)

        bias_val = str(getattr(lora_cfg, "bias", "none"))
        bias: Literal["none", "all", "lora_only"] = cast(
            Literal["none", "all", "lora_only"], bias_val
        )

        modules_to_save_raw = getattr(lora_cfg, "modules_to_save", None)
        modules_to_save = list(modules_to_save_raw) if modules_to_save_raw is not None else None

        init_lora_weights = getattr(lora_cfg, "init_lora_weights", True)

        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            modules_to_save=modules_to_save,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            task_type=TaskType.CAUSAL_LM,
        )

    def build_quantization_config(self) -> BitsAndBytesConfig | None:
        """Build BitsAndBytesConfig dynamically from Hydra configuration.

        Supports 4-bit (QLoRA) and 8-bit quantization. Returns None if quantization is disabled.
        """
        quant_cfg = getattr(self.cfg.training, "quantization", None)
        if quant_cfg is None:
            quant_cfg = getattr(self.cfg, "quantization", None)

        if quant_cfg is None:
            return None

        validate_quantization_config(quant_cfg)

        enabled = bool(_get_cfg_val(quant_cfg, "enabled", False))
        if not enabled:
            return None

        load_in_8bit = bool(_get_cfg_val(quant_cfg, "load_in_8bit", False))

        if load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]

        compute_dtype_str = str(
            _get_cfg_val(quant_cfg, "bnb_4bit_compute_dtype", "bfloat16")
        ).lower()
        compute_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(compute_dtype_str, torch.bfloat16)

        return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type=str(_get_cfg_val(quant_cfg, "bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                _get_cfg_val(quant_cfg, "bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    @staticmethod
    def get_lora_metadata(lora_cfg: LoraConfig | None, quant_cfg: Any = None) -> dict[str, Any]:
        """Extract structured metadata distinguishing standard_lora, rslora, qlora, and full_finetuning.

        Calculates effective scaling factor:
          Standard LoRA : s = alpha / r
          rsLoRA        : s = alpha / sqrt(r)
        """
        is_quantized = False
        quant_type = "none"
        compute_dtype_str = None
        if quant_cfg is not None and _get_cfg_val(quant_cfg, "enabled", False):
            is_quantized = True
            if _get_cfg_val(quant_cfg, "load_in_8bit", False):
                quant_type = "8bit"
            else:
                bnb_type = str(_get_cfg_val(quant_cfg, "bnb_4bit_quant_type", "nf4")).lower()
                quant_type = f"4bit_{bnb_type}"
            compute_dtype_str = str(_get_cfg_val(quant_cfg, "bnb_4bit_compute_dtype", "bfloat16"))

        if lora_cfg is None:
            lora_type = "full_finetuning"
            if is_quantized:
                lora_type = "quantized_full_finetuning"
            return {
                "lora_enabled": False,
                "lora_type": lora_type,
                "quantization_enabled": is_quantized,
                "quantization_type": quant_type,
                "quantization_compute_dtype": compute_dtype_str,
                "r": None,
                "lora_alpha": None,
                "scaling_factor": None,
                "lora_dropout": None,
                "target_modules": None,
                "bias": None,
                "use_rslora": False,
                "init_lora_weights": None,
                "modules_to_save": None,
            }

        is_rslora = bool(getattr(lora_cfg, "use_rslora", False))
        r = int(lora_cfg.r)
        alpha = float(lora_cfg.lora_alpha)
        scaling = (alpha / (r**0.5)) if is_rslora else (alpha / r)

        target_modules: Any = lora_cfg.target_modules
        if isinstance(target_modules, set):
            target_modules = sorted(target_modules)
        elif isinstance(target_modules, (list, tuple)):
            target_modules = list(target_modules)

        modules_to_save: Any = lora_cfg.modules_to_save
        if isinstance(modules_to_save, (list, tuple)):
            modules_to_save = list(modules_to_save)

        if is_quantized:
            lora_type = "qlora_rslora" if is_rslora else "qlora_standard"
        else:
            lora_type = "rslora" if is_rslora else "standard_lora"

        return {
            "lora_enabled": True,
            "lora_type": lora_type,
            "quantization_enabled": is_quantized,
            "quantization_type": quant_type,
            "quantization_compute_dtype": compute_dtype_str,
            "r": r,
            "lora_alpha": alpha if not alpha.is_integer() else int(alpha),
            "scaling_factor": round(float(scaling), 6),
            "lora_dropout": float(lora_cfg.lora_dropout),
            "target_modules": target_modules,
            "bias": str(lora_cfg.bias),
            "use_rslora": is_rslora,
            "init_lora_weights": getattr(lora_cfg, "init_lora_weights", True),
            "modules_to_save": modules_to_save,
        }

    def log_effective_lora_config(
        self, lora_cfg: LoraConfig | None, quant_cfg: Any = None
    ) -> dict[str, Any]:
        """Log the complete effective LoRA / QLoRA configuration at the beginning of each SFT run."""
        if quant_cfg is None:
            quant_cfg = getattr(self.cfg.training, "quantization", None) or getattr(
                self.cfg, "quantization", None
            )
        metadata = self.get_lora_metadata(lora_cfg, quant_cfg)
        logger.info(
            "================================================================================\n"
            "Effective LoRA / PEFT / Quantization Configuration:\n"
            "  • Enabled          : %s\n"
            "  • Architecture     : %s\n"
            "  • Quantization     : %s (Type: %s, Dtype: %s)\n"
            "  • Rank (r)         : %s\n"
            "  • Alpha            : %s\n"
            "  • Scaling Factor   : %s\n"
            "  • Dropout          : %s\n"
            "  • Target Modules   : %s\n"
            "  • Bias             : %s\n"
            "  • use_rslora       : %s\n"
            "  • Init Weights     : %s\n"
            "  • Modules to Save  : %s\n"
            "================================================================================",
            metadata["lora_enabled"],
            metadata["lora_type"],
            metadata["quantization_enabled"],
            metadata["quantization_type"],
            metadata["quantization_compute_dtype"],
            metadata["r"],
            metadata["lora_alpha"],
            metadata["scaling_factor"],
            metadata["lora_dropout"],
            metadata["target_modules"],
            metadata["bias"],
            metadata["use_rslora"],
            metadata["init_lora_weights"],
            metadata["modules_to_save"],
        )
        return metadata

    def save_lora_metadata(self, output_dir: Path, metadata: dict[str, Any]) -> Path:
        """Preserve effective LoRA configuration alongside training outputs and artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_file = output_dir / "lora_metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2) + "\n")
        (output_dir / "effective_lora_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        return meta_file

    def build_training_args(self, output_dir: str | Path | None = None) -> SFTConfig:
        """Construct Hugging Face SFTConfig / TrainingArguments."""
        out_dir = str(output_dir or self.output_dir)
        training_cfg = self.cfg.training

        # Precision detection:
        # Default to bf16 if supported by hardware (Ampere+, A100, H100) or requested in config.
        # GradScaler is only enabled for fp16; passing fp16 with bfloat16 weights/gradients causes
        # PyTorch to crash with NotImplementedError ("_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16').
        is_cuda = torch.cuda.is_available()
        cuda_bf16 = _is_bf16_supported()

        bf16_cfg = getattr(training_cfg, "bf16", None)
        fp16_cfg = getattr(training_cfg, "fp16", None)

        if fp16_cfg is True:
            bf16 = False
            fp16 = is_cuda
        elif bf16_cfg is True or (bf16_cfg is not False and cuda_bf16):
            bf16 = cuda_bf16
            fp16 = False
        else:
            bf16 = False
            fp16 = bool(fp16_cfg) and is_cuda

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

        if report_to == "wandb":
            try:
                import wandb  # noqa: F401

                project_name = getattr(wandb_cfg, "project", None) or getattr(
                    self.cfg, "project_name", "ask-the-database"
                )
                if project_name and not os.environ.get("WANDB_PROJECT"):
                    os.environ["WANDB_PROJECT"] = str(project_name)
            except ImportError:
                logger.warning(
                    "Weights & Biases (wandb) is not installed in the environment. "
                    "Falling back to report_to='none'. Install with `pip install wandb`."
                )
                report_to = "none"

        max_length = int(getattr(training_cfg, "max_seq_length", 2048))

        sft_args = SFTConfig(
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

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if world_size <= 1 and local_rank == -1:
            try:
                sft_args.__class__ = type(
                    "SingleDeviceSFTConfig",
                    (sft_args.__class__,),
                    {"n_gpu": property(lambda self: 1)},
                )
            except Exception as exc:
                logger.debug("Could not rebind __class__ for n_gpu property on SFTConfig: %s", exc)
            sft_args._n_gpu = 1

        return sft_args

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        """Load tokenizer and set padding token."""
        trust_remote = bool(getattr(self.cfg.model, "trust_remote_code", True))
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, model_name: str) -> PreTrainedModel:
        """Load base causal language model with appropriate precision, device mapping, and quantization."""
        trust_remote = bool(getattr(self.cfg.model, "trust_remote_code", True))

        # Device mapping:
        # Avoid device_map="auto" by default during training, which shards layers across multiple GPUs
        # and causes cross-device tensor mismatch in TRL SFTTrainer.compute_loss.
        device_map = getattr(self.cfg.model, "device_map", None)
        if device_map is None:
            device_map = getattr(self.cfg.training, "device_map", None)
        if device_map is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device_target = (
                    local_rank
                    if local_rank < torch.cuda.device_count()
                    else torch.cuda.current_device()
                )
                device_map = {"": device_target}
            else:
                device_map = None

        dtype = torch.bfloat16 if _is_bf16_supported() else torch.float32

        quant_config = self.build_quantization_config()

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )

        if quant_config is not None:
            gradient_checkpointing = bool(
                getattr(self.cfg.training, "gradient_checkpointing", True)
            )
            model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
                model, use_gradient_checkpointing=gradient_checkpointing
            )

        return model

    def train(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> Path:
        """Execute SFT training and save model checkpoint.

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
        quant_cfg = getattr(self.cfg.training, "quantization", None) or getattr(
            self.cfg, "quantization", None
        )
        lora_meta = self.log_effective_lora_config(lora_config, quant_cfg)
        self.save_lora_metadata(output_dir, lora_meta)

        training_args = self.build_training_args(output_dir)

        # Update Weights & Biases configuration if telemetry active
        wandb_cfg = getattr(self.cfg.training, "wandb", None)
        if wandb_cfg and getattr(wandb_cfg, "enabled", False):
            try:
                import wandb

                if wandb.run is not None:
                    wandb.run.config.update({"peft": lora_meta}, allow_val_change=True)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.debug("W&B telemetry update skipped: %s", e)

        # Delegate PEFT wrapping to SFTTrainer via peft_config argument.
        # If model is already a PeftModel instance (e.g. in test mock), pass peft_config=None.
        peft_cfg_for_trainer = lora_config if not isinstance(model, PeftModel) else None

        trainer_cls = (
            RobustSFTTrainer if SFTTrainer is RobustSFTTrainer.__bases__[0] else SFTTrainer
        )
        trainer = trainer_cls(
            model=model,
            peft_config=peft_cfg_for_trainer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        if trainer.model is not None:
            print_fn = getattr(trainer.model, "print_trainable_parameters", None)
            if callable(print_fn):
                print_fn()

        logger.info("Starting SFT training run...")
        trainer.train()

        logger.info("Saving SFT checkpoint to %s", output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        self.save_lora_metadata(output_dir, lora_meta)

        return output_dir
