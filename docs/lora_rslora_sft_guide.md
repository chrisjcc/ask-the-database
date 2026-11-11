# Standard LoRA and Rank-Stabilized LoRA (rsLoRA) Guide for SFT Training

This guide details the implementation, configuration, and empirical mechanics of **Standard LoRA** and **Rank-Stabilized LoRA (rsLoRA)** for Supervised Fine-Tuning (SFT) in the AskTheDatabase pipeline.

---

## 1. Overview & Mathematical Foundations

Low-Rank Adaptation (LoRA; Hu et al., 2021) freezes the pre-trained backbone parameter matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices:

$$W = W_0 + \Delta W = W_0 + s \cdot B A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$, and $s$ is the scaling factor determined by hyperparameter $\alpha$ (`lora_alpha`) and rank $r$ (`r`).

### Standard LoRA vs. Rank-Stabilized LoRA (rsLoRA)

In Standard LoRA, the adapter output is scaled inversely proportional to the rank $r$:

$$s_{\text{Standard}} = \frac{\alpha}{r}$$

While effective for modest ranks ($r \in \{4, 8, 16\}$), Kalajdzievski (2023) demonstrated that as rank $r$ increases, the learning dynamics of standard LoRA scale as $\mathcal{O}(1/r)$, leading to vanishing updates and representation collapse for larger ranks.

**Rank-Stabilized LoRA (rsLoRA)** replaces the $1/r$ scaling with a stable $1/\sqrt{r}$ scaling:

$$s_{\text{rsLoRA}} = \frac{\alpha}{\sqrt{r}}$$

| Adaptation Architecture | Scaling Factor ($s$) | Example ($r=16, \alpha=32$) | Scaling with Rank Increase | Target Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Standard LoRA** | $\displaystyle\frac{\alpha}{r}$ | $s = \frac{32}{16} = 2.0$ | $\mathcal{O}(1/r)$ (diminishing updates) | Baseline configurations, low rank ($r \le 16$) |
| **Rank-Stabilized LoRA (rsLoRA)** | $\displaystyle\frac{\alpha}{\sqrt{r}}$ | $s = \frac{32}{\sqrt{16}} = 8.0$ | $\mathcal{O}(1/\sqrt{r})$ (stable learning dynamics) | High-capacity adaptation ($r \ge 16, 32, 64$), complex schemas |

AskTheDatabase uses PEFT's native `use_rslora=True` implementation directly in `LoraConfig`, ensuring zero custom overhead and perfect framework compatibility.

---

## 2. Configuration via YAML

All LoRA and rsLoRA parameters are declared in `configs/training/sft.yaml` under the `training.lora` block:

```yaml
# -----------------------------------------------------------------------------
# LoRA Configuration
# -----------------------------------------------------------------------------

lora:
  enabled: true                            # Controls whether LoRA/PEFT adapters are used

  # Adapter configuration
  r: 16                                    # LoRA rank dimension (positive integer)
  lora_alpha: 32                           # LoRA scaling parameter (alpha > 0)
  lora_dropout: 0.05                       # Adapter dropout (in [0, 1))

  # false = Standard LoRA (s = alpha / r)
  # true  = Rank-Stabilized LoRA (rsLoRA, s = alpha / sqrt(r))
  use_rslora: false

  # Target modules for adapter injection (string shortcut or explicit list)
  target_modules: "all-linear"

  # Optional adapter configuration
  bias: "none"                             # Supported: "none", "all", "lora_only"
  modules_to_save: null                    # Optional additional modules to train & save

  # Initialization strategy
  init_lora_weights: true                  # true, "gaussian", "loftq", etc.
```

### Parameter Reference

- `enabled` (*bool*): When `true`, constructs a PEFT `LoraConfig` and trains parameter-efficient adapters. When `false`, disables PEFT (`peft_config=None`) and routes training through full fine-tuning.
- `r` (*int*): Rank dimension. Must be a positive integer ($r > 0$).
- `lora_alpha` (*int* or *float*): Scaling factor numerator. Must be positive ($\alpha > 0$).
- `lora_dropout` (*float*): Dropout probability applied to adapter outputs. Must lie in $[0, 1)$.
- `use_rslora` (*bool*): Enables Rank-Stabilized LoRA when `true`. Defaults to `false` for Standard LoRA.
- `target_modules` (*str* or *list[str]*): Projection layers to adapt. Supports PEFT string shortcut `"all-linear"` or explicit module lists (e.g. `["q_proj", "k_proj", "v_proj", "o_proj"]`).
- `bias` (*str*): Bias training policy. Supported PEFT values: `"none"`, `"all"`, `"lora_only"`.
- `modules_to_save` (*list[str]* or *null*): Additional modules (e.g. `["lm_head"]`) to train fully and persist in checkpoint.
- `init_lora_weights` (*bool* or *str*): Initialization scheme recognized by PEFT (`true`, `"gaussian"`, `"loftq"`).

---

## 3. Switching Between Standard LoRA and rsLoRA

You can switch between Standard LoRA and rsLoRA purely through configuration without modifying Python code:

### Standard LoRA (Default)
```yaml
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  use_rslora: false
```

### Rank-Stabilized LoRA (rsLoRA)
```yaml
lora:
  enabled: true
  r: 32
  lora_alpha: 64
  use_rslora: true
```

### Full Fine-Tuning (LoRA Disabled)
```yaml
lora:
  enabled: false
```

---

## 4. CLI & Hydra Overrides

Parameters can be overridden via Makefile or direct CLI command lines:

```bash
# Run SFT with rsLoRA enabled (r=32, alpha=64):
make train-sft ARGS="training.lora.use_rslora=true training.lora.r=32 training.lora.lora_alpha=64"

# Run SFT with Standard LoRA targeting specific projections:
make train-sft ARGS="training.lora.use_rslora=false training.lora.target_modules='[q_proj,v_proj]'"

# Run full fine-tuning (LoRA disabled):
make train-sft ARGS="training.lora.enabled=false training.learning_rate=1e-5"
```

---

## 5. Experiment Reproducibility & Telemetry

At the beginning of each SFT run, `SFTWarmupPipeline` logs the complete effective adapter configuration:

```text
================================================================================
Effective LoRA / PEFT Configuration:
  • Enabled          : True
  • Architecture     : rslora
  • Rank (r)         : 16
  • Alpha            : 32
  • Scaling Factor   : 8.0
  • Dropout          : 0.05
  • Target Modules   : all-linear
  • Bias             : none
  • use_rslora       : True
  • Init Weights     : True
  • Modules to Save  : None
================================================================================
```

The configuration is automatically persisted to disk alongside the checkpoint artifacts:
- `models/sft/final/lora_metadata.json`
- `models/sft/final/effective_lora_config.json`
- `models/sft/final/adapter_config.json` (includes `"use_rslora": true/false`)

When Weights & Biases (W&B) is enabled, the pipeline automatically records the architecture distinction:
- `peft.lora_type`: `"standard_lora"`, `"rslora"`, `"qlora_standard"`, `"qlora_rslora"`, or `"full_finetuning"`
- `peft.scaling_factor`: effective numerical scaling factor $s$
- `peft.use_rslora`: boolean flag
- `peft.quantization_enabled`: boolean flag
- `peft.quantization_type`: quantization format (e.g. `"4bit_nf4"` or `"8bit"`)

---

## 6. Quantized LoRA (QLoRA) Support

For memory-constrained environments or high-throughput scaling on consumer or cloud GPUs (e.g. CoreWeave, DESY, Slurm nodes), AskTheDatabase supports **4-bit NormalFloat (NF4)** and **8-bit bitsandbytes quantization** seamlessly integrated with both Standard LoRA and rsLoRA.

### YAML Configuration

In `configs/training/sft.yaml`:

```yaml
# -----------------------------------------------------------------------------
# Quantization (QLoRA) Configuration
# -----------------------------------------------------------------------------

quantization:
  enabled: true                            # Enable 4-bit / 8-bit QLoRA
  load_in_4bit: true                       # 4-bit precision (default for QLoRA)
  load_in_8bit: false                      # 8-bit precision (mutually exclusive with 4-bit)
  bnb_4bit_quant_type: "nf4"               # "nf4" (recommended) or "fp4"
  bnb_4bit_use_double_quant: true          # Nested / double quantization for memory savings
  bnb_4bit_compute_dtype: "bfloat16"       # Compute precision ("bfloat16", "float16", "float32")
```

### Combining QLoRA with rsLoRA

To run 4-bit QLoRA with Rank-Stabilized LoRA:

```bash
make train-sft ARGS="training.quantization.enabled=true training.lora.use_rslora=true training.lora.r=32 training.lora.lora_alpha=64"
```

---

## 7. Multi-GPU Execution & Device Placement

### The Cross-Device Tensor Error & Resolution

When training on multi-GPU servers (such as DESY worker nodes with 2x or 4x A100/H100 GPUs), standard Hugging Face pipelines that set `device_map="auto"` distribute model layers across GPUs (e.g., layers 0–14 on `cuda:0` and 15–28 on `cuda:1`). 

In single-process training with TRL's `SFTTrainer`, token reduction inside `compute_loss`:
```python
loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min=1)
```
creates `torch.as_tensor(num_tokens)` on the default device (`cuda:0`), while the final output loss tensor resides on the last GPU (`cuda:1`), resulting in:
```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

### Architectural Fixes in AskTheDatabase

1. **Single-Device Process Mapping**:
   The SFT pipeline automatically detects available GPUs and maps the model to `{"": local_rank}` (or `{"": current_device}`), preventing naive layer splitting across GPUs during training.
2. **`RobustSFTTrainer` Multi-GPU Safeguards**:
   - **`nn.DataParallel` Prevention**: When several GPUs are visible to a single process without a distributed runner (`accelerate`/`torchrun`), standard Hugging Face `Trainer` attempts to wrap models in `torch.nn.DataParallel`, which TRL explicitly forbids (`ValueError: SFTTrainer does not support nn.DataParallel...`). `RobustSFTTrainer` automatically clamps `n_gpu` to 1 in single-process mode, avoiding the `nn.DataParallel` trap.
   - **Cross-Device Recovery**: Defensively traps any cross-device tensor mismatches during `compute_loss()` and gracefully falls back to local per-device loss normalization if multi-GPU model sharding is explicitly forced.
3. **Environment & Makefile Defaults**:
   The `Makefile` targets (`make train-sft`, `make train-filtered`, `make train-harness-zero`) and `scripts/train_sft.py` default `CUDA_VISIBLE_DEVICES` to `0` when executing in single-process mode on multi-GPU hosts, ensuring clean device visibility out of the box.
4. **Explicit Device Map Overrides**:
   You can override device targeting via Hydra configuration:
   ```bash
   make train-sft ARGS="training.device_map='cuda:0'"
   ```
5. **Distributed Training (DDP via Accelerate)**:
   For multi-GPU data parallelism across all GPUs:
   ```bash
   accelerate launch --num_processes=2 scripts/train_sft.py training.lora.use_rslora=true
   ```
