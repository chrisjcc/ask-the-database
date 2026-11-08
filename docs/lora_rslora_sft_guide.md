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
- `peft.lora_type`: `"standard_lora"` vs. `"rslora"` vs. `"full_finetuning"`
- `peft.scaling_factor`: effective numerical scaling factor $s$
- `peft.use_rslora`: boolean flag
