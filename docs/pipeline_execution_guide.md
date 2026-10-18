# Step-by-Step Guide: End-to-End Training, Evaluation & Harness-Zero MVE Pipeline

This operational guide provides end-to-end instructions for running the complete `AskTheDatabase` training, evaluation, Minimum Viable Experiment (MVE), and Harness-Zero distillation pipeline on a remote compute server using `make` commands.

```
                      End-to-End Operational Pipeline & MVE Architecture
  ┌───────────────┐     ┌────────────────────────────────────────────────────────┐
  │ 1. Preprocess │ ──> │ 2. MVE Conditions (500 Samples / Unseen Schemas)       │
  └───────────────┘     │  • Cond 1: Base Student (Zero-Shot)                   │
                        │  • Cond 2: Conventional SFT Baseline (Gold Data)       │
                        │  • Cond 3: Rejection Sampling (Filtered Rollouts)     │
                        │  • Cond 4: Harness-Zero SFT (K_SQL Distillation)      │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
  ┌─────────────────┐                        ┌────────────────┐
  │ 4. Cond 5 RLVR  │ <───────────────────── │ 3. MVE Report  │
  │  (Cond 2 vs 4)  │   (Gate: GO Passed)    │ (Adjudication) │
  └─────────────────┘                        └────────────────┘
```

---

## 0. Initial Remote Server Setup

Clone the repository and configure an isolated Python 3.11 virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/chrisjcc/ask-the-database.git
cd ask-the-database

# 2. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install AskTheDatabase in editable mode with development and training dependencies
make install

# Optional (GPU Cluster): If you need a specific CUDA build of PyTorch matching your NVIDIA drivers:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Verify test suite and type safety
make lint
make test
```

### Configure Credentials (`.env`)
Copy the environment template and populate your API credentials:
```bash
cp .env.template .env
```
Populate the required keys in `.env`:
```bash
# Hugging Face token for downloading base models and Spider datasets
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Hugging Face Cache Directory (Essential on HPC clusters to avoid filling ~/.cache / home quota)
HF_HOME=/data/dust/user/clseitz/.cache/huggingface

# Google AI Studio key for free Gemini 2.0 Flash teacher review (0 token costs)
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxx

# Weights & Biases for live training telemetry and trajectory logging
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=ask-the-database
```

---

## 1. Data Acquisition & Preprocessing

Download the SQLite database archives, extract the Spider training and validation splits, and build verified JSONL dataset splits:

```bash
make preprocess
```
* **Artifacts Generated:**
  - `data/databases/*.sqlite` (Extracted SQLite database files)
  - `data/processed/train.jsonl` (Normalized training split, 6,997 instances)
  - `data/processed/val.jsonl` (Held-out validation split, 1,034 instances)

---

## 2. The 4-Condition Minimum Viable Experiment (MVE)

The Minimum Viable Experiment (MVE) establishes whether agent-as-harness distillation delivers measurable generalization improvements over conventional baselines on unseen database schemas *before* allocating cluster-scale compute.

The MVE systematically evaluates four controlled conditions:
1. **Condition 1**: Base Student (Zero-Shot) — unguided baseline without training.
2. **Condition 2**: Conventional SFT Baseline — fine-tuned on gold-standard Spider demonstrations (`make train-sft`).
3. **Condition 3**: Rejection Sampling Baseline — fine-tuned on self-generated rollouts filtered for test-suite correctness ($TS = 1.0$).
4. **Condition 4**: Harness-Zero SFT (Proposed) — fine-tuned on on-policy rollouts reviewed and repaired by the reference harness $\mathcal{K}_{\text{SQL}}$ (`make train-harness-zero`).

---

### Condition 1: Base Student (Zero-Shot)

Evaluate the unguided base model directly under the tool environment without fine-tuning:

```bash
# Evaluate Condition 1 on benchmark suites
python3 scripts/evaluate_benchmark.py \
  evaluation.model_path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  evaluation.output_dir="results/cond1" \
  evaluation.summary_file="results/cond1/summary.json"
```
*(If using the 1.5B model, replace with `evaluation.model_path="Qwen/Qwen2.5-Coder-1.5B-Instruct"`)*

---

### Condition 2: Conventional SFT Baseline

Train the base model on ground-truth static text-to-SQL pairs using LoRA ($r=16, \alpha=32$):

```bash
# Full dataset training (Default: Qwen2.5-Coder-7B-Instruct, ~15.2 GB download)
make train-sft

# Lightweight 1.5B alternative:
make train-sft ARGS="model=qwen2_5_coder_1_5b"

# Focused MVE Baseline: 500 gold demonstrations into MVE checkpoint directory
make train-sft ARGS="training.output_dir=models/mve/sft_baseline data.max_samples=500"

# Fast dry-run check:
make train-sft ARGS="dry_run=true"
```
* **Resulting Checkpoint:** `models/sft/checkpoint-final` (or `models/mve/sft_baseline`)

```bash
# Evaluate Condition 2
python3 scripts/evaluate_benchmark.py \
  evaluation.model_path="models/mve/sft_baseline" \
  evaluation.output_dir="results/cond2" \
  evaluation.summary_file="results/cond2/summary.json"
```

---

### Condition 3: Successful-Rollout Filtering Baseline (Rejection Sampling)

Sample trajectories generated by the student, filter out any failures, and train strictly on rollouts that achieve Test-Suite Correctness ($TS = 1.0$) with zero interventions:

```bash
# Step 1: Collect rejection sampling rollouts with rule engine
python3 scripts/collect_reviewed_trajectories.py \
  --output data/distillation/filtered_rollouts.jsonl \
  --metrics-output data/distillation/filtered_metrics.json \
  --teacher-backend rule

# Step 2: Train SFT on filtered successful rollouts
make train-filtered

# Or lightweight 1.5B alternative:
make train-filtered ARGS="model=qwen2_5_coder_1_5b"
* **Resulting Checkpoint:** `models/mve/filtered_rollouts`

```bash
# Evaluate Condition 3
python3 scripts/evaluate_benchmark.py \
  evaluation.model_path="models/mve/filtered_rollouts" \
  evaluation.output_dir="results/cond3" \
  evaluation.summary_file="results/cond3/summary.json"
```

---

### Condition 4: Harness-Zero SFT (Proposed Method)

Condition 4 evaluates the core contribution of the Harness-Zero paradigm: fine-tuning the student policy on on-policy rollouts reviewed and minimally corrected by the reference harness $\mathcal{K}_{\text{SQL}}$.

> **Why `make train-harness-zero` is pre-configured specifically for Condition 4:**  
> In this repository's Makefile convention:
> - `make train-sft`: fine-tunes the baseline model on static gold demonstrations (Condition 2).
> - `make train-harness-zero`: fine-tunes the proposed student on the Harness-Zero reviewed trajectory dataset (`data/distillation/reviewed_trajectories.jsonl`).
> Both commands invoke the shared `scripts/train_sft.py` engine with appropriate dataset overrides. By default, it fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` (or `1.5B` via `ARGS="model=qwen2_5_coder_1_5b"`).

```bash
# Step 1: Collect stratified reviewed trajectories (150 Medium, 200 Hard, 150 Extra)
make collect-distillation

# Fast dry-run smoke test (5 sample episodes):
python3 scripts/collect_reviewed_trajectories.py --dry-run
```
* **Teacher Backend Options:**
  - **Google Gemini 2.0 Flash (Default, Free Tier via `GEMINI_API_KEY`):**
    ```bash
    python3 scripts/collect_reviewed_trajectories.py \
      --teacher-backend litellm \
      --teacher-model gemini/gemini-2.0-flash
    ```
  - **Deterministic Pure Rule Engine (100% Offline, Zero Network Requests):**
    ```bash
    python3 scripts/collect_reviewed_trajectories.py \
      --teacher-backend rule
    ```

```bash
# Step 2: Fine-tune on reviewed trajectories
make train-harness-zero ARGS="training.output_dir=models/mve/harness_zero"

# Lightweight 1.5B alternative:
make train-harness-zero ARGS="model=qwen2_5_coder_1_5b training.output_dir=models/mve/harness_zero"

# Fast dry-run check:
make train-harness-zero ARGS="dry_run=true"
```
* **Resulting Checkpoint:** `models/mve/harness_zero` (or `models/sft/harness_zero`)

```bash
# Evaluate Condition 4
python3 scripts/evaluate_benchmark.py \
  evaluation.model_path="models/mve/harness_zero" \
  evaluation.output_dir="results/cond4" \
  evaluation.summary_file="results/cond4/summary.json"
```

---

## 3. MVE Decision Gate Adjudication & Report Generation

Run the automated decision gate evaluator to compute empirical margins across all 4 conditions, test formal stop conditions, and publish the research report:

```bash
make evaluate-mve
```

This runs [`scripts/run_mve_evaluation.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/scripts/run_mve_evaluation.py) against [`configs/distillation/mve_conditions.yaml`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/configs/distillation/mve_conditions.yaml) and outputs:
1. **Serialized JSON Summary**: `results/mve_evaluation_summary.json`
2. **Markdown Research Report**: `docs/research/mve_harness_zero_results.md` and `docs/research/mve_harness_zero_distillation.md`

### Decision Gate Thresholds:

* **Primary Gate (Generalization Gain)**:
  $$\Delta \text{TS}_{4-2} \ge +5.0\text{ pp} \quad \text{and} \quad \Delta \text{TS}_{4-3} \ge +3.0\text{ pp}$$
* **Behavioral Gate (Error Recovery)**:
  $$\text{Relative Gain in } R_{\text{rec}} \ge +20.0\% \text{ over Condition 2}$$
* **Efficiency Gate (Trajectory Turns)**:
  $$\Delta \text{Turns} \le +1.5 \text{ turns over Condition 1}$$
* **Stop Conditions**: Verifies that Condition 4 does not perform within $\pm 1.5\text{ pp}$ of Condition 2 and is not matched by rejection sampling alone.

---

## 4. Condition 5: Multi-Turn RLVR / GRPO Sample Efficiency Comparison

Once `make evaluate-mve` yields **`Adjudication Verdict: GO`**, execute Condition 5 to measure reinforcement learning sample efficiency and asymptotic correctness:

Train agent policies using Group Relative Policy Optimization (GRPO) in `SkyRLSQLEnvironment` comparing initialization from Condition 2 vs. Condition 4:

```bash
# Run A: RLVR initialized from Condition 2 baseline checkpoint
make train-grpo ARGS="training.model_path=models/mve/sft_baseline training.output_dir=models/rlvr/from_cond2 training.max_steps=200"

# Run B: RLVR initialized from Condition 4 Harness-Zero distilled checkpoint
make train-grpo ARGS="training.model_path=models/mve/harness_zero training.output_dir=models/rlvr/from_cond4 training.max_steps=200"

# Fast dry-run check:
make train-grpo ARGS="dry_run=true"
```
* **Resulting Checkpoints:** `models/rlvr/from_cond2` vs. `models/rlvr/from_cond4`

---

## 5. Scaling Beyond MVE: Full-Cluster Distributed Sweeps

With the MVE decision gates validated and the Condition 4 distilled checkpoint ready, scale multi-turn GRPO across Slurm GPU clusters:

```bash
# Launch multi-GPU hyperparameter array sweeps on Slurm:
sbatch scripts/slurm/launch_rlvr_sweep.sh
```

---

## 6. Summary of Makefile Operational Commands

| Command | Action | Output / Checkpoint |
| :--- | :--- | :--- |
| `make install` | Installs dependencies in editable mode | Virtualenv configured |
| `make lint` | Runs Ruff linter and strict Mypy type-check | Zero warnings |
| `make test` | Executes full Pytest unit test suite | 149/149 tests passing |
| `make preprocess` | Downloads and tokenizes Spider datasets | `data/processed/train.jsonl` |
| `make train-sft` | Trains Conventional SFT Baseline (Cond 2) | `models/mve/sft_baseline` |
| `make train-filtered` | Trains Rejection Sampling SFT (Cond 3) | `models/mve/filtered_rollouts` |
| `make train-harness-zero` | Trains Harness-Zero SFT (Cond 4) | `models/mve/harness_zero` |
| `make train-grpo` | Runs multi-turn GRPO / RLVR (Cond 5) | `models/rlvr/final` |
| `make collect-distillation` | Collects K_SQL reviewed trajectories | `data/distillation/reviewed_trajectories.jsonl` |
| `make evaluate` | Evaluates models on Spider test suites | `results/evaluation_summary.json` |
| `make evaluate-mve` | Adjudicates Decision Gates (`GO`/`NO-GO`) | `docs/research/mve_harness_zero_results.md` |
