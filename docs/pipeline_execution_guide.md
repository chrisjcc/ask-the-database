# Step-by-Step Guide: End-to-End Training, Evaluation & Harness-Zero MVE Pipeline

This operational guide provides end-to-end instructions for running the complete `AskTheDatabase` training, evaluation, and Harness-Zero distillation pipeline on a remote compute server using `make` commands.

```
                    End-to-End Operational Pipeline
 ┌───────────────┐     ┌────────────────┐     ┌────────────────┐
 │ 1. Preprocess │ ──> │ 2. SFT Warmup  │ ──> │ 3. GRPO / RLVR │
 └───────────────┘     └────────────────┘     └────────────────┘
                                                       │
 ┌────────────────┐     ┌────────────────┐             ▼
 │ 6. MVE Report  │ <── │ 5. Harness SFT │ <── ┌────────────────┐
 │  (Gate: GO)    │     │  (Distillation)│     │ 4. K_SQL Collect│
 └────────────────┘     └────────────────┘     └────────────────┘
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
  - `data/processed/train.jsonl` (Normalized training split)
  - `data/processed/val.jsonl` (Held-out validation split)

---

## 2. Condition 2 Baseline: Conventional SFT on Base Model

Train `Qwen/Qwen2.5-Coder-7B-Instruct` on ground-truth static text-to-SQL pairs using LoRA ($r=16, \alpha=32$) via Hugging Face TRL:

```bash
make train-sft
```
* **Fast dry-run check (verifies configs, tokenizers, and LoRA layers without GPU compute):**
  ```bash
  python scripts/train_sft.py dry_run=true
  ```
* **Resulting Checkpoint:** `models/sft/checkpoint-final`

---

## 3. Multi-Turn RLVR / GRPO Policy Optimization

Train the agent policy using Group Relative Policy Optimization (GRPO) in `SkyRLSQLEnvironment`, optimizing multi-turn test-suite rewards and error-recovery bonuses:

```bash
make train-grpo
```
* **Fast dry-run check (verifies multi-turn environment stepping and group reward normalization):**
  ```bash
  python scripts/train_rlvr.py dry_run=true
  ```
* **Resulting Checkpoint:** `models/rlvr/final`

---

## 4. Harness-Zero Reviewed Trajectory Collection

Collect 500 stratified Spider training tasks (150 Medium, 200 Hard, 150 Extra Hard) where student proposals are intercepted and repaired step-by-step by the reference harness $\mathcal{K}_{\text{SQL}}$:

```bash
make collect-distillation
```
* **Teacher Backend Options:**
  - **Google Gemini 2.0 Flash (Default, Free Tier via `GEMINI_API_KEY`):**
    ```bash
    python scripts/collect_reviewed_trajectories.py \
      --teacher-backend litellm \
      --teacher-model gemini/gemini-2.0-flash
    ```
  - **Deterministic Pure Rule Engine (100% Offline, Zero Network Requests):**
    ```bash
    python scripts/collect_reviewed_trajectories.py \
      --teacher-backend rule
    ```
  - **Hermetic dry-run smoke test (5 sample episodes):**
    ```bash
    python scripts/collect_reviewed_trajectories.py --dry-run
    ```
* **Artifacts Generated:**
  - `data/distillation/reviewed_trajectories.jsonl` (ChatML formatted trajectories with reviewer thoughts masked)
  - `data/distillation/collection_metrics.json` (Intervention rates, rule trigger counts, and execution metrics)

---

## 5. Condition 4 Proposed: Harness-Zero Agent SFT

Fine-tune `Qwen/Qwen2.5-Coder-7B-Instruct` on the collected reviewed trajectories to internalize schema inspection and error recovery:

```bash
make train-harness-zero
```
* **Resulting Checkpoint:** `models/sft/harness_zero`

---

## 6. Multi-Benchmark Evaluation Across All 4 Conditions

Run interactive evaluation against the Spider Dev and robustness test suites across the four conditions:

```bash
# Condition 1: Base Student (Zero-Shot)
python scripts/evaluate_benchmark.py \
  evaluation.model_path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  evaluation.output_dir="results/cond1" \
  evaluation.summary_file="results/cond1/summary.json"

# Condition 2: Conventional SFT Baseline
python scripts/evaluate_benchmark.py \
  evaluation.model_path="models/sft/checkpoint-final" \
  evaluation.output_dir="results/cond2" \
  evaluation.summary_file="results/cond2/summary.json"

# Condition 3: Successful-Rollout Filtering Baseline (Rejection Sampling)
python scripts/evaluate_benchmark.py \
  evaluation.model_path="models/sft/filtered" \
  evaluation.output_dir="results/cond3" \
  evaluation.summary_file="results/cond3/summary.json"

# Condition 4: Harness-Zero SFT (Proposed)
python scripts/evaluate_benchmark.py \
  evaluation.model_path="models/sft/harness_zero" \
  evaluation.output_dir="results/cond4" \
  evaluation.summary_file="results/cond4/summary.json"
```

---

## 7. MVE Decision Gate Adjudication & Report Generation

Run the automated decision gate evaluator to compute empirical margins, test stop conditions, and publish the research report:

```bash
make evaluate-mve
```

This runs [`scripts/run_mve_evaluation.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/scripts/run_mve_evaluation.py) and outputs:
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

## 8. Summary of Makefile Operational Commands

| Command | Action | Output / Checkpoint |
| :--- | :--- | :--- |
| `make install` | Installs dependencies in editable mode | Virtualenv configured |
| `make lint` | Runs Ruff linter and strict Mypy type-check | Zero warnings |
| `make test` | Executes full Pytest unit test suite | 149/149 tests passing |
| `make preprocess` | Downloads and tokenizes Spider datasets | `data/processed/train.jsonl` |
| `make train-sft` | Trains SFT on static gold SQL (Cond 2) | `models/sft/checkpoint-final` |
| `make train-grpo` | Runs multi-turn RLVR / GRPO training | `models/rlvr/final` |
| `make collect-distillation` | Collects K_SQL reviewed trajectories | `data/distillation/reviewed_trajectories.jsonl` |
| `make train-harness-zero` | Trains SFT on reviewed trajectories (Cond 4) | `models/sft/harness_zero` |
| `make evaluate` | Evaluates models on Spider test suites | `results/evaluation_summary.json` |
| `make evaluate-mve` | Adjudicates Decision Gates (`GO`/`NO-GO`) | `docs/research/mve_harness_zero_results.md` |

---

## 9. Next Step: Full Scale Beyond MVE

Once the `make evaluate-mve` verdict outputs **`Adjudication Verdict: GO`**, you can scale beyond the MVE to full-cluster RLVR sweeps:
1. Warm-start multi-turn GRPO with the Condition 4 distilled checkpoint:
   ```bash
   python scripts/train_rlvr.py training.base_model_path="models/sft/harness_zero" training.max_steps=1000
   ```
2. Launch multi-GPU hyperparameter array sweeps on Slurm:
   ```bash
   sbatch scripts/slurm/launch_rlvr_sweep.sh
   ```
