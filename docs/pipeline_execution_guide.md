# Step-by-Step Guide: End-to-End Training & Evaluation Pipeline

This operational guide provides end-to-end instructions for running the complete `AskTheDatabase` training, distillation, reinforcement learning, and evaluation pipeline on remote compute servers or HPC GPU clusters using `make` commands.

> [!NOTE]
> **Research & Experimental Findings:**  
> For the theoretical feasibility study, 4-condition Minimum Viable Experiment (MVE) validation protocol, decision gate adjudications, and RLVR sample efficiency analysis, refer to the dedicated research track in [`docs/research/harness-zero/mve_final_report.md`](research/harness-zero/mve_final_report.md) and [`docs/research/harness-zero/feasibility_study.md`](research/harness-zero/feasibility_study.md).

```
                      End-to-End Operational Pipeline Architecture
   ┌─────────────────┐       ┌────────────────────────────────────────────────────────┐
   │ 0. Server Setup │ ───>  │ 1. Data Preprocessing (Spider Schemas & Splits)        │
   └─────────────────┘       └───────────────────────────┬────────────────────────────┘
                                                         │
                                                         ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │ 2. Supervised Fine-Tuning & Distillation                                         │
   │    • 2.1 Conventional SFT Warmup (`make train-sft`)                              │
   │    • 2.2 Harness-Zero Trajectory Distillation (`make collect-distillation` / SFT)│
   └─────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │ 3. Multi-Turn RLVR Agent Training (SkyRL / GRPO in `SkyRLSQLEnvironment`)        │
   │    • 3.1 Fast Dry-Run Smoke Test (`make train-grpo ARGS="dry_run=true"`)         │
   │    • 3.2 Single-Node Policy Training (`make train-grpo`)                         │
   │    • 3.3 Multi-GPU Cluster Sweeps (`sbatch scripts/slurm/launch_rlvr_sweep.sh`)  │
   └─────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │ 4. Benchmark Evaluation & Telemetry                                              │
   │    • 4.1 Benchmark Evaluation (`make evaluate`)                                  │
   │    • 4.2 Sample Efficiency & Accuracy Comparison (`make compare-rlvr`)           │
   │    • 4.3 Weights & Biases Live Telemetry                                         │
   └─────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │ 5. Local Serving & Interactive UI (`streamlit run app/streamlit_app.py`)         │
   └──────────────────────────────────────────────────────────────────────────────────┘
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

# 3. Install AskTheDatabase in editable mode with development, training, and tracking dependencies (including wandb)
make install

# HPC Cluster Best Practice: Prevent user ~/.local site-packages from leaking into environment
export PYTHONNOUSERSITE=1

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

## 2. Supervised Fine-Tuning & Distillation

### 2.1 Conventional SFT Baseline Training (Condition 2)

Train the base model on ground-truth static text-to-SQL demonstration pairs using LoRA ($r=16, \alpha=32$):

```bash
# Standard Makefile invocation (Condition 2: Conventional Baseline):
make train-sft ARGS="model=qwen2_5_coder_1_5b training.dataset_path=data/processed/train.jsonl training.output_dir=models/mve/sft_baseline training.num_train_epochs=2 training.per_device_train_batch_size=4 training.gradient_accumulation_steps=2 training.learning_rate=2e-5"

# (Direct CLI equivalent for Slurm scripts):
# python3 scripts/train_sft.py model=qwen2_5_coder_1_5b training.dataset_path=data/processed/train.jsonl training.output_dir=models/mve/sft_baseline training.num_train_epochs=2 training.per_device_train_batch_size=4 training.gradient_accumulation_steps=2 training.learning_rate=2e-5
```
* **Resulting Checkpoint:** `models/mve/sft_baseline` (or custom `training.output_dir`)
* **Data Format:** `data/processed/train.jsonl` consists of single-turn demonstration pairs where the user prompt contains the question and full database schema DDL, and the assistant responds directly with the target SQL.

---

### 2.2 Harness-Zero Procedural Trajectory Distillation (Condition 4)

Fine-tune the student policy on on-policy multi-turn rollouts reviewed and repaired by the reference review harness $\mathcal{K}_{\text{SQL}}$ to internalize schema inspection and error recovery behaviors:

> [!TIP]
> **Why run dry runs first?**  
> Dry runs are fast smoke tests designed to catch configuration, environment, database path, and syntax errors without allocating long-running GPU compute:
> 1. `make collect-distillation ARGS="--dry-run"` verifies SQLite paths, reference rules, and JSONL export.
> 2. `make train-harness-zero ARGS="dry_run=true"` verifies tokenizer loading, LoRA configuration, and optimizer allocation.

#### Step 1: Collect Reviewed Trajectories
Collect balanced, difficulty-stratified teacher-reviewed trajectories (150 medium, 200 hard, 150 extra-hard):

```bash
# Standard Makefile invocation (Full 500-sample balanced stratification):
make collect-distillation ARGS="--raw-spider data/raw/spider_data/train_spider.json --databases-dir data/databases --n-medium 150 --n-hard 200 --n-extra 150 --output data/distillation/reviewed_trajectories.jsonl --metrics-output data/distillation/collection_metrics.json"

# Fast dry-run smoke test (5 sample episodes to verify pipeline health):
make collect-distillation ARGS="--dry-run"
```

* **Teacher Backend Options:**
  - **Deterministic Pure Rule Engine (Default — 100% Offline, Zero API Calls):**
    ```bash
    make collect-distillation ARGS="--teacher-backend rule"
    ```
  - **Optional: Google Gemini 2.0 Flash (External LLM Teacher via `GEMINI_API_KEY`):**
    ```bash
    make collect-distillation ARGS="--teacher-backend litellm --teacher-model gemini/gemini-2.0-flash"
    ```

#### Step 2: Fine-Tune on Reviewed Trajectories

```bash
# Standard Makefile invocation (Condition 4: Harness-Zero Distillation):
make train-harness-zero ARGS="model=qwen2_5_coder_1_5b training.output_dir=models/mve/harness_zero training.num_train_epochs=2 training.per_device_train_batch_size=2 training.gradient_accumulation_steps=4 training.learning_rate=2e-4 training.max_seq_length=2048"

# (Direct CLI equivalent for Slurm scripts):
# python3 scripts/train_sft.py model=qwen2_5_coder_1_5b training.dataset_path=data/distillation/reviewed_trajectories.jsonl training.output_dir=models/mve/harness_zero training.num_train_epochs=2 training.per_device_train_batch_size=2 training.gradient_accumulation_steps=4 training.learning_rate=2e-4 training.max_seq_length=2048
```
* **Resulting Checkpoint:** `models/mve/harness_zero`

> [!IMPORTANT]
> **Multi-Turn Trajectory Preservation (`load_sft_dataset`):**  
> `data/distillation/reviewed_trajectories.jsonl` stores the full 4–9 turn interactive conversation (`inspect_schema`, `execute_sql`, `final_answer`) inside `item["messages"]`. The training dataset loader [`src/ask_the_database/training/dataset.py`](../src/ask_the_database/training/dataset.py) preserves these multi-turn message sequences verbatim. This ensures the student policy learns active schema exploration and test query execution before emitting final SQL answers, bypassing the need for pre-computed schema DDL in the initial user prompt.

---

## 3. Multi-Turn Reinforcement Learning (RLVR / GRPO)

Train agent policies in `SkyRLSQLEnvironment` using Group Relative Policy Optimization (GRPO) with multi-turn verifiable rewards (test-suite correctness, execution accuracy, AST validity, and error recovery bonus).

### 3.1 Fast Dry-Run Smoke Test
Always verify environment initialization, rollout buffering, and reward computation before launching long training jobs:
```bash
make train-grpo ARGS="dry_run=true"
```

### 3.2 Single-Node Policy Training
Train the agent policy starting from either an SFT baseline or the Harness-Zero distilled checkpoint:

```bash
# Train starting from Harness-Zero checkpoint (Recommended for fast convergence):
make train-grpo ARGS="training.model_path=models/sft/harness_zero training.output_dir=models/rlvr/harness_zero_grpo training.max_steps=200"

# Train starting from Conventional SFT baseline:
make train-grpo ARGS="training.model_path=models/sft/checkpoint-final training.output_dir=models/rlvr/baseline_grpo training.max_steps=200"
```
* **Resulting Checkpoint:** `models/rlvr/.../training_state.json` and policy adapter weights.

### 3.3 Scaling to Cluster Infrastructure: Distributed Slurm Sweeps
To scale multi-turn GRPO across HPC GPU clusters (such as DESY Maxwell):
```bash
sbatch scripts/slurm/launch_rlvr_sweep.sh
```

---

## 4. Benchmark Evaluation & Telemetry

### 4.1 Benchmark Evaluation on Held-Out Splits
Evaluate checkpoints across standard Spider splits and robustness variants (`spider_dev`, `spider_realistic`). Note the evaluation protocol differences between interactive agent policies and conventional single-turn baselines:

#### A. Interactive Multi-Turn Evaluation (Harness-Zero & RLVR Policies)
Evaluated with `max_turns=5` to permit interactive schema inspection, row sampling, syntax verification, and error recovery:

```bash
# Standard Makefile invocation (Condition 4: Harness-Zero):
make evaluate ARGS="evaluation.model_path=models/mve/harness_zero 'evaluation.benchmarks=[spider_dev]' evaluation.max_turns=5 evaluation.output_dir=results/harness_zero_full_eval evaluation.summary_file=results/harness_zero_full_eval/summary.json"

# Standard Makefile invocation (Condition 5: RLVR Policy):
make evaluate ARGS="evaluation.model_path=models/rlvr/final 'evaluation.benchmarks=[spider_dev]' evaluation.max_turns=5 evaluation.output_dir=results/rlvr_eval evaluation.summary_file=results/rlvr_eval/summary.json"
```

#### B. Conventional Single-Turn Evaluation (SFT Baseline)
Evaluated with `max_turns=1`, matching standard single-turn text-to-SQL benchmark protocols:

```bash
# Standard Makefile invocation (Condition 2: Conventional SFT Baseline):
make evaluate ARGS="evaluation.model_path=models/mve/sft_baseline 'evaluation.benchmarks=[spider_dev]' evaluation.max_turns=1 evaluation.output_dir=results/baseline_full_eval evaluation.summary_file=results/baseline_full_eval/summary.json"
```

### 4.2 Compare RLVR Sample Efficiency & Asymptotic Accuracy
Quantify convergence speed, cumulative reward AUC, and held-out test-suite gains between two comparative training runs:

```bash
make compare-rlvr
```
*(Or specify explicit paths: `python3 scripts/compare_rlvr_efficiency.py --run-a-dir models/rlvr/baseline_grpo --run-b-dir models/rlvr/harness_zero_grpo`)*

* **Outputs Generated:**
  - `results/rlvr_sample_efficiency_report.md`
  - `results/rlvr_sample_efficiency_summary.json`

### 4.3 Weights & Biases (W&B) Dashboard Inspection
Weights & Biases telemetry is **enabled by default** across all Hydra training configurations (`sft.yaml`, `grpo_single_turn.yaml`, `rlvr_skyrl_agent.yaml`). Monitor live telemetry under project `ask-the-database`:

> [!TIP]
> **Offline Toggle:** If running on an offline compute node or without W&B credentials, disable tracking by passing `training.report_to=none` or `training.wandb.enabled=false`.

1. **`mean_reward` vs. `step` (Convergence Speed Curve)**:
   - Tracks the progression of mean rollout reward. Harness-Zero initialized runs cross positive reward ($R > 0$) early (bypassing the zero-variance cold-start plateau).
2. **`advantage_mean` vs. `step` (Gradient Learning Signal)**:
   - Confirms non-zero, healthy GRPO gradient updates without advantage collapse.
3. **`min_reward` & `max_reward` vs. `step` (Rollout Envelope)**:
   - Highlights group diversity among the $G=4$ parallel rollouts per prompt.

---

## 5. Interactive Querying & Local Serving

Launch the Streamlit web application to inspect database schemas, ask natural language questions, and trace multi-turn query execution in real time:

```bash
streamlit run app/streamlit_app.py
```

---

## 6. Hugging Face Release & Card Generation

The repository provides automated tooling to generate and publish Model Cards (`MODEL_README.md`) and Dataset Cards (`DATASET_README.md`) incorporating Spider benchmark metrics and Zero-Budget MVE decision gate results.

### 6.1 Local Offline Inspection (`make generate-hf-cards`)
```bash
make generate-hf-cards
```
- **Invokes:** `python3 scripts/release_to_hf.py --card-only`
- **Output:** Writes markdown cards locally to `results/cards/DATASET_README.md` and `results/cards/MODEL_README.md`.
- **Purpose:** Allows offline review and verification of metadata, benchmark tables, and tool usage examples before publishing to remote repositories.

### 6.2 Remote Hub Publication (`make publish-hf-cards`)
Deploy the generated cards directly to Hugging Face Hub:
- **Model Card:** [https://huggingface.co/chrisjcc/ask-the-database](https://huggingface.co/chrisjcc/ask-the-database)
- **Dataset Card:** [https://huggingface.co/datasets/chrisjcc/ask-the-database](https://huggingface.co/datasets/chrisjcc/ask-the-database)

```bash
# Publish via make with target repositories
make publish-hf-cards ARGS="--model-repo chrisjcc/ask-the-database --dataset-repo chrisjcc/ask-the-database"

# Or directly via Python CLI
python3 scripts/release_to_hf.py --publish-cards \
  --model-repo chrisjcc/ask-the-database \
  --dataset-repo chrisjcc/ask-the-database
```
*Note: Requires `HF_TOKEN` with write permissions in your environment or `.env` file.*

For full details and CLI flag references, see [`docs/huggingface_card_generation_guide.md`](huggingface_card_generation_guide.md).

---

## 7. Summary of Operational Makefile Commands

| Command | Action | Output / Checkpoint |
| :--- | :--- | :--- |
| `make install` | Installs dependencies in editable mode | Virtualenv configured |
| `make lint` | Runs Ruff linter and strict Mypy type-check | Zero warnings |
| `make test` | Executes full Pytest unit test suite | 171/171 tests passing |
| `make smoke-eval` | Runs fast in-process zero-budget smoke check (< 3.0s) | Sub-second regression check |
| `make evaluate-zero-budget-mve` | Adjudicates 4-gate Zero-Budget MVE decision gates | `results/zero_budget_mve_summary.json` |
| `make setup-hooks` | Installs native local git pre-commit hook (< 0.2s) | `.git/hooks/pre-commit` |
| `make generate-hf-cards` | Generates local Hugging Face README cards | `results/cards/*.md` |
| `make publish-hf-cards` | Publishes README cards to Hugging Face Hub | Remote Hub repositories |
| `make preprocess` | Downloads and tokenizes Spider datasets | `data/processed/train.jsonl` |
| `make train-sft` | Trains Conventional SFT Baseline | `models/sft/checkpoint-final` |
| `make collect-distillation` | Collects K_SQL reviewed trajectories | `data/distillation/reviewed_trajectories.jsonl` |
| `make train-harness-zero` | Trains SFT on Harness-Zero trajectories | `models/sft/harness_zero` |
| `make train-grpo` | Runs multi-turn GRPO / RLVR training | `models/rlvr/final` |
| `make evaluate` | Evaluates models on Spider test suites | `results/evaluation_summary.json` |
| `make compare-rlvr` | Compares RLVR sample efficiency & accuracy | `results/rlvr_sample_efficiency_report.md` |
| `make evaluate-mve` | Adjudicates Decision Gates (`GO`/`NO-GO`) | `docs/research/harness-zero/mve_final_report.md` |

