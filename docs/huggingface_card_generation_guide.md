# Hugging Face Model & Dataset Card Generation Guide

This guide documents the Hugging Face Model Card and Dataset Card generators implemented in [`scripts/release_to_hf.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/scripts/release_to_hf.py). It covers operational CLI commands, metadata specifications, benchmark telemetry integration, and automated hub deployment.

---

## 1. Architectural Role & Overview

When publishing fine-tuned models and datasets to the Hugging Face Hub, high-quality documentation cards (`README.md`) are essential for discoverability, reproducibility, and safety. 

[`scripts/release_to_hf.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/scripts/release_to_hf.py) provides a unified CLI and Python API that:
1. **Generates Native Dataset Cards (`DATASET_README.md`)**: Fully structured metadata and usage examples for the multi-turn trajectories and Spider benchmark splits.
2. **Generates Native Model Cards (`MODEL_README.md`)**: Documents model architecture, interactive tool capabilities, empirical benchmark accuracy, and pre-registered MVE decision gate verdicts.
3. **Supports Offline / Zero-Credential Generation (`--card-only`)**: Generates and formats markdown cards locally without requiring network access, Hugging Face tokens, or W&B API keys.
4. **Automates Hub Deployment (`--deploy`)**: Publishes datasets and verified model weights directly to Hugging Face Hub when `HF_TOKEN` is supplied.

---

## 2. Quickstart Operational Commands

### 2.1 One-Click Operations via Make: Local Inspection vs. Hub Publication

The repository provides two distinct operations: **Local Offline Generation** (for inspection without credentials) and **Remote Hub Publication** (for deploying live to Hugging Face).

#### A. Local Offline Card Generation (`make generate-hf-cards`)

```bash
make generate-hf-cards
```

Under the hood, this invokes:
```bash
python3 scripts/release_to_hf.py --card-only
```

- **Execution Mode:** The `--card-only` flag runs in an entirely local, offline generation mode. No network calls or API credentials are required.
- **Artifacts Written to Disk:**
  - `results/cards/DATASET_README.md`
  - `results/cards/MODEL_README.md`
- **Architectural Rationale:** This design allows developers and researchers to inspect, review, and diff the generated cards before publishing, preventing accidental overwrites or deployment failures when credentials are not configured.

---

#### B. Publish Cards Directly to Hugging Face Repositories (`make publish-hf-cards`)

To publish the generated documentation cards directly to your public or private Hugging Face repositories:
- **Model Card Destination:** [https://huggingface.co/chrisjcc/ask-the-database](https://huggingface.co/chrisjcc/ask-the-database)
- **Dataset Card Destination:** [https://huggingface.co/datasets/chrisjcc/ask-the-database](https://huggingface.co/datasets/chrisjcc/ask-the-database)

**Run the publication command via `make`:**
```bash
make publish-hf-cards ARGS="--model-repo chrisjcc/ask-the-database --dataset-repo chrisjcc/ask-the-database"
```

**Or directly via Python:**
```bash
python3 scripts/release_to_hf.py --publish-cards \
  --model-repo chrisjcc/ask-the-database \
  --dataset-repo chrisjcc/ask-the-database
```

**What the publication script executes automatically:**
1. **Authenticates:** Connects to the Hugging Face Hub using your `HF_TOKEN` environment variable.
2. **Auto-Creates Repositories:** Calls `api.create_repo(..., exist_ok=True)` to create the model and dataset repositories if they do not already exist on your Hugging Face account.
3. **Ingests Live Telemetry:** Automatically ingests Spider Dev benchmark metrics from `results/evaluation_summary.json` and verified decision gates from the Zero-Budget MVE.
4. **Deploys Documentation Cards:** Uploads `README.md` to both [`chrisjcc/ask-the-database`](https://huggingface.co/chrisjcc/ask-the-database) and [`datasets/chrisjcc/ask-the-database`](https://huggingface.co/datasets/chrisjcc/ask-the-database).

---

### 2.2 CLI Usage Options

You can invoke [`scripts/release_to_hf.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/scripts/release_to_hf.py) directly with custom parameters:

```bash
# Generate both cards for the default Harness-Zero model
python scripts/release_to_hf.py --card-only

# Generate only the Model Card for a specific model variant
python scripts/release_to_hf.py --card-only --card-type model --model-variant harness_zero

# Generate Model Card for Multi-Turn RLVR Agent
python scripts/release_to_hf.py --card-only --card-type model --model-variant rlvr_skyrl

# Generate Model Card for SFT Baseline with a specific release tag
python scripts/release_to_hf.py --card-only --card-type model \
  --model-variant sft_baseline \
  --release-tag v1.0.0

# Generate Dataset Card for a custom Hugging Face repo
python scripts/release_to_hf.py --card-only --card-type dataset \
  --dataset-repo your-username/ask-the-database-trajectories

# Output cards to a specific directory (e.g. Hugging Face Space repository)
python scripts/release_to_hf.py --card-only --output-dir hf_space/
```

---

### 2.3 CLI Flags Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--card-only` | Flag | `False` | Generates cards locally on disk without uploading to Hugging Face. |
| `--card-type` | Choice | `all` | Card to generate: `model`, `dataset`, or `all`. |
| `--output-dir` | Path | `results/cards` | Directory where markdown files (`MODEL_README.md`, `DATASET_README.md`) are saved. |
| `--model-variant` | Choice | `harness_zero` | Target model variant: `base`, `sft_baseline`, `filtered_rollouts`, `harness_zero`, `rlvr_skyrl`. |
| `--base-model` | String | `Qwen/Qwen2.5-Coder-7B-Instruct` | Base foundation model identifier. |
| `--model-repo` | String | `chrisjcc/ask-the-database-coder-7b` | Target Hugging Face model repository ID. |
| `--dataset-repo` | String | `chrisjcc/ask-the-database-trajectories` | Target Hugging Face dataset repository ID. |
| `--release-tag` | String | `""` | Optional release tag (e.g., `v1.0.0`). |
| `--summary-json` | Path | `results/evaluation_summary.json` | Path to benchmark evaluation summary JSON to populate live metrics. |
| `--deploy` | Flag | `False` | Uploads generated cards and weights to Hugging Face Hub (requires `HF_TOKEN`). |

---

## 3. What is Included in Generated Cards

### 3.1 Model Card Generator (`generate_model_card`)

The generated Model Card (`MODEL_README.md`) provides a complete, production-grade documentation package compliant with Hugging Face Hub standards:

1. **YAML Frontmatter Metadata**:
   - `language: [en]`
   - `license: apache-2.0`
   - `library_name: transformers`
   - `pipeline_tag: text-generation`
   - `base_model: Qwen/Qwen2.5-Coder-7B-Instruct`
   - `datasets: [chrisjcc/ask-the-database-trajectories]`
   - `tags: [text-to-sql, reinforcement-learning, rlvr, harness-zero, spider, verifiable-rewards, sqlite, zero-budget-eval, qwen2.5-coder]`

2. **Model Architecture & Paradigm**:
   - Specifies the exact training paradigm (e.g. Multi-Turn Interactive Exploration vs. Single-Turn Greedy Baseline).
   - Details base model, fine-tuning condition, and dataset provenance.

3. **Interactive Tools & Procedural Primitives**:
   - `\schema [table_name]`: Inspects DDL schemas, column types, primary keys, and foreign keys.
   - `\sample [table_name] [column_name]`: Samples distinct categorical values to eliminate literal hallucinations.
   - `\sql [query]`: Dry-runs trial SQL queries against an in-process SQLite sandbox to catch execution errors.

4. **Empirical Benchmark & Decision Gate Telemetry**:
   - **Spider Dev Benchmark**: Ingests `results/evaluation_summary.json` to render live Execution Accuracy (EX), Test Suite Accuracy (TS), and Valid Execution Rate (VA).
   - **Zero-Budget MVE Gates**: Embeds verified pre-registered gate adjudications:
     - **Gate 1 (Fast Execution Latency Budget):** $0.1268\text{ s}$ ($126.80\text{ ms}$) across 5 passes ($\le 3.0\text{ s}$ budget).
     - **Gate 2 (Benchmark Rank Correlation):** Spearman $\rho = 1.0000$ ($\rho \ge 0.88$ threshold).
     - **Gate 3 (Diagnostic Resolution):** $100.0\%$ Attribution, $100.0\%$ Hallucination Reduction ($10 \to 0$ misses).
     - **Gate 4 (Mutation Sensitivity):** $100.0\%$ Bug Detection Rate, $0.0\%$ Escape Rate on dual fixtures.
     - **Official Verdict:** `MVE ADJUDICATION DECISION: GO`.

5. **Quickstart Code Examples**:
   - High-level pipeline execution via `ask_the_database.inference.pipeline.TextToSQLPipeline`.
   - Direct PyTorch / Hugging Face Transformers snippet with `AutoModelForCausalLM` and ChatML prompt formatting.

6. **Safety Sandbox & Guardrails**:
   - Read-only SQLite sandbox isolation.
   - Execution timeout protection ($< 2.0\text{ s}$).
   - Query plan inspection (`EXPLAIN QUERY PLAN`) blocking Cartesian fanouts (`SCAN TABLE ... WITHOUT INDEX`).

---

### 3.2 Dataset Card Generator (`generate_dataset_card`)

The generated Dataset Card (`DATASET_README.md`) structures the multi-turn training data and benchmark evaluation sets:

1. **YAML Frontmatter Configuration**:
   - Configures distinct subsets:
     - `spider`: Standard Spider SQL benchmark (`train`: 7,000 samples, `validation`: 1,034 samples).
     - `reviewed_trajectories`: 500+ curated multi-turn trajectories with procedural exploration thoughts and tool calls reviewed by $K_{\text{SQL}}$ rules.
     - `filtered_rollouts`: Rollouts from student models passing verifiable test-suite execution rewards.

2. **Schema & Feature Type Definitions**:
   - `db_id`: Unique SQLite database identifier.
   - `question`: User natural language prompt.
   - `gold_sql`: Ground-truth SQL query.
   - `trajectory`: Structured sequence of steps containing:
     - `step`: Turn index (int).
     - `thought`: Internal agent reasoning / hypothesis.
     - `tool`: Tool called (`inspect_schema`, `sample_data`, `execute_sql`).
     - `argument`: Invocation payload.
     - `observation`: Execution result or environment feedback.
   - `final_sql`: Final candidate query submitted by the agent.
   - `test_suite_passed`: Boolean flag indicating whether the query passed dual-fixture mutation testing.

3. **Loading Examples**:
   - Python code snippets using Hugging Face `datasets.load_dataset`.

4. **Licensing & Attribution**:
   - Spider benchmark (CC BY-SA 4.0).
   - Trajectories and fixtures (Apache 2.0).

---

## 4. Full Deployment Pipeline (`--deploy`)

To publish directly to Hugging Face Hub from a deployment machine or CI runner:

```bash
# 1. Export your Hugging Face write token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 2. Execute deployment
python scripts/release_to_hf.py --deploy \
  --model-variant harness_zero \
  --model-repo chrisjcc/ask-the-database-coder-7b \
  --dataset-repo chrisjcc/ask-the-database-trajectories
```

The script will:
1. Initialize the Hugging Face API client (`HfApi`).
2. Generate and upload `README.md` to the dataset repository.
3. Generate and upload `README.md` to the model repository.
4. Verify repository existence and upload status.

---

## 5. Automated Testing & Verification

The card generation pipeline is fully covered by unit tests in [`tests/unit/test_release_to_hf.py`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/tests/unit/test_release_to_hf.py):

```bash
# Run unit tests
pytest tests/unit/test_release_to_hf.py -v

# Run linter and type-checker
ruff check scripts/release_to_hf.py
mypy scripts/release_to_hf.py
```
