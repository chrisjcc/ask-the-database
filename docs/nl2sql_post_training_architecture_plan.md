# Natural-Language-to-SQL Post-Training Architecture & Implementation Blueprint
## Adapting AskBeforeAnswer Principles to Reinforcement Learning with Verifiable Rewards (RLVR) and Multi-Turn Database Agents

---

## 1. Executive Summary & Vision

### 1.1 The Shift: From Semantic Preferences to Verifiable Program Execution
The **AskBeforeAnswer** project demonstrated that aligning an LLM to distinguish ambiguous from answerable questions requires strict schema formatting, contrastive preference modeling (SFT, DPO, ORPO), and multi-turn agentic simulation. However, in open-domain natural language question answering, reward signals fundamentally rely on semantic heuristics or LLM-as-a-judge scorers, which introduce softness, gradient noise, and reward hacking risks.

Natural Language-to-SQL (NL2SQL) translation provides an ideal environment for **Reinforcement Learning with Verifiable Rewards (RLVR)**. Because SQL is an executable formal language, a generated query can be executed directly against an isolated database engine to verify output correctness. 

However, verifiable execution in SQL is fraught with subtle failure modes:
1. **Execution Success $\neq$ Semantic Correctness**: A query can execute without errors (`SELECT * FROM table`) yet fail completely to answer the user's intent.
2. **Syntax Multiplicity & Syntactic Divergence**: Multiple syntactically distinct SQL queries (e.g., `JOIN` vs. subquery, `GROUP BY ... HAVING` vs. `WHERE`) produce mathematically identical relation tables.
3. **Spurious Execution Equivalence ("Right for the Wrong Reason")**: On small database states, a flawed query can accidentally return the reference table (e.g., matching on empty sets `NULL`, or small tables where filters happen to coincide).
4. **Destructive & Unbounded Compute Risks**: Unsanitized model rollouts can trigger Cartesian products, infinite loops, resource exhaustion, or state mutations (`DROP`, `DELETE`, `UPDATE`).

### 1.2 Core Thesis: Multi-Turn Agentic Database Exploration
Recent state-of-the-art results from **SkyRL-SQL** (*"SKYRL-SQL: Multi-turn SQL Data Agents via RL"*, NovaSky AI, 2025) demonstrate that multi-turn agentic interaction enables language models to mirror human data analysts: probing schemas, executing sample queries, inspecting intermediate errors, and refining the final query iteratively. 

Using only **~650 curated samples** (from `SynSQL-2.5M` and `Spider`), a 7B parameter model trained with multi-turn RLVR can outperform models orders of magnitude larger (including GPT-4o and o4-mini) on rigorous benchmarks like Spider, Spider-DK, Spider-Syn, and Spider-Realistic.

### 1.3 Key Architectural Goal: Scientific Durability
The most durable assets of this research project must be:
- The **SQL Execution Environment & Sandbox**
- The **Verifiable Multi-Tier Reward Design**
- The **Evaluation & Multi-Turn Benchmark Protocol**
- The **Single Source of Truth (SSoT) Configuration Architecture**

These components must remain **100% framework-independent Python modules**. Whether rollouts are trained using **SkyRL / SkyRL-Agent**, **veRL**, **NeMo RL**, **OpenRLHF**, or **TRL**, the scientific and verifiable reward core must remain immutable.

---

## 2. Single Source of Truth (SSoT) & Infrastructure Separation

A central lesson learned from AskBeforeAnswer is that configuration drift and dual-parameter declarations across the CLI, Hydra, Make, DVC, and W&B create brittle pipelines, silent bugs, and broken sweep integrations. 

Below is the definitive matrix establishing strict separation of concerns and a single source of truth across all tools.

```
+---------------------------------------------------------------------------------------+
|                                    USER / CLI / CI / SLURM                            |
+-------------------------------------------+-------------------------------------------+
                                            |
                                   [1] make <target>
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                        MAKEFILE                                       |
|  - Role: Workflow entrypoint & command interface.                                     |
|  - Owns: PHONY aliases, OS/hardware sanity checks, lint/test drivers.                |
|  - Strictly forbids: Inline hyperparameter overrides or script logic.                 |
+-------------------------------------------+-------------------------------------------+
                                            |
                                  [2] dvc repro <stage>
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                       DVC (DAG)                                       |
|  - Role: Deterministic data pipeline, stage dependencies, and artifact tracking.      |
|  - Owns: Stage DAG (data -> sft -> rlvr -> eval), cache integrity, data hashes.      |
|  - Reads: params.yaml (mirrored from Hydra defaults).                                 |
+-------------------------------------------+-------------------------------------------+
                                            |
                                  [3] python scripts/<stage>.py
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                HYDRA CONFIGS (configs/)                               |
|  - Role: Pure hierarchical configuration and parameter definition.                    |
|  - Owns: Default hyperparameters, model architectures, environment constraints.      |
|  - Strictly forbids: Execution code or bash logic.                                    |
+-------------------------------------------+-------------------------------------------+
                                            |
                                  [4] Ingests config object
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                             PYTHON CORE (src/ask_before_sql/)                          |
|  - Role: Framework-agnostic scientific implementation.                                |
|  - Owns: SQLEnvironment, RewardRubric, DB Sandbox, MultiTurnAgent, Evaluator.         |
|  - Strictly forbids: Hardcoded paths, hardcoded hyperparams, framework coupling.      |
+---------------------------------------------------------------------------------------+
```

### 2.1 Tool Responsibility Matrix

| Layer / Tool | Single Source of Truth For | What It Must NEVER Do |
| :--- | :--- | :--- |
| **`pyproject.toml`** | Project metadata, build system, tool configurations (`ruff`, `mypy`, `pytest`), dependency specifications (abstract). | Pin cluster-specific CUDA binary wheels or host system library paths. |
| **`requirements.txt`** | Exact, frozen, reproducible package pins (`==`), wheel URLs for CUDA/Torch acceleration stacks (`vllm`, `flash-attn`, `unsloth`). | Declare build metadata or package discovery settings. |
| **`Makefile`** | Command entrypoints, target aliases (`make train-rlvr`), environment variable guards (`PYTHONNOUSERSITE=1`), CI shortcuts. | Hardcode hyperparameters or pass ad-hoc CLI flags directly to Python scripts. |
| **`configs/` (Hydra)** | **ALL hyperparameters**, model architectures, prompt templates, database connection limits, sweep search spaces. | Execute pipeline stages or trigger subprocesses. |
| **`dvc.yaml`** | Stage dependency DAG, data artifact hashing, output model checkpoint tracking, reproducibility lock (`dvc.lock`). | Re-declare default parameter values (must bind to `params.yaml` or Hydra). |
| **W&B / Weave** | Run tracking, experiment comparison, Bayesian optimization states (Sweeps), multi-turn interactive trajectory logging. | Act as the primary storage for checkpoints or dictate dataset versions. |
| **Slurm (`slurm/`)** | Cluster queue parameters (`#SBATCH`), hardware allocations (GPUs, RAM), array throttling (`%1`), environment isolation. | Contain custom training code, unversioned scripts, or bypass the `Makefile`/DVC. |

### 2.2 Prevention of Anti-Patterns (Lessons from AskBeforeAnswer)

1. **No Redundant CLI Parameter Overrides**: In AskBeforeAnswer, training scripts occasionally accepted both argparse flags and Hydra overrides. In this project, all scripts strictly use Hydra (`@hydra.main(config_path="configs", config_name="config")`).
2. **Safe Cluster Sweeps**: Never trigger `make sweep` inside a Slurm job. The sweep is initialized **once** on the login node (`wandb sweep sweeps/rlvr.yaml`), producing a `SWEEP_ID`. The Slurm array script executes `wandb agent $SWEEP_ID --count $COUNT` to prevent launching fragmented, unlinked sweeps.
3. **Environment Parity**: Cluster compute nodes must strictly export `PYTHONNOUSERSITE=1` and dynamically set `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` to eliminate cluster CXXABI/GLIBC mismatch errors.
4. **Framework Independence**: The RL engine (`SkyRL-Agent`, `veRL`, etc.) is isolated in `src/ask_before_sql/training/adapters/`. If the chosen RL framework is deprecated or superseded, the core database environment and reward evaluators remain completely untouched.

---

## 3. Standardized Repository Architecture

```text
ask-before-sql/
├── .env.example                      # Secrets template (WANDB_API_KEY, HF_TOKEN, DB_CREDENTIALS)
├── .gitignore                        # Git exclusion rules
├── .dvcignore                        # DVC tracking exclusion rules
├── dvc.yaml                          # Canonical pipeline DAG definition
├── params.yaml                       # Root parameter mirror for DVC pipeline tracking
├── Makefile                          # Unified interface for local and cluster execution
├── pyproject.toml                    # Build system & linter (ruff, mypy, pytest) definitions
├── requirements.txt                  # Locked production and training dependencies
├── Dockerfile                        # Containerized environment for local/HF deployment
│
├── configs/                          # Hydra Configuration Tree (Single Source of Truth)
│   ├── config.yaml                   # Global defaults composition
│   ├── data/                         # Data ingestion and preprocessing parameters
│   │   ├── skyrl_sql_653.yaml        # SynSQL + Spider 653-sample training split
│   │   ├── spider.yaml               # Standard Spider evaluation configuration
│   │   └── spider_robust.yaml        # Spider-DK, Spider-Syn, Spider-Realistic configs
│   ├── environment/                  # Database sandbox and execution bounds
│   │   ├── sqlite_sandbox.yaml       # In-memory / ephemeral SQLite configuration
│   │   └── postgres_sandbox.yaml     # Containerized read-only PostgreSQL configuration
│   ├── model/                        # Backbone LLM architecture & LoRA configs
│   │   ├── qwen2_5_coder_7b.yaml     # Primary backbone: Qwen2.5-Coder-7B-Instruct
│   │   └── deepseek_coder_7b.yaml    # Baseline backbone: DeepSeek-Coder-V2 / 7B
│   ├── training/                     # Training algorithms & hyperparameters
│   │   ├── sft.yaml                  # Supervised Fine-Tuning warm-up
│   │   ├── rlvr_grpo.yaml            # GRPO with Verifiable Rewards
│   │   └── rlvr_skyrl_agent.yaml     # Multi-turn SkyRL-Agent interactive RL
│   ├── evaluation/                   # Evaluation suites and test-set execution
│   │   ├── execution_eval.yaml       # Test-Suite (TS) and Execution Accuracy (EX)
│   │   └── multi_turn_eval.yaml      # Multi-turn probe efficiency & recovery metrics
│   └── sweeps/                       # W&B Bayesian Sweep specifications
│       ├── sft_sweep.yaml            # SFT learning rate & warmup sweep
│       └── rlvr_sweep.yaml           # RLVR beta, discount factor, and reward weight sweep
│
├── data/                             # DVC-managed local data storage (git-ignored)
│   ├── raw/                          # Raw benchmark downloads (Spider, SynSQL)
│   ├── databases/                    # SQLite .sqlite database files for evaluation
│   ├── processed/                    # Cleaned JSONL splits for SFT and RLVR
│   └── test_suites/                  # Test-suite generated databases for edge-case eval
│
├── models/                           # DVC-managed model checkpoints (git-ignored)
│   ├── sft/final/                    # SFT warmed-up baseline
│   ├── rlvr/final/                   # RLVR aligned policy
│   └── baselines/                    # Reference baseline weights
│
├── src/ask_before_sql/               # Core Scientific Library (Framework-Agnostic)
│   ├── __init__.py
│   ├── environment/                  # Database Execution Sandboxes
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract BaseSQLEnvironment interface
│   │   ├── sqlite_env.py             # SQLite ephemeral execution engine
│   │   ├── schema_manager.py         # Dynamic schema inspection, table/column DDL parser
│   │   └── test_suite_evaluator.py   # Multiple database state evaluator (test-suite generation)
│   ├── rewards/                      # Verifiable Reward Functions & Rubrics
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract RewardFunction & DiagnosticPayload
│   │   ├── execution_reward.py       # Exact execution equivalence (EX / VES)
│   │   ├── syntax_reward.py          # SQL parsing, AST validation, and linting
│   │   ├── schema_linking_reward.py  # Precision/recall of referenced tables & columns
│   │   ├── multi_turn_reward.py      # Turn-efficiency penalties and recovery bonuses
│   │   └── rubric.py                 # Composite weighted rubric with strict bounds
│   ├── agent/                        # Multi-Turn Agentic Interaction Loop
│   │   ├── __init__.py
│   │   ├── interaction_state.py      # Conversation history, intermediate SQL, execution traces
│   │   ├── tool_registry.py          # DB probing tools (schema inspect, sample 3 rows, run SQL)
│   │   └── prompt_formatter.py       # Chat templates (Qwen/DeepSeek) for multi-turn SQL
│   ├── evaluation/                   # Systematic Evaluation Harness
│   │   ├── __init__.py
│   │   ├── metrics.py                # EX (Execution Accuracy), TS (Test Suite), VA (Valid Rate)
│   │   ├── hardness_categorizer.py   # Query complexity categorization (Easy, Medium, Hard, Extra)
│   │   └── benchmark_runner.py       # Full benchmark execution against Spider / Robust variants
│   ├── inference/                    # Serving & Demonstration Pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py               # End-to-end Text-to-SQL generation & verification pipeline
│   │   └── fallback_handler.py       # Rule-based fallback on terminal execution errors
│   └── training/                     # Framework Adapters (Isolating RL Frameworks)
│       ├── __init__.py
│       ├── adapters/                 # Framework adapters bridging to src/ask_before_sql/rewards
│       │   ├── skyrl_adapter.py      # SkyRL-Train & SkyRL-Agent environment bridge
│       │   ├── verl_adapter.py       # veRL Ray-based actor/critic environment bridge
│       │   └── trl_grpo_adapter.py   # Hugging Face TRL GRPOTrainer reward callback bridge
│       └── trainer.py                # Main orchestrator executing the selected adapter
│
├── scripts/                          # Deterministic CLI Entrypoints (invoked by DVC & Makefile)
│   ├── preprocess_data.py            # Converts raw benchmarks into SFT & RLVR formats
│   ├── train_sft.py                  # Executes SFT warmup stage
│   ├── train_rlvr.py                 # Executes RLVR training stage via adapter
│   ├── run_sweep_trial.py            # W&B sweep trial runner calling `dvc exp run`
│   ├── evaluate_benchmark.py         # Runs single-turn and multi-turn evaluation suites
│   └── generate_report.py            # Synthesizes evaluation summaries into Markdown reports
│
├── slurm/                            # Hardened HPC Slurm Scripts
│   ├── README.md                     # Slurm cluster operations manual
│   ├── logs/                         # Slurm stdout/stderr runtime logs
│   ├── 01_preprocess.slurm           # CPU-only data preparation job
│   ├── 02_sft_warmup.slurm           # GPU job for supervised fine-tuning
│   ├── 03_rlvr_train.slurm           # GPU job for full RLVR policy optimization
│   ├── 04_sweep_array.slurm          # Sequential (%1) throttled Bayesian sweep array
│   └── 05_evaluate.slurm             # Multi-benchmark evaluation suite
│
├── tests/                            # Comprehensive Test Suite
│   ├── unit/                         # Unit tests for rewards, parsing, and sandbox isolation
│   ├── integration/                  # End-to-end integration tests for multi-turn rollout
│   └── test_portability.py           # Portability test running 50 samples across frameworks
│
└── docs/                             # Comprehensive 9-Phase Engineering Documentation
    ├── 01_problem_definition.md
    ├── 02_dataset_curation.md
    ├── 03_reward_design_and_verification.md
    ├── 04_agentic_multiturn_loop.md
    ├── 05_training_and_adapters.md
    ├── 06_evaluation_and_benchmarks.md
    ├── 07_hyperparameter_sweeps.md
    ├── 08_cluster_slurm_operations.md
    └── 09_production_deployment.md
```

---

## 4. The Framework-Independent Core: Environment & Verifiable Rewards

The central architectural innovation of this system is that the **Database Execution Environment** and the **Reward Evaluator** are completely decoupled from the training framework. 

If SkyRL is updated, or if the project transitions to veRL or NeMo RL, **zero lines of code** in `src/ask_before_sql/rewards/` or `src/ask_before_sql/environment/` need to change.

```
+---------------------------------------------------------------------------------------+
|                                    RL FRAMEWORK                                       |
|                  (SkyRL-Agent / veRL / OpenRLHF / HuggingFace TRL)                    |
+-------------------------------------------+-------------------------------------------+
                                            |
                      Passes query candidate + rollout trajectory
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                   FRAMEWORK ADAPTER (src/.../training/adapters/)                      |
|                  Maps framework tensor/string types to Python API                     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|               CORE REWARD RUBRIC (src/.../rewards/rubric.py)                           |
|                                                                                       |
|  [Tier 1: AST & Syntax]        -> sqlglot parsing, DDL compliance                     |
|  [Tier 2: Execution Safety]    -> Read-only checks, timeout limits, AST mutation gate  |
|  [Tier 3: DB State Result]     -> SQLite / DuckDB isolated run on target DB           |
|  [Tier 4: Test-Suite (TS) Ex.] -> Run across 5+ mutated DB states (prevents false +)  |
|  [Tier 5: Multi-Turn Penalty]  -> Step budget decay & error recovery bonus            |
+-------------------------------------------+-------------------------------------------+
                                            |
                      Interacts with Sandboxed DB Engine
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                SANDBOXED ENVIRONMENT (src/.../environment/sqlite_env.py)              |
|                   - In-memory / Ephemeral read-only database copies                   |
|                   - Hard walltime cutoff (e.g., 3.0s per query)                       |
|                   - Memory bounded (< 512MB per process)                              |
+---------------------------------------------------------------------------------------+
```

### 4.1 Sandboxed Database Execution (`src/ask_before_sql/environment/`)

To prevent model rollouts from corrupting databases, executing denial-of-service queries, or exhausting memory, all database interactions occur within an isolated sandbox.

#### Operational Safety Constraints:
1. **Strict Read-Only Enforcement**: SQLite connections are opened using URI mode: `file:db.sqlite?mode=ro`. Additionally, before execution, the AST is inspected via `sqlglot` to verify that the root operation is strictly `SELECT` or `WITH ... SELECT`. Any statement containing `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, or `ATTACH` is immediately intercepted and awarded maximum negative reward.
2. **Walltime Cutoff**: Every query execution is guarded by a hard timeout (default: `3.0 seconds`). Queries exceeding this limit (e.g., accidental unindexed cross-joins on 100k rows) are killed via Python's `signal.SIGALRM` or thread timer and penalized.
3. **Result Memory Limit**: Results are fetched using `cursor.fetchmany(1000)`. If a query attempts to dump an entire table exceeding buffer limits, execution terminates with an overflow flag.

### 4.2 Multi-Tier Verifiable Reward Rubric (`src/ask_before_sql/rewards/`)

A fundamental lesson of SQL RLVR is that **binary execution accuracy is too sparse for early training, while naive reward shaping causes reward hacking**. 

We establish a **Strict Hierarchical Multi-Tier Rubric** where partial-credit validity signals can **never** outweigh or compromise actual execution correctness.

$$\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{correctness}} + \mathcal{R}_{\text{validity}} + \mathcal{R}_{\text{efficiency}}$$

#### Tier 1: Syntax & Parse Validity ($\mathcal{R}_{\text{syntax}} \in [-1.0, 0.0]$)
- Evaluates whether the generated string contains a valid, well-formed SQL query.
- Uses `sqlglot` with dialect-specific transpilations (SQLite / Postgres).
- **Malformed SQL / Missing Markdown code block**: $-1.0$ penalty.
- **Valid SQL Syntax**: $0.0$ (neutral).

#### Tier 2: Execution Feasibility & Schema Linking ($\mathcal{R}_{\text{schema}} \in [-0.5, 0.2]$)
- Evaluates whether the tables and columns referenced in the query actually exist in the schema.
- **Hallucinated table/column causing runtime exception (`no such table`, `no such column`)**: $-0.5$ penalty + error diagnostic message returned to the agent.
- **Valid Schema Reference (query executes without database exceptions)**: $+0.2$ bonus.

#### Tier 3: Execution Equivalence (Primary Signal: $\mathcal{R}_{\text{exec}} \in [0.0, 1.0]$)
- Evaluates whether the generated SQL query $Q_{\text{gen}}$ produces the exact same relation (table) as the gold standard query $Q_{\text{gold}}$ on the target database state $\mathcal{D}_0$.
- **Set vs. Bag Semantics**: Unless an explicit `ORDER BY` is required by the prompt, comparisons are order-agnostic (multiset / tuple set equivalence).
- **Float Tolerance**: Numeric values are compared with $\epsilon = 10^{-4}$ tolerance.
- **Null Handling**: Distinguishes between empty sets returned because of a mismatch vs. genuine empty answers.
- **Mismatch**: $0.0$ reward.
- **Execution Equivalence ($Q_{\text{gen}}(\mathcal{D}_0) \equiv Q_{\text{gold}}(\mathcal{D}_0)$)**: $+1.0$ reward.

#### Tier 4: Test-Suite (TS) Verification (Eliminating Spurious Success)
- *The Problem*: In benchmarks like Spider, many database tables contain only 5–20 rows. A query that selects the wrong column or has a flawed filter can accidentally return the same single row as the gold query.
- *The Solution*: As proven by Zhong et al. (*Test-Suite Equivalence for Text-to-SQL*), we evaluate candidate queries against **multiple mutated database states** $\{\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_k\}$ (generated via database sampling and constraint inversion).
- A candidate query is awarded full correctness **if and only if** it matches the reference result across **all** mutated states in the test suite:
$$\mathcal{R}_{\text{TS}} = \prod_{i=0}^{k} \mathbb{I}\left(Q_{\text{gen}}(\mathcal{D}_i) == Q_{\text{gold}}(\mathcal{D}_i)\right)$$

#### Tier 5: Multi-Turn Turn-Budget & Recovery Bonus ($\mathcal{R}_{\text{multi-turn}}$)
- In multi-turn agentic settings (SkyRL-Agent), the agent is given a maximum turn budget $T_{\max} = 5$.
- **Step Penalty**: A small discount penalty ($-0.05 \times t$) applied per interaction step to encourage concise, efficient SQL generation and prevent infinite exploratory loops.
- **Self-Correction Recovery Bonus**: If turn $t-1$ produced a database error (e.g., column syntax error) and turn $t$ successfully resolved the error and achieved execution equivalence, a $+0.3$ recovery bonus is awarded. This explicitly rewards the model for using database error feedback as an observational signal.

---

## 5. Multi-Turn Database Agent Architecture (SkyRL Alignment)

Single-turn NL2SQL forces the model to perform schema resolution, table joins, aggregation, and syntax formatting in a single blind shot. When faced with large schemas (e.g., 50+ tables in enterprise databases), single-turn models suffer from context dilution and hallucinated joins.

### 5.1 The Agentic Interaction Loop
Following the **SkyRL-Agent** formulation, we frame NL2SQL as an interactive Markov Decision Process (MDP) where the agent can interact with the environment via three explicit tools before producing the final query:

1. **`Tool: inspect_schema(tables: list[str])`**: Returns schema definitions (DDL) and primary/foreign key relationships for requested tables.
2. **`Tool: sample_data(table: str, limit: int = 3)`**: Returns 3 representative rows from a table to allow the model to inspect string casing, date formats, or categorical values.
3. **`Tool: execute_sql(query: str)`**: Executes a trial SQL query against the read-only sandbox and returns the tabular result or the raw database error message.
4. **`Action: FinalAnswer(query: str)`**: Submits the final SQL query for formal evaluation and terminates the rollout.

```
       USER QUERY: "What is the average salary of employees hired after 2020 in Engineering?"
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| TURN 1: [Observation] Full schema is large (28 tables).                           |
|         [Action]      Tool: inspect_schema(["employees", "departments"])          |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| TURN 2: [Observation] DDL returned. Notice "dept_id" foreign key.                 |
|         [Action]      Tool: sample_data("employees", limit=2)                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| TURN 3: [Observation] Sample shows dates formatted as 'YYYY-MM-DD'.               |
|         [Action]      Tool: execute_sql("SELECT avg(salary) FROM employees ...")  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| TURN 4: [Observation] Execution succeeds, returns 1 row with expected scalar.     |
|         [Action]      Action: FinalAnswer("SELECT avg(salary) FROM ...")          |
+-----------------------------------------+-----------------------------------------+
```

### 5.2 Curated Data Strategy: Quality Over Scale
As demonstrated by SkyRL-SQL, massive datasets (hundreds of thousands of examples) are **not** required when verifiable rewards and multi-turn environments are present. 

We adopt the curated **SkyRL-SQL-653** dataset strategy:
- **Training Set (653 samples)**:
  - 540 high-density samples curated from `SynSQL-2.5M` covering nested joins, recursive CTEs, and complex aggregations.
  - 113 challenging cross-domain samples from `Spider`.
- **Evaluation Benchmark Suite**:
  - `Spider (Dev & Test)`: Standard cross-domain benchmark (1,034 queries across 200 databases).
  - `Spider-DK`: Tests robustness against domain knowledge injection where column names do not match question terminology.
  - `Spider-Syn`: Tests robustness against synonym substitutions in questions.
  - `Spider-Realistic`: Removes explicit schema hints from natural language questions, mirroring messy human prompts.

---

## 6. Training Pipeline & Framework Hedging

### 6.1 Two-Stage Training Curriculum

```
+-------------------------------------------------------------------------------+
| STAGE 1: Supervised Fine-Tuning (SFT Warmup)                                  |
| - Dataset: SkyRL-SQL-653 + SynSQL Format Subsets (~1,500 samples)              |
| - Objective: Standard Next-Token Prediction                                   |
| - Purpose: Warm up model to tool-calling syntax, DDL parsing, and XML/JSON    |
|   action formatting. Prevents early RL exploration collapse.                  |
+---------------------------------------+---------------------------------------+
                                        |
                             models/sft/final (DVC tracked)
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STAGE 2: Reinforcement Learning with Verifiable Rewards (RLVR)                |
| - Framework: SkyRL-Agent (Primary) / veRL / TRL GRPO (Adapters)               |
| - Algorithm: Group Relative Policy Optimization (GRPO) or PPO                 |
| - Reward: Strict Hierarchical Rubric (Syntax -> Schema -> Test-Suite EX)      |
| - Environment: Multi-turn DB Sandbox (Rollouts: 8 completions per prompt)     |
+---------------------------------------+---------------------------------------+
                                        |
                            models/rlvr/final (DVC tracked)
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STAGE 3: Systematic Benchmark Evaluation                                      |
| - Suites: Spider, Spider-DK, Spider-Syn, Spider-Realistic                     |
| - Evaluation Metrics: TS (Test Suite), EX (Execution Acc), VA (Valid Rate)    |
| - Artifacts: docs/ablation_report.md + W&B/Weave Multi-Turn Traces            |
+-------------------------------------------------------------------------------+
```

### 6.2 Managing Open-Source Framework Churn (The Adapter Architecture)
Open-source RL frameworks (veRL, SkyRL, OpenRLHF, NeMo RL) evolve rapidly and can be abandoned. To protect the investment of this project:

1. **Framework-Agnostic Interface**: The training loop interacts exclusively with `src/ask_before_sql/training/adapters/base_adapter.py`.
2. **Adapter Implementation**:
   - `SkyRLAdapter`: Bridges `src/ask_before_sql/environment` into `SkyRL-Gym` and `SkyRL-Train`.
   - `VerlAdapter`: Bridges the environment to veRL's Ray rollout workers.
   - `TrlGrpoAdapter`: Bridges reward functions into Hugging Face `trl.GRPOTrainer` for lightweight local runs.
3. **Portability Test Harness (`tests/test_portability.py`)**:
   A deterministic test script that runs **50 fixed samples** using a lightweight 1.5B model (`Qwen2.5-Coder-1.5B`) through both candidate frameworks. It asserts that reward assignments, trajectory lengths, and gradient updates match before committing to an 80GB GPU cluster sweep.

---

## 7. Systematic Evaluation Suite & Metrics

In SQL generation, reporting only "BLEU" or "Exact Match (EM)" syntax accuracy is scientifically invalid because two semantically identical queries may use completely different syntax. 

Our automated evaluation pipeline (`scripts/evaluate_benchmark.py`) computes four primary metrics:

1. **Test-Suite Execution Accuracy (TS)** *(Primary Leaderboard Metric)*:
   $$\text{TS} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\forall d \in \text{TestSuites}_i, \, Q_{\text{gen}}(d) == Q_{\text{gold}}(d)\right)$$
   Measures whether the query returns the exact correct answer across all mutated database states. Eliminates false positives.
2. **Standard Execution Accuracy (EX)**:
   Measures execution match on the single primary benchmark database state.
3. **Valid Execution Rate (VA)**:
   $$\text{VA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(Q_{\text{gen}} \text{ executes without database errors}\right)$$
   Measures structural validity and schema compliance.
4. **Multi-Turn Recovery Rate (MRR)**:
   $$\text{MRR} = \frac{\sum \text{Queries resolved after } \ge 1 \text{ syntax error}}{\sum \text{Queries with initial syntax error}}$$
   Measures the agent's ability to self-correct using error messages from the database environment.

---

## 8. HPC Cluster & Slurm Operations Blueprint

Cluster execution must follow the lessons formalized in AskBeforeAnswer: defensive preambles, sequential array throttling (`%1`), and isolated environment variables.

### 8.1 Universal Slurm Preamble
Every `.slurm` script in `slurm/` must strictly implement this preamble:

```bash
#!/bin/bash
# ---------------------------------------------------------------------------
# UNIVERSAL PREAMBLE: Environment & Secrets Isolation
# ---------------------------------------------------------------------------
# 1. Enforce submit directory snap-back
cd "${SLURM_SUBMIT_DIR}"

# 2. Activate Conda Environment dynamically
source ~/.conda/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ask-before-sql

# 3. Dynamic library path injection (eliminates CXXABI/GLIBC errors)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 4. Enforce clean Python user site isolation
export PYTHONNOUSERSITE=1

# 5. Load environment secrets (.env)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi
```

### 8.2 Slurm Job Scripts

1. `01_preprocess.slurm` *(CPU Queue, 8 cores, 32GB RAM)*:
   Downloads Spider and SynSQL benchmarks, generates SQLite databases, and caches test-suite mutations.
2. `02_sft_warmup.slurm` *(GPU Queue, 1x A100/A6000, 8 hours)*:
   Executes `make train-sft` to establish the baseline model checkpoint.
3. `03_rlvr_train.slurm` *(GPU Queue, 1x A100/H100 80GB, 24–48 hours)*:
   Executes `make train-rlvr` using SkyRL-Agent multi-turn rollout loops.
4. `04_sweep_array.slurm` *(Sequential Array Queue, `#SBATCH --array=1-8%1`, 24 hours)*:
   Throttles Bayesian sweep agents so each array task runs sequentially, allowing the W&B Gaussian Process to incorporate the results of prior trials and bypassing 24h queue walltime cutoffs.
5. `05_evaluate.slurm` *(GPU Queue, 1x GPU, 4 hours)*:
   Executes the multi-benchmark evaluation suite across Spider, Spider-DK, Spider-Syn, and Spider-Realistic.

---

## 9. Phased Implementation Roadmap & GitHub Issues Breakdown

This architecture plan is designed to be directly converted into **5 Phased Epics** and granular GitHub issues.

```
+---------------------------------------------------------------------------------------+
| PHASE 1: Repository Foundation, SSoT Architecture & Database Sandboxing               |
|          (Issues #1 to #5)                                                            |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 2: Verifiable Reward Engine & Test-Suite Verification                           |
|          (Issues #6 to #10)                                                           |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 3: SFT Warmup Pipeline & Framework Adapters                                     |
|          (Issues #11 to #15)                                                          |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 4: Multi-Turn Agent Environment & SkyRL Integration                             |
|          (Issues #16 to #20)                                                          |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| PHASE 5: Benchmarking, Sweeps, Slurm Deployment & Production UI                       |
|          (Issues #21 to #25)                                                          |
+---------------------------------------------------------------------------------------+
```

### Phase 1: Repository Foundation, SSoT Architecture & DB Sandbox
- **Issue #1: Repository Skeleton & SSoT Toolchain Configuration**
  - Implement `pyproject.toml`, `requirements.txt`, `Makefile`, and Hydra root `configs/config.yaml`.
  - Establish `ruff` linting, `mypy` strict type checking, and GitHub Actions CI.
- **Issue #2: DVC Pipeline & Remote Storage Setup**
  - Configure `dvc.yaml` and `.dvcignore` to track `data/` and `models/` artifacts.
- **Issue #3: SQLite Ephemeral Sandbox Implementation**
  - Implement `src/ask_before_sql/environment/sqlite_env.py` with read-only enforcement, timeout signal handlers, and memory bounding.
- **Issue #4: Benchmark Data Ingestion & Preprocessing**
  - Create `scripts/preprocess_data.py` to ingest Spider and `SkyRL-SQL-653-data-newfmt`, converting them into unified schema formats.
- **Issue #5: Unit Testing for Sandbox Safety & Isolation**
  - Author comprehensive test suite in `tests/unit/test_sandbox.py` verifying that destructive queries (`DROP`, `DELETE`, recursive Cartesian explosions) are safely intercepted.

### Phase 2: Verifiable Reward Engine & Test-Suite Verification
- **Issue #6: SQL Syntax & AST Validator Implementation**
  - Build `src/ask_before_sql/rewards/syntax_reward.py` using `sqlglot` for syntax verification and AST dialect normalization.
- **Issue #7: Execution Equivalence Evaluator (EX)**
  - Implement `src/ask_before_sql/rewards/execution_reward.py` comparing output tables with float tolerance, null handling, and order-agnostic multiset matching.
- **Issue #8: Test-Suite (TS) Database Mutation Engine**
  - Implement `src/ask_before_sql/environment/test_suite_evaluator.py` to generate mutated database states, eliminating spurious false-positive rewards.
- **Issue #9: Hierarchical Reward Rubric Composite**
  - Implement `src/ask_before_sql/rewards/rubric.py` integrating syntax penalties, schema linking signals, and test-suite execution rewards into a bounded score.
- **Issue #10: Unit Testing for Reward Rubric Edge Cases**
  - Author tests in `tests/unit/test_rewards.py` verifying behavior on empty result sets, division by zero, duplicate rows, and schema mismatches.

### Phase 3: SFT Warmup Pipeline & Framework Adapters
- **Issue #11: Abstract Framework Adapter Interface**
  - Define `src/ask_before_sql/training/adapters/base_adapter.py` establishing standard rollout, reward evaluation, and checkpoint hooks.
- **Issue #12: Supervised Fine-Tuning (SFT) Training Pipeline**
  - Implement `scripts/train_sft.py` supporting QLoRA (4-bit/8-bit) and full precision training on `Qwen2.5-Coder-7B-Instruct` via Hydra configuration.
- **Issue #13: TRL GRPO Adapter (Lightweight Baseline)**
  - Build `src/ask_before_sql/training/adapters/trl_grpo_adapter.py` enabling single-node GRPO experimentation using Hugging Face `trl`.
- **Issue #14: Framework Portability Test Harness**
  - Implement `tests/test_portability.py` testing candidate adapters on 50 samples to verify gradient stability and reward alignment.
- **Issue #15: DVC Pipeline Integration for SFT**
  - Add `preprocess` and `train-sft` stages to `dvc.yaml` and verify deterministic reproduction via `make train-sft`.

### Phase 4: Multi-Turn Agent Environment & SkyRL Integration
- **Issue #16: Multi-Turn Interaction State & Tool Registry**
  - Build `src/ask_before_sql/agent/interaction_state.py` and `tool_registry.py` providing `inspect_schema`, `sample_data`, and `execute_sql` actions.
- **Issue #17: SkyRL-Gym Environment Implementation**
  - Implement `src/ask_before_sql/training/adapters/skyrl_adapter.py` bridging the multi-turn SQL environment into SkyRL-Gym.
- **Issue #18: Multi-Turn Step Discount & Self-Correction Rewards**
  - Implement `src/ask_before_sql/rewards/multi_turn_reward.py` awarding bonuses for self-correction after execution errors and penalizing step budget exhaustion.
- **Issue #19: Multi-Turn RLVR Training Pipeline**
  - Create `scripts/train_rlvr.py` orchestrating multi-turn agentic RL rollouts using SkyRL-Agent.
- **Issue #20: Multi-Turn Interactive Integration Tests**
  - Author integration tests in `tests/integration/test_multiturn.py` verifying full trajectory rollouts from initial query to verified database answer.

### Phase 5: Benchmarking, Sweeps, Slurm Deployment & Production UI
- **Issue #21: Comprehensive Evaluation Harness (Spider, DK, Syn, Realistic)**
  - Implement `scripts/evaluate_benchmark.py` computing TS, EX, VA, and MRR metrics across all Spider test variants.
- **Issue #22: W&B / Weave Trajectory Tracing & Automated Reporting**
  - Implement `scripts/generate_report.py` logging multi-turn execution graphs to Weave and generating `docs/evaluation_leaderboard.md`.
- **Issue #23: Bayesian Sweep Infrastructure & Array Throttling**
  - Define `configs/sweeps/rlvr_sweep.yaml` and implement `slurm/04_sweep_array.slurm` with `#SBATCH --array=1-8%1` sequential scheduling.
- **Issue #24: Hardened Slurm Cluster Scripts & Documentation**
  - Author production `.slurm` wrappers in `slurm/` with Conda environment activation, dynamic library paths, and submit-directory snaps.
- **Issue #25: Interactive Streamlit UI & SSH Tunneling Deployment**
  - Implement `app/app.py` providing an interactive database inspection and text-to-SQL query playground, supported by `slurm/05_demo_tunnel.slurm` for cluster tunneling.

---

## 10. Conclusion & Strategic Next Steps

By adapting the architectural rigor of **AskBeforeAnswer**—strict Single Source of Truth, DVC reproducibility, hardened Slurm operations, and modular design—to the verifiable realm of **SQL RLVR and SkyRL-Agent**, this project establishes a rock-solid research and engineering foundation. 

The framework-independent reward evaluator guarantees scientific longevity, while the multi-turn database exploration agent enables state-of-the-art data efficiency on complex relational benchmarks.

### Immediate Action Items:
1. Initialize the new repository (`ask-before-sql`) using the directory structure outlined in **Section 3**.
2. Convert **Section 9** into formal GitHub milestones, epics, and issues.
3. Begin execution of **Phase 1 (Issues #1 to #5)**.
