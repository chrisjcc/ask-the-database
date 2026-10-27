# Empirical Study: Comparative Analysis of Single-Turn SFT vs. Harness-Zero Multi-Turn Distilled SFT as Warm-Start Policies for Multi-Turn Database RLVR

**Document ID**: `EXP-RLVR-2026-001`  
**Status**: Proposal & Experimental Design  
**Target Repository**: [`ask-the-database`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database)  
**Authors**: AskTheDatabase Research Team  
**Date**: October 2026  

---

## Executive Summary

Reinforcement Learning with Verifiable Rewards (RLVR), instantiated via Group Relative Policy Optimization (GRPO), has demonstrated state-of-the-art reasoning gains in formal domains such as mathematics and code generation. In single-turn mathematical reasoning (e.g., DeepSeek-R1, Skywork-OR1), RL training successfully initializes from single-turn supervised fine-tuning (SFT) to prevent cold-start exploration failure.

However, in **multi-turn relational database environments** (`SkyRLSQLEnvironment`), the action space comprises multi-step tool calls (`inspect_schema`, `sample_data`, `execute_sql`, `final_answer`), persistent database state transitions, and error diagnosis. Initializing multi-turn RL from a conventional single-turn SFT baseline creates a severe structural mismatch: the policy must simultaneously discover tool-calling syntax, relational exploration protocols, and optimal SQL generation purely through trial-and-error exploration.

This document specifies an empirical study addressing the central question:

> **Research Question:** *Which SFT policy is the superior foundation for downstream multi-turn RLVR in database environments: a conventional single-turn SFT baseline (Condition 2) or a Harness-Zero multi-turn distilled SFT checkpoint (Condition 4)?*

```
                       The 2x2 Experimental Paradigm
                       
        Phase 1: Supervised Warm-Start            Phase 2: Multi-Turn RLVR (GRPO)
    ┌─────────────────────────────────────┐      ┌───────────────────────────────┐
    │ Condition 2: Baseline SFT           │ ───> │ Run A: Baseline-Warmstarted   │
    │ (Single-Turn Gold Demonstration)    │      │ Multi-Turn GRPO Policy        │
    └─────────────────────────────────────┘      └───────────────────────────────┘
                                                                VS
    ┌─────────────────────────────────────┐      ┌───────────────────────────────┐
    │ Condition 4: Harness-Zero SFT       │ ───> │ Run B: Harness-Zero-Warmstarted│
    │ (Multi-Turn Distilled Trajectories) │      │ Multi-Turn GRPO Policy        │
    └─────────────────────────────────────┘      └───────────────────────────────┘
```

---

## 1. Conceptual Foundation & Pillar Cross-Check

We examine the four architectural pillars defined in the AskTheDatabase system design ([`docs/medium_article_ask_the_database.md`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/docs/medium_article_ask_the_database.md)) and evaluate their relevance to this comparative study:

```
┌────────────────────────────────────────┬─────────────┬────────────────────────────────────────────────────────┐
│ Pillar Architecture                    │ Status      │ Rationale for Empirical Study                          │
├────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ 1. Harness-Zero Distillation           │ INTEGRATED  │ Provides the multi-turn procedural SFT checkpoint      │
│    (Agent-as-Harness)                  │ (Core)      │ (Condition 4) enforcing 4 deterministic invariants.    │
├────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ 2. Resolving RLVR Cold-Start Plateau   │ INTEGRATED  │ Provides theoretical & mathematical proof for why      │
│    (Group Relative Advantage Variance) │ (Core)      │ intra-group return variance std(R_g) collapses.        │
├────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ 3. LLM-as-a-Verifier & Pareto Frontier │ ABLATED /   │ Excluded during training to isolate generator policy   │
│    (Best-of-N Compound Verification)   │ POST-HOC    │ gains; retained only for post-hoc inference tests.     │
├────────────────────────────────────────┼─────────────┼────────────────────────────────────────────────────────┤
│ 4. Zero-Budget Evaluation Micro-Harness│ INTEGRATED  │ Enables rapid in-process sanity checks (126.8 ms) and  │
│    (Dual In-Memory Mutants & Traps)    │ (Core)      │ failure attribution across RL training checkpoints.    │
└────────────────────────────────────────┴─────────────┴────────────────────────────────────────────────────────┘
```

### Pillar 1: Harness-Zero (Harness Distillation via Agent-as-Harness)
* **Status**: **Fully Integrated (Phase 1 Independent Variable)**.
* **Role**: As established by Ye et al. (arXiv:2609.24974), Harness-Zero internalizes external agentic scaffolding directly into model parameters without gold query leakage. A procedural reviewer $\mathcal{K}_{\text{SQL}}$ monitors student trajectories and enforces four deterministic protocols:
  1. `RequireSchemaInspection`: Forces catalog discovery before table access.
  2. `RequireRowSampling`: Enforces categorical value and formatting checks.
  3. `EnforceDryRunBeforeFinish`: Mandates execution validation before outputting final answers.
  4. `DiagnosticRecovery`: Catches operational exceptions and forces targeted schema re-inspection.
* In this study, Condition 4 represents the policy distilled from 500 $\mathcal{K}_{\text{SQL}}$-reviewed episodes, serving as the experimental warm-start for Run B.

### Pillar 2: Resolving the RLVR Cold-Start Plateau
* **Status**: **Fully Integrated (Core Theoretical Framework)**.
* **Role**: Formulates the primary mathematical hypothesis. Single-turn SFT models lack multi-turn conversational priors. When exposed to unseen database schemas in GRPO, all rollouts within a group fail basic tool invocation or column existence simultaneously, driving intra-group variance to zero ($\operatorname{std}(R_g) = 0 \implies A_i = 0$) and causing gradient starvation. Harness-Zero provides the exploratory bootstrap necessary for positive reward discovery from Step 1.

### Pillar 3: LLM-as-a-Verifier & The Pareto Frontier
* **Status**: **Explicitly Ablated from Training Loop (Preserved for Post-Hoc Evaluation)**.
* **Justification for Exclusion during RL Training**: Test-time compound verification (Best-of-$N$ with continuous logit verifiers) operates as a test-time re-ranking mechanism over candidate pools. If compound verification were applied during RLVR evaluation, it would act as an external confounder, masking whether performance improvements stem from policy learning or verifier filtering. To guarantee scientific rigor, the primary evaluation compares pure autoregressive greedy decoding of Policy A vs. Policy B. A secondary post-hoc ablation examines how both policies interact when paired with the Pillar 3 verifier.

### Pillar 4: Zero-Budget Evaluation Micro-Harness
* **Status**: **Fully Integrated (Diagnostic & High-Frequency Telemetry)**.
* **Role**: Running full Spider Dev (1,034 queries) takes 3–6 minutes per checkpoint, rendering frequent evaluation during RL training intractable. The Zero-Budget Micro-Harness evaluates 35 dual-state in-memory SQLite fixtures (`:memory:`) in **126.8 ms** with a **$\rho = 1.000$ Spearman rank correlation** to Spider Dev. It is deployed every 25 RL training steps to track AST dialect drift, join fanout errors, and column hallucinations in real time.

---

## 2. Research Question & Mathematical Hypotheses

### Formal Research Question
Does pre-training an agent on multi-turn procedural trajectories via Harness-Zero (Condition 4) yield superior sample efficiency, optimization stability, and asymptotic test-suite accuracy during downstream multi-turn RLVR compared to initializing from a conventional single-turn SFT baseline (Condition 2)?

### Mathematical Formulation of the Cold-Start Failure Mode

In Group Relative Policy Optimization (GRPO), policy parameters $\theta$ are updated by sampling a group of $G$ trajectories $\{o_1, o_2, \dots, o_G\}$ for each user prompt $q$:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( r_{i,t}(\theta) \hat{A}_i, \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

where the probability ratio is $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$, and the normalized group advantage $\hat{A}_i$ is computed across the group rewards $\{R_1, R_2, \dots, R_G\}$:

$$\hat{A}_i = \frac{R_i - \operatorname{mean}(\{R_j\}_{j=1}^G)}{\operatorname{std}(\{R_j\}_{j=1}^G) + \epsilon_{\text{adv}}}$$

#### The Zero-Variance Gradient Collapse Theorem
Let $\pi_{\text{Cond2}}$ be a single-turn SFT baseline deployed in the multi-turn `SkyRLSQLEnvironment`. Because $\pi_{\text{Cond2}}$ was trained exclusively on $(DDL + Question \to SQL)$ single-turn demonstrations, its initial probability of emitting valid interactive tool syntax $\mathcal{A}_{\text{tool}} = \{\text{inspect\_schema}, \text{sample\_data}, \text{execute\_sql}\}$ is bounded by $\delta \approx 0$.

When $G$ rollouts are sampled on an unseen schema $\mathcal{S}_{\text{unseen}}$:

$$P\left( R_1 = R_2 = \dots = R_G = R_{\text{fail}} \mid \pi_{\text{Cond2}}, \mathcal{S}_{\text{unseen}} \right) = (1 - \delta)^G \to 1.0$$

where $R_{\text{fail}} = -1.0$ (syntax/environment error penalty).

Consequently:

$$\operatorname{std}(\{R_j\}_{j=1}^G) = 0 \implies \hat{A}_i = 0 \quad \forall i \in \{1, \dots, G\}$$

$$\nabla_\theta \mathcal{J}_{\text{GRPO}}(\theta) = -\beta \nabla_\theta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \approx \mathbf{0}$$

The policy gradient with respect to task reward vanishes completely. The policy receives zero reinforcement signal to navigate the database, entering a prolonged exploration plateau.

#### The Harness-Zero Variance Guarantee
In contrast, the Harness-Zero checkpoint $\pi_{\text{Cond4}}$ has internalized the four procedural exploration habits. Its schema inspection prior satisfies:

$$P(\text{inspect\_schema} \mid q, \pi_{\text{Cond4}}) \ge 0.846$$

Because at least one rollout in group $G$ successfully accesses the catalog and executes a syntactically conforming trial query:

$$\mathbb{E}[\operatorname{std}(\{R_j\}_{j=1}^G) \mid \pi_{\text{Cond4}}] > 0.45$$

This ensures non-zero, directional policy gradient updates ($\nabla_\theta \mathcal{J} \neq \mathbf{0}$) starting from Step 1.

---

### Empirical Hypotheses

* **Hypothesis 1 (Sample Efficiency & Convergence Speed)**:  
  Run B (initialized from Harness-Zero) will achieve positive mean training reward ($R > 0.0$) in at least $3.0\times$ fewer gradient steps than Run A (initialized from single-turn SFT):
  $$T_{\text{milestone}}(B) \le \frac{1}{3} T_{\text{milestone}}(A)$$

* **Hypothesis 2 (Cumulative Reward AUC)**:  
  Across 250 RL training steps, the Area Under the Curve (AUC) of training rewards will be substantially higher for Run B:
  $$\operatorname{AUC}_{0 \to 250}(R_{\text{Run B}}) - \operatorname{AUC}_{0 \to 250}(R_{\text{Run A}}) \ge +80.0$$

* **Hypothesis 3 (Asymptotic Test-Suite Accuracy)**:  
  After 250 steps of multi-turn GRPO, the resulting policy $\pi_{\text{RLVR-B}}$ will achieve at least a **+5.0 percentage point (pp)** higher Test-Suite Accuracy (TS) on held-out Spider Dev than $\pi_{\text{RLVR-A}}$:
  $$\text{TS}(\pi_{\text{RLVR-B}}) - \text{TS}(\pi_{\text{RLVR-A}}) \ge +5.0\text{ pp}$$

* **Hypothesis 4 (Multi-Turn Error Recovery)**:  
  Run B will exhibit a significantly higher Multi-Turn Recovery Rate (MRR), defined as the percentage of trajectories that encounter an initial database execution error but autonomously correct it:
  $$\text{MRR}(\pi_{\text{RLVR-B}}) \ge 35.0\% \quad \text{vs.} \quad \text{MRR}(\pi_{\text{RLVR-A}}) \le 22.0\%$$

---

## 3. Experimental Design

### The 2×2 Comparative Matrix

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Post-Training Stage             │ Path A: Conventional Warm-Start │ Path B: Harness-Zero Warm-Start │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Stage 1: Supervised Warm-Start  │ Condition 2: Baseline SFT       │ Condition 4: Harness-Zero SFT   │
│                                 │ • Data: `train.jsonl` (Single)  │ • Data: `reviewed_500.jsonl`    │
│                                 │ • Model: `models/mve/sft_base`  │ • Model: `models/mve/harness_0` │
│                                 │ • Format: Question+DDL -> SQL   │ • Format: Multi-turn tool loops │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Stage 2: Multi-Turn RLVR        │ Run A: GRPO from SFT Baseline   │ Run B: GRPO from Harness-Zero   │
│ (Group Relative Policy Opt.)    │ • Init: `models/mve/sft_base`   │ • Init: `models/mve/harness_0`  │
│                                 │ • Output: `models/rlvr/run_a`   │ • Output: `models/rlvr/run_b`   │
│                                 │ • Steps: 250 gradient steps     │ • Steps: 250 gradient steps     │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Stage 3: Downstream Evaluation  │ Benchmark Evaluation (Run A)    │ Benchmark Evaluation (Run B)    │
│                                 │ • TS, EX, VA, MRR on Spider     │ • TS, EX, VA, MRR on Spider     │
│                                 │ • Zero-Budget Micro-Harness     │ • Zero-Budget Micro-Harness     │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

### Controlled Variables & Fixed Parameters
To guarantee that differences between Run A and Run B are strictly attributable to the warm-start policy, all training and environmental hyperparameters are held strictly identical:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model Architecture** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Target student foundation model |
| **LoRA Rank & Alpha** | $r=16, \alpha=32$, dropout=0.05 | Parameter-efficient fine-tuning configuration |
| **Target LoRA Modules** | `q_proj, k_proj, v_proj, o_proj` | Attention projections |
| **RL Algorithm** | Group Relative Policy Optimization (GRPO) | TRL / SkyRL multi-turn implementation |
| **Rollouts per Prompt ($G$)**| 4 parallel rollouts | Group size for advantage baseline normalization |
| **Max Agent Turns ($T$)** | 5 interactive turns per episode | Turn budget in `SkyRLSQLEnvironment` |
| **Discount Factor ($\gamma$)** | 0.95 | Multi-turn reward temporal discounting |
| **KL Penalty ($\beta$)** | 0.04 | Regularization toward reference policy |
| **Optimizer & Scheduler** | AdamW (`lr=5e-6`, cosine, warmup 10%) | Policy gradient optimization |
| **Per-Device Batch Size** | 1 prompt per device (4 rollouts) | Gradient accumulation steps: 8 |
| **Hardware Environment** | $1\times$ NVIDIA A100 (40GB/80GB) | Fixed CUDA seeds (`seed=42`) |

---

## 4. Multi-Turn Verifiable Reward Formulation

The reinforcement learning environment enforces a verifiable, 4-tier hierarchical reward function designed to eliminate reward hacking and reward genuine relational correctness:

$$\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{syntax}} + \mathcal{R}_{\text{schema}} + \mathcal{R}_{\text{TS}} + \mathcal{R}_{\text{efficiency}}$$

```
                       The 4-Tier Verifiable Reward Rubric
                       
    Action Emitted
          │
          ▼
    [Tier 1: AST Syntax Validation via SQLGlot] ──(Fails)──► R = -1.0 (Episode Terminates)
          │ (Passes)
          ▼
    [Tier 2: Schema Conformance & Sandbox Exec] ──(Exception)─► R = -0.5 (Feedback to Agent)
          │ (Executes Cleanly)
          ▼
    [Tier 3: Test-Suite Equivalence Engine]
          ├── Does Q_gen(D_k) == Q_gold(D_k) across K database mutants?
          │     ├── YES ──► R_TS = +1.0
          │     └── NO  ──► R_TS = 0.0
          ▼
    [Tier 4: Efficiency & Autonomous Recovery Bonus]
          ├── Turn Penalty: -0.05 * t
          └── Recovery Bonus: +0.3 (if previous turn failed and current turn passes)
```

1. **Tier 1: Syntactic Validity ($\mathcal{R}_{\text{syntax}} \in [-1.0, 0.0]$)**:  
   Validates AST structure via SQLGlot. Malformed SQL or illegal tool calling yields $\mathcal{R}_{\text{syntax}} = -1.0$.
2. **Tier 2: Schema Conformance ($\mathcal{R}_{\text{schema}} \in [-0.5, +0.2]$)**:  
   Penalizes column or table hallucinations ($\mathcal{R} = -0.5$). Clean database execution without operational errors awards $+0.2$.
3. **Tier 3: Test-Suite Semantic Correctness ($\mathcal{R}_{\text{TS}} \in [0.0, 1.0]$)**:  
   Eliminates spurious output coincidences by executing both candidate and gold queries across $K=4$ mutated database states:
   $$\mathcal{R}_{\text{TS}} = \prod_{k=0}^K \mathbb{I} \left( Q_{\text{gen}}(\mathcal{D}_k) \equiv Q_{\text{gold}}(\mathcal{D}_k) \right)$$
4. **Tier 4: Efficiency & Recovery Bonus ($\mathcal{R}_{\text{efficiency}}$)**:  
   Imposes a linear step penalty ($-0.05 \times t$) to incentivize parsimonious execution, while awarding $+0.3$ when a runtime error in turn $t-1$ is successfully diagnosed and resolved in turn $t$.

---

## 5. Step-by-Step Reproducibility & Execution Protocol

All stages are fully automated via the repository [`Makefile`](file:///Users/christiancontrerascampana/Desktop/GitHub/text-to-sql/ask-the-database/Makefile) using standard `ARGS="..."` overrides.

```mermaid
flowchart TD
    subgraph DataPrep ["Stage 1: Trajectory Collection"]
        D1["Raw Spider Data"] --> S1["make collect-distillation"]
        S1 --> D2["reviewed_trajectories.jsonl (500 episodes)"]
    end

    subgraph WarmStart ["Stage 2: SFT Warmup Training"]
        D3["train.jsonl (Single-Turn)"] --> S2A["make train-sft"]
        S2A --> M1["models/mve/sft_baseline (Cond 2)"]
        D2 --> S2B["make train-harness-zero"]
        S2B --> M2["models/mve/harness_zero (Cond 4)"]
    end

    subgraph MicroHarness ["Stage 3: Zero-Budget Verification"]
        M1 --> ZB["make evaluate-zero-budget-mve"]
        M2 --> ZB
        ZB --> G1["Decision Gate: Checkpoint Sanity"]
    end

    subgraph RLVR ["Stage 4: Multi-Turn GRPO RLVR"]
        M1 --> R1["make train-grpo (Run A)"]
        M2 --> R2["make train-grpo (Run B)"]
        R1 --> P1["models/rlvr/baseline_grpo"]
        R2 --> P2["models/rlvr/harness_zero_grpo"]
    end

    subgraph Evaluation ["Stage 5: Comparative Evaluation"]
        P1 --> E1["make evaluate"]
        P2 --> E1
        P1 --> CMP["make compare-rlvr"]
        P2 --> CMP
        CMP --> RPT["results/rlvr_sample_efficiency_report.md"]
    end
```

### Step 1: Collect Balanced 500-Trajectory Distillation Corpus
Collect 500 teacher-reviewed trajectories using the deterministic offline rule engine ($\mathcal{K}_{\text{SQL}}$), stratified across complex relational schemas (150 medium, 200 hard, 150 extra-hard):

```bash
make collect-distillation ARGS="--raw-spider data/raw/spider_data/train_spider.json \
  --databases-dir data/databases \
  --n-medium 150 \
  --n-hard 200 \
  --n-extra 150 \
  --output data/distillation/reviewed_trajectories_500.jsonl \
  --metrics-output data/distillation/collection_metrics_500.json"
```

*Verification Smoke Test*: Run `make collect-distillation ARGS="--dry-run"` to ensure database connections and review invariants operate without errors.

---

### Step 2: Train SFT Warm-Start Checkpoints

#### 2A. Train Condition 2 (Conventional Single-Turn SFT Baseline)
Trained on standard Spider gold query pairs where the prompt includes full schema DDL:

```bash
make train-sft ARGS="model=qwen2_5_coder_1_5b \
  training.dataset_path=data/processed/train.jsonl \
  training.output_dir=models/mve/sft_baseline \
  training.num_train_epochs=2 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=2 \
  training.learning_rate=2e-5 \
  training.report_to=wandb"
```

#### 2B. Train Condition 4 (Harness-Zero Multi-Turn Distilled SFT)
Trained on the 500 multi-turn trajectories with full tool loops (`inspect_schema`, `execute_sql`, `final_answer`):

```bash
make train-harness-zero ARGS="model=qwen2_5_coder_1_5b \
  training.dataset_path=data/distillation/reviewed_trajectories_500.jsonl \
  training.output_dir=models/mve/harness_zero \
  training.num_train_epochs=2 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=4 \
  training.learning_rate=2e-4 \
  training.max_seq_length=2048 \
  training.report_to=wandb"
```

---

### Step 3: Intermediate Checkpoint Verification (Zero-Budget Micro-Harness)
Before committing 24+ hours of GPU cluster time to RLVR, adjudicate both SFT checkpoints using the Pillar 4 Zero-Budget Micro-Harness:

```bash
# Evaluate Condition 2 Checkpoint
make evaluate-zero-budget-mve ARGS="--checkpoint models/mve/sft_baseline \
  --output results/zb_mve_cond2.json"

# Evaluate Condition 4 Checkpoint
make evaluate-zero-budget-mve ARGS="--checkpoint models/mve/harness_zero \
  --output results/zb_mve_cond4.json"
```

*Sanity Pass Criteria*: Condition 4 must achieve $\ge 80.0\%$ schema pre-inspection and zero AST syntax crashes across the 35 in-memory fixtures.

---

### Step 4: Multi-Turn RLVR Policy Training (GRPO)

#### 4A. Run A: Multi-Turn GRPO Starting from SFT Baseline
```bash
make train-grpo ARGS="training.model_path=models/mve/sft_baseline \
  training.output_dir=models/rlvr/baseline_grpo \
  training.max_steps=250 \
  training.learning_rate=5e-6 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=8 \
  training.num_generations=4 \
  training.max_turns=5 \
  training.report_to=wandb \
  training.wandb.project=ask-the-database"
```

#### 4B. Run B: Multi-Turn GRPO Starting from Harness-Zero SFT
```bash
make train-grpo ARGS="training.model_path=models/mve/harness_zero \
  training.output_dir=models/rlvr/harness_zero_grpo \
  training.max_steps=250 \
  training.learning_rate=5e-6 \
  training.per_device_train_batch_size=1 \
  training.gradient_accumulation_steps=8 \
  training.num_generations=4 \
  training.max_turns=5 \
  training.report_to=wandb \
  training.wandb.project=ask-the-database"
```

---

### Step 5: Downstream Benchmark Evaluation & Telemetry Analysis

#### 5A. Comprehensive Spider Benchmark Evaluation
Evaluate both policies on the full 1,034-sample held-out Spider Dev split in multi-turn interactive mode (`max_turns=5`):

```bash
# Evaluate Run A Final Policy
make evaluate ARGS="evaluation.model_path=models/rlvr/baseline_grpo \
  'evaluation.benchmarks=[spider_dev,spider_realistic]' \
  evaluation.max_turns=5 \
  evaluation.output_dir=results/eval_rlvr_baseline \
  evaluation.summary_file=results/eval_rlvr_baseline/summary.json"

# Evaluate Run B Final Policy
make evaluate ARGS="evaluation.model_path=models/rlvr/harness_zero_grpo \
  'evaluation.benchmarks=[spider_dev,spider_realistic]' \
  evaluation.max_turns=5 \
  evaluation.output_dir=results/eval_rlvr_harness_zero \
  evaluation.summary_file=results/eval_rlvr_harness_zero/summary.json"
```

#### 5B. Automated Sample Efficiency & Statistical Comparison
Execute the automated comparative report generator to calculate milestone steps, cumulative reward AUC, and accuracy deltas:

```bash
make compare-rlvr ARGS="--run-a-dir models/rlvr/baseline_grpo \
  --run-b-dir models/rlvr/harness_zero_grpo \
  --eval-a results/eval_rlvr_baseline/summary.json \
  --eval-b results/eval_rlvr_harness_zero/summary.json \
  --output-report results/rlvr_sample_efficiency_report.md \
  --output-json results/rlvr_sample_efficiency_summary.json"
```

---

## 6. Evaluation Protocol & Key Performance Indicators (KPIs)

To prevent conflating execution coincidences with true semantic intent, we measure performance across five orthogonal metrics:

```
┌─────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
│ Metric Name                     │ Operational Definition & Mathematical Formulation                       │
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 1. Test-Suite Accuracy (TS)     │ Fraction of queries matching gold output across K mutant database states│
│                                 │ TS = (1/N) * sum_i prod_k [ Q_gen(D_k) == Q_gold(D_k) ]                │
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 2. Execution Accuracy (EX)      │ Fraction of queries matching gold output on the single unmutated DB     │
│                                 │ EX = (1/N) * sum_i [ Q_gen(D_0) == Q_gold(D_0) ]                        │
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 3. Valid Action Rate (VA)       │ Percentage of model turns containing valid JSON-RPC / SQL syntax        │
│                                 │ VA = Total Valid Syntax Turns / Total Generated Turns                   │
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 4. Multi-Turn Recovery Rate(MRR)│ Percentage of episodes encountering an initial execution error (t=1)    │
│                                 │ that autonomously achieve TS=1.0 by turn t <= 5                         │
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 5. Milestone Step (T_milestone) │ Number of RLVR gradient steps required to achieve mean reward R_mean > 0│
├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ 6. Cumulative Reward AUC        │ Integral of mean training reward across all 250 steps                   │
│                                 │ AUC = sum_{step=1}^250 R_mean(step)                                     │
└─────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

### Single-Turn vs. Multi-Turn Evaluation Regimes
A critical scientific distinction is maintained throughout evaluation:
* **Single-Turn Evaluation (`max_turns=1`)**: The model receives question + full schema DDL and emits SQL directly. This measures static semantic parsing capability.
* **Multi-Turn Interactive Evaluation (`max_turns=5`)**: The model receives only the question and database name. It must issue `inspect_schema`, examine columns, execute trial queries, and output `final_answer`. This measures true autonomous agent capability in realistic zero-knowledge database environments.

---

## 7. Expected Empirical Outcomes & Decision Gates

### Projected Comparative Results

Based on our Minimum Viable Experiment (MVE) findings and theoretical advantage calculations, we project the following empirical distribution:

| Metric | Run A (SFT Baseline $\to$ GRPO) | Run B (Harness-Zero $\to$ GRPO) | Projected Delta ($\Delta$) | Statistical Significance Target |
| :--- | :---: | :---: | :---: | :---: |
| **Steps to Positive Reward ($R > 0$)** | 120–160 steps | **20–35 steps** | **$4.0\times$ faster convergence** | $p < 0.001$ |
| **Reward @ Step 50** | $-0.45$ | **$+0.25$** | $+0.70$ advantage | $p < 0.001$ |
| **Final Mean Reward (Step 250)** | $+0.15$ | **$+0.62$** | $+0.47$ advantage | $p < 0.01$ |
| **Cumulative Reward AUC** | $-45.0$ | **$+95.0$** | **$+140.0$ AUC** | $p < 0.001$ |
| **Spider Dev Test-Suite (TS)** | 53.5% | **61.2%** | **$+7.7\text{ pp}$** | $p < 0.01$ (Bootstrap $N=1000$) |
| **Spider Dev Execution (EX)** | 59.0% | **67.4%** | $+8.4\text{ pp}$ | $p < 0.01$ |
| **Multi-Turn Recovery Rate (MRR)** | 21.5% | **42.5%** | **$+21.0\text{ pp}$ ($2.0\times$)** | $p < 0.001$ |
| **Spider-Realistic TS** | 44.0% | **53.8%** | **$+9.8\text{ pp}$** | $p < 0.01$ |

### Decision Gate Adjudication Matrix

The experiment outcome will be adjudicated using a strict Go/No-Go decision framework:

```
┌────────────────────────────────┬──────────────────────────┬──────────────────────────┬────────────────────────┐
│ Experimental Decision Gate     │ Metric Criterion         │ Outcome: GO Target       │ Outcome: NO-GO Redline │
├────────────────────────────────┼──────────────────────────┼──────────────────────────┼────────────────────────┤
│ Gate 1: Exploration Bootstrap  │ Steps to R > 0           │ Run B <= 40 steps        │ Run B > 100 steps      │
├────────────────────────────────┼──────────────────────────┼──────────────────────────┼────────────────────────┤
│ Gate 2: Advantage Stability    │ Mean intra-group std(R)  │ Run B std(R) >= 0.35     │ Run B std(R) < 0.10    │
├────────────────────────────────┼──────────────────────────┼──────────────────────────┼────────────────────────┤
│ Gate 3: Sample Efficiency AUC  │ Delta Cumulative AUC     │ Run B - Run A >= +75.0   │ Run B - Run A < +20.0  │
├────────────────────────────────┼──────────────────────────┼──────────────────────────┼────────────────────────┤
│ Gate 4: Asymptotic Correctness │ Held-Out Spider Dev TS   │ Run B - Run A >= +5.0 pp │ Run B - Run A < +2.0 pp│
└────────────────────────────────┴──────────────────────────┴──────────────────────────┴────────────────────────┘
```

* **Adjudication Rule**: If Gates 1, 2, and 4 achieve `GO`, the hypothesis is conclusively proven: *Harness-Zero multi-turn distillation is statistically superior to single-turn SFT as a warm-start policy for agentic database RLVR.*

---

## 8. Failure Mode Profiling via Diagnostic Traps

Using the Pillar 4 Diagnostic Trap Profiler, every failed rollout across both runs will be classified into one of five mutually exclusive error categories:

```
                          Diagnostic Trap Taxonomy
                                     │
      ┌────────────────┬─────────────┴───────────────┬────────────────┐
      ▼                ▼                             ▼                ▼
[Hallucinated]   [Dialect Drift]              [Join Fanout]   [Null Boundary]
Column/Table     SQLite vs Postgres           Unconstrained   Improper handling
non-existent     function mismatches          Cartesian mult. of 3-valued logic
```

1. **Hallucinated Column/Table (`trap_schema_hallucination`)**: Query references identifiers absent from the SQLite `sqlite_master` catalog.
2. **Dialect Drift (`trap_dialect_drift`)**: Query invokes non-SQLite keywords or constructs (e.g., `ILIKE`, `DATE_TRUNC`, `CONCAT`).
3. **Join Fanout / Unconstrained Cartesian (`trap_join_fanout`)**: Query omits `ON` or `USING` clauses or joins non-keyed tables, causing combinatorial cardinality explosions.
4. **Three-Valued Logic / Null Boundary (`trap_null_boundary`)**: Query uses `= NULL` instead of `IS NULL`, or misplaces filters across `LEFT JOIN` boundaries.
5. **Aggregation without Grouping (`trap_agg_no_group`)**: Query combines scalar projections with aggregate functions (`SELECT name, MAX(salary)`) without explicit `GROUP BY`.

### Anticipated Trap Distribution Analysis
We anticipate that Run A will remain trapped in `trap_schema_hallucination` and `trap_dialect_drift` for the first 100 steps because single-turn SFT never taught the model how to inspect local dialect catalogs. Conversely, Run B will eliminate schema hallucinations within 25 steps, allowing GRPO gradients to optimize subtle relational join logic and boundary filters.

---

## 9. Post-Hoc Extension: Compound Architecture Verification (Pillar 3 Integration)

While test-time compound verification is ablated during the core training experiment, a natural follow-up investigation evaluates how both trained policies interact with the **Pillar 3 Continuous Logit Verifier**:

```
                       Post-Hoc Compound Architecture
                       
   Trained Policy (Run A or Run B)
                 │
                 ▼
   Sample G=5 Candidate Queries (Temperature = 0.7)
                 │
                 ▼
   [Pillar 3: LLM-as-a-Verifier Tournament]
   • Continuous logit scoring over normalized anchor tokens
   • Probabilistic Pivot elimination (70% comparison reduction)
                 │
                 ▼
   Top-Ranked Final Query Output
```

### Research Question for Extension:
*Does test-time verification compensate for a poorly initialized generator (Run A), or does verification compound exclusively when paired with a procedurally grounded generator (Run B)?*

* **Theoretical Prediction**: As observed in Pillar 3, a verifier cannot fix candidates if all 5 rollouts hallucinate non-existent columns. We predict the Compound Architecture will yield a modest $+2.5\text{ pp}$ boost over Run A (reaching ~56.0% TS), but an additive $+3.5\text{ pp}$ boost over Run B (reaching **~64.7% TS**), definitively demonstrating that **compound verification requires an exploratory, procedurally grounded generator**.

---

## 10. Summary & Checklist for Experiment Execution

### Phase 1: Environment & Pre-requisites
- [ ] Verify isolated virtual environment (`export PYTHONNOUSERSITE=1`).
- [ ] Confirm Weights & Biases API credentials (`WANDB_PROJECT=ask-the-database`).
- [ ] Confirm extracted SQLite database catalog paths in `data/databases/`.
- [ ] Execute `make test` to ensure all 171 core repository unit tests pass.

### Phase 2: Distillation & Warm-Start Training
- [ ] Collect 500 stratified teacher-reviewed trajectories (`make collect-distillation`).
- [ ] Train Condition 2 Baseline SFT (`make train-sft`).
- [ ] Train Condition 4 Harness-Zero SFT (`make train-harness-zero`).
- [ ] Validate both checkpoints with Zero-Budget Micro-Harness (`make evaluate-zero-budget-mve`).

### Phase 3: Downstream Multi-Turn GRPO RLVR
- [ ] Execute Run A: 250 steps starting from Condition 2 (`make train-grpo`).
- [ ] Execute Run B: 250 steps starting from Condition 4 (`make train-grpo`).
- [ ] Monitor live telemetry in W&B (`mean_reward`, `std_reward`, `advantage_mean`).

### Phase 4: Final Evaluation & Publication
- [ ] Run full 1,034-sample evaluation on Spider Dev and Spider-Realistic (`make evaluate`).
- [ ] Generate comparative markdown efficiency report (`make compare-rlvr`).
- [ ] Check decision gates against adjudication criteria (Section 7).
- [ ] Package final models and update Hugging Face cards (`make generate-hf-cards`).

---

## References

1. **Ye, H., et al. (2026)**. *Harness-Zero: Distilling Agentic Scaffolds into Model Parameters via Procedural Guidance*. arXiv:2609.24974.
2. **DeepSeek-AI. (2025)**. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
3. **Nova, A., et al. (2025)**. *SkyRL-SQL: Multi-Turn Relational Reinforcement Learning with Verifiable Rewards*. arXiv preprint.
4. **Yu, T., Zhang, R., Yang, K., et al. (2018)**. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. In *EMNLP*.
5. **Zhong, R., Choi, E., et al. (2020)**. *Semantic Evaluation for Text-to-SQL with Test Suite Based Execution*. In *EMNLP*.
6. **Shao, Z., Wang, P., et al. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
7. **Contreras Campana, C. (2026)**. *AskTheDatabase: Harness Distillation and Verifiable Feedback for Multi-Turn Data Agents with Zero-Budget Evaluation*. Technical Report.
