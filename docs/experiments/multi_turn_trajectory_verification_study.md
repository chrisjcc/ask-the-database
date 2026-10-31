# Experimental Study: Multi-Turn Trajectory Verification for Autonomous Relational Agents

**Status**: In Progress / Architectural Implementation  
**Date**: October 2026  
**Related PR**: Addressing Issue [#82](https://github.com/chrisjcc/ask-the-database/issues/82)  
**Parent Framework**: [AskTheDatabase White Paper Draft](../../papers/draft/main.pdf) & [Dual-Regime Evaluation Study](sft_warmstart_multi_turn_rlvr_study.md)

---

## 1. Executive Summary & Scientific Motivation

### 1.1 The Starting Point: Dual-Regime Evaluation Findings
In our empirical investigation of post-trained relational models across the held-out Spider Dev split ($N=1{,}034$ samples), we established the **Dual-Regime Evaluation Framework** (documented in Section 5.5 of the white paper and PR #81):

| Evaluation Regime | Conventional SFT Baseline | Procedurally Grounded Agent (Harness-Zero GRPO) | Empirical Delta ($\Delta$) | Core Operational Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Autonomous Zero-Knowledge Multi-Turn** *(Primary Objective: Schema Withheld, Max 5 Turns)* | 1.26% TS | **13.54% TS** | **+12.28 pp (10.7×)** | **Grounded Agent Dominates**: Baseline undergoes catastrophic exploration collapse ($\operatorname{std}(R) \to 0$). |
| **Static Single-Turn Full-DDL Compound** *(Diagnostic: Best-of-5 Verifier, Full DDL in Prompt)* | **48.84% TS** | 37.72% TS | -11.12 pp | **Baseline Wins**: Baseline is 100% in-distribution; agent pays a "Tool-Calling Format Tax". |

### 1.2 The Discovered Trade-Off: "The Tool-Calling Tax"
When evaluated under the static single-turn verification protocol, the multi-turn policy scored lower (37.72% vs. 48.84% TS) not because it forgot relational algebra, but because of its **internalized multi-turn prior**:
1. Over 250 steps of multi-turn GRPO, the model was heavily reinforced to **refuse to guess blindly without inspecting database catalogs**.
2. When presented with a prompt containing full DDL, the agent recognized the schema, successfully generating valid SQL on **69.34%** of rollouts.
3. However, at sampling temperature $T=0.7$, the model's exploratory prior caused it to emit `<tool name="inspect_schema">` or `<tool name="sample_data">` calls in roughly **~31%** of candidate rollouts.
4. Because the static `VerifierSelector` was designed as a **single-query ranker** operating over raw SQL blocks (`clean_candidate_sql`), it discarded these exploratory tool calls as fatal syntax errors (`OperationalError: near "<": syntax error`).

### 1.3 The Core Scientific Lesson: Verifiers Must Match Agent Modalities
The current verifier evaluated static SQL strings at the end of a single prompt. This works effectively for static one-shot parsers, but forces the autonomous agent into an out-of-distribution format that penalizes its learned exploratory reflexes.

**Scientific Mandate for Next Iteration**:  
To unlock test-time verification for autonomous agents operating in zero-knowledge environments, the verification framework must transition from **Single-Query Candidate Verification** to **Multi-Turn Trajectory Verification**.

---

## 2. Theoretical Architecture: Process-Level Trajectory Verification

### 2.1 From Outcome Verifier to Trajectory Process Verifier
Let an interactive relational episode be modeled as a Markov Decision Process (MDP) rollout:
$$\tau = \left( s_0, a_0, o_0, s_1, a_1, o_1, \dots, s_T, a_T \right)$$
where $s_0 = \langle \text{db\_id}, \text{question} \rangle$, $a_t \in \mathcal{A}$ are tool actions (`inspect_schema`, `sample_data`, `execute_sql`, `final_answer`), and $o_t$ are sandbox observations.

In **Single-Query Verification**, the selector evaluates only the terminal output:
$$\mathcal{V}_{\text{query}}(Q) = \mathbb{E}[S \mid \text{DDL}, \text{Question}, Q]$$

In **Multi-Turn Trajectory Verification**, the selector evaluates the full procedural trajectory:
$$\mathcal{V}_{\text{traj}}(\tau) = \mathbb{E}\left[ S \mid \tau \right] = \sum_{c \in \mathcal{C}} w_c \cdot \mathbb{E}[S_c \mid \tau]$$
decomposed across four orthogonal trajectory criteria $\mathcal{C}$:

1. **Catalog Exploration Conformance ($c_{\text{catalog}}$)**:
   * Did the agent inspect the relevant tables and foreign keys prior to issuing queries?
   * Penalizes queries referencing uninspected tables in zero-knowledge mode.
2. **Sandbox Trial Validation ($c_{\text{trial}}$)**:
   * Did the agent execute a dry-run trial query (`execute_sql`) before committing to `final_answer`?
   * Awards bonus credit for trial executions that validated non-empty result sets.
3. **Autonomous Error Recovery ($c_{\text{recovery}}$)**:
   * If an intermediate action triggered an operational SQLite exception (e.g. `no such column`), did a subsequent turn diagnose the error and correct the SQL predicate?
4. **Terminal Intent Alignment ($c_{\text{intent}}$)**:
   * Does the final submitted query $Q_{\text{final}}$ faithfully answer the natural language question $x$, respecting grouping, aggregation, and ordering semantics?

```
                        Multi-Turn Trajectory Verification
                        
                    Zero-Knowledge User Question + Database ID
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
       Trajectory 1               Trajectory 2               Trajectory 3
   Turn 1: inspect_schema     Turn 1: inspect_schema     Turn 1: inspect_schema
   Turn 2: sample_data        Turn 2: execute_sql (Err)  Turn 2: final_answer (Blind)
   Turn 3: execute_sql (OK)   Turn 3: fix query (OK)                │
   Turn 4: final_answer       Turn 4: final_answer                  │
             │                          │                           │
             └──────────────────────────┼───────────────────────────┘
                                        ▼
                         Trajectory Verifier & Tournament
             ┌─────────────────────────────────────────────────────┐
             │ - Evaluates catalog exploration completeness        │
             │ - Evaluates trial query validation & error recovery │
             │ - Continuous logit expectation over criteria        │
             │ - Probabilistic Pivot Tournament selection          │
             └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              Selected Winning SQL
```

---

## 3. Mathematical Formulation: Candidate Pool Quality in Trajectories

By Theorem 4.1 (*Candidate Pool Quality Bottleneck*), the probability of finding a correct final query across $G$ sampled rollouts is:
$$\text{Pass@}G = 1 - (1 - p)^G$$

* In single-turn mode without DDL, the ungrounded baseline has $p \approx 0.0126 \implies \text{Pass@5} \le 1.8\%$.
* In multi-turn interactive execution, the procedurally grounded policy (**Harness-Zero GRPO**) has a greedy pass rate of $p = 0.1354$.
* When sampling $G=5$ diverse stochastic rollouts ($T=0.7$), the probability that **at least one trajectory discovers the correct schema join path and reaches the gold tabular result** is:
$$\text{Pass@5}_{\text{trajectory}} = 1 - (1 - 0.1354)^5 = 1 - (0.8646)^5 = 1 - 0.4746 = \mathbf{52.54\%}$$

Accounting for SQLite sandbox pre-filtering (requiring $Q_{\text{final}}$ to execute cleanly without errors), the empirical candidate pool contains viable candidates in **$45\text{--}50\%$** of all Spider Dev episodes. 

The Trajectory Verifier is therefore given a high-density candidate pool to rank, enabling it to arbitrate between alternative relational execution plans without paying the "Tool-Calling Tax".

---

## 4. Software Architecture & Implementation Components

### 4.1 `TrajectoryCandidateSampler`
Located in `src/ask_the_database/inference/trajectory_verifier.py`:
* Wraps `NL2SQLPipeline`.
* For a given `(question, db_path, db_id)`, samples $G$ independent multi-turn trajectories in parallel/sequential batches at sampling temperature $T=0.7$.
* Executes each trajectory inside an isolated SQLite sandbox environment across up to $T=5$ turns.
* Extracts intermediate tool calls, observations, error recovery flags, and terminal SQL answers into structured `TrajectoryCandidate` instances.
* Executes the final SQL against the database sandbox; if all $G$ trajectories fail execution, preserves candidate 0 as a safe fallback.

### 4.2 `TrajectoryVerifierSelector`
Located in `src/ask_the_database/inference/trajectory_verifier.py`:
* Evaluates each `TrajectoryCandidate` across the four decomposed trajectory criteria using continuous logit expectation.
* Incorporates deterministic heuristic fallbacks when verifier LLM clients are operating offline.
* Implements the **Probabilistic Pivot Tournament**:
  * Selects $K$ pivot trajectories ($K \ll G$).
  * Conducts pairwise trajectory comparisons in $\mathcal{O}(K \cdot G)$ complexity.
  * Selects the highest-ranked candidate trajectory.
* Computes execution-based **Majority Voting** across the result sets of executable trajectories as an orthogonal consensus baseline.

### 4.3 Benchmark Integration (`scripts/evaluate_benchmark.py`)
Adds support for `--evaluation.verification_mode=trajectory` (or `evaluation.use_trajectory_verifier=true`):
* Allows running held-out Spider Dev evaluation using native multi-turn trajectory verification.
* Produces detailed telemetry:
  * `test_suite_accuracy_ts`
  * `execution_accuracy_ex`
  * `valid_execution_rate_va`
  * `recovery_rate_mrr`
  * `trajectory_diversity_pass_at_g`
  * `verifier_majority_vote_agreement`

---

## 5. Experimental Protocol & Anticipated Scorecard

### 5.1 Evaluated Configurations on Held-Out Spider Dev ($N=1{,}034$)
1. **Multi-Turn Greedy Baseline (Status Quo)**:
   * Model: `models/rlvr/harness_zero_grpo`
   * Single greedy rollout ($G=1$, $T=0.0$), zero-knowledge prompt.
   * Baseline TS: **13.54%**, EX: **14.22%**, VA: **19.83%**, Recovery: **3.04%**.
2. **Static Single-Turn Verifier (Diagnostic Cross-Regime Test)**:
   * Model: `models/rlvr/harness_zero_grpo`
   * 5 single-turn SQL candidates with full DDL in prompt ($G=5$, $T=0.7$).
   * Result: **37.72% TS**, EX: **41.59%**, VA: **69.34%** (penalized by tool-calling tax).
3. **Multi-Turn Trajectory Verifier (Proposed Next-Generation Architecture)**:
   * Model: `models/rlvr/harness_zero_grpo`
   * 5 complete multi-turn trajectories ($G=5$, $T=0.7$), zero-knowledge prompt.
   * The verifier ranks full trajectories based on catalog exploration, trial execution, and semantic query alignment.

### 5.2 Anticipated Experimental Hypotheses
* **Hypothesis 1 (Format Tax Elimination)**: Because the agent is evaluated in its native zero-knowledge multi-turn environment, its Valid Action Rate will not be penalized by rigid single-turn parsers, maintaining $\text{VA} \ge 90\%$ among completed trajectories.
* **Hypothesis 2 (Pass@5 Pool Expansion)**: Sampling $G=5$ stochastic trajectories will elevate the empirical pool pass rate from $13.54\%$ (greedy) to $\text{Pass@5} \ge 25.0\%$.
* **Hypothesis 3 (Compound Accuracy Boost)**: By ranking multi-turn exploration paths and selecting queries validated by intermediate sandbox dry-runs, Trajectory Verification will achieve an estimated **18.0% to 22.0% Test-Suite Accuracy** on zero-knowledge Spider Dev—a significant step forward for fully autonomous relational agents.
