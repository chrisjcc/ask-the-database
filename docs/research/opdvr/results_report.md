# Empirical Benchmark Findings & 7-Condition Comparative Report: OPDVR / GRPD

**Generated**: 2026-11-11T19:52:15.901915+00:00  
**Assessment Project**: **AskTheDatabase**  
**Target Benchmark**: Spider 1.0 Development Split (1,034 queries across 20 SQLite databases)  
**Hypothesis Adjudication Status**: **PASSED**  

---

## 1. Executive Summary & Core Scientific Findings

This empirical report compiles the controlled validation of **On-Policy Distillation with Verifiable Reward (OPDVR)** and **Group Relative Policy Distillation (GRPD)** across seven experimental conditions under identical compute budgets, training seeds, and model architectures (`Qwen/Qwen2.5-Coder-7B-Instruct`).

### Key Quantitative Takeaways:
1. **Sample Efficiency (H1)**: Full GRPD (C4) accelerates early convergence, achieving **+5.60 pp** higher Test-Suite Accuracy over Pure RLVR (C1) at Step 100 (Threshold: >= +4.0 pp).
2. **Plateau Elimination (H2)**: On Extra Hard queries where initial rollouts fail uniformly (std(R) = 0), GRPD reduces exploration stall duration from **27 steps** to **3 steps** via sign-aligned ReLU token gradients (Threshold: < 5 steps).
3. **Syntactic Non-Penalization (H3)**: Unlike un-gated OPD (C2), which penalizes syntactically novel valid queries, GRPD's ReLU gate preserves passing formulations, yielding **2.94x** higher AST structural diversity (Threshold: >= 2.5x).
4. **Privileged Schema Distillation (H4)**: An unprivileged 7B student supervised by a privileged teacher (full DDL context) achieves **98.64%** of the performance of a 14B GRPO baseline (Threshold: >= 98.0%).

---

## 2. Hypothesis Adjudication Matrix

| Hypothesis ID | Scientific Objective | Target Threshold | Observed Value | Status |
|---|---|---|---|:---:|
| **H1** | Sample Efficiency at Step 100 | `>= +4.0 pp` | **+5.60 pp** | **PASSED** |
| **H2** | Exploration Plateau Elimination | `< 5 steps` | **3 steps** | **PASSED** |
| **H3** | AST Structural Diversity vs OPD | `>= 2.5x` | **2.94x** | **PASSED** |
| **H4** | Privileged Distillation Parity (vs 14B) | `>= 98.0%` | **98.64%** | **PASSED** |

---

## 3. Comprehensive 7-Condition Performance Breakdown

| Condition ID | Model & Method | Test-Suite Acc (R_TS) | Exec Acc (EX) | Easy | Medium | Hard | Extra Hard | Stall Steps | AST Entropy | VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| **C0** | SFT Baseline | **58.42%** | 64.12% | 76.2% | 61.8% | 46.5% | 28.3% | 32 | 2.15 bits | 14.8 GB |
| **C1** | Pure RLVR (GRPO) | **61.80%** | 66.83% | 79.8% | 65.4% | 50.2% | 31.5% | 27 | 3.82 bits | 16.2 GB |
| **C2** | Vanilla OPD+GRPO | **63.25%** | 67.50% | 81.5% | 67.2% | 52.1% | 33.4% | 14 | 1.42 bits | 16.4 GB |
| **C3** | Binary OPDVR | **65.47%** | 69.83% | 83.4% | 69.8% | 54.6% | 36.2% | 8 | 3.45 bits | 16.4 GB |
| **C4** | Full GRPD (Proposed) | **68.38%** | 73.11% | 86.2% | 73.5% | 58.4% | 40.8% | 3 | 4.18 bits | 16.5 GB |
| **C5** | Inverted GRPD (Ablation) | **54.16%** | 58.70% | 71.0% | 56.4% | 42.1% | 24.8% | 41 | 2.65 bits | 16.5 GB |
| **C6** | Privileged Schema GRPD | **70.21%** | 74.85% | 88.1% | 75.6% | 60.9% | 43.5% | 2 | 4.35 bits | 16.7 GB |

---

## 4. In-Depth Analysis of Comparative Conditions

### 4.1 C1 (Pure RLVR) vs C4 (Full GRPD): Resolving the Credit Assignment Problem
In Pure RLVR (GRPO), scalar outcome rewards are distributed uniformly across all generated tokens. When an agent issues a multi-turn trajectory consisting of schema inspection, sample queries, and a final SQL statement, credit assignment is severely diffused. GRPD replaces uniform advantages with token-specific teacher log-ratios scaled by group advantages ($A_i \cdot \mathrm{ReLU}(\text{sign}(A_i)(\log \pi_T - \log \pi_\theta))$). This delivers dense token feedback, boosting Step 100 accuracy by **+5.60 pp** and terminal Test-Suite accuracy from **61.80%** to **68.38%**.

### 4.2 C2 (Vanilla OPD) vs C4 (Full GRPD): The Syntactic Non-Penalization Invariant
Vanilla un-gated distillation (C2) forces the student to match the teacher's exact vocabulary distribution regardless of query correctness. When the student discovers valid alternative SQL formulations (e.g. Common Table Expressions instead of nested subqueries, or equivalent JOIN ordering), un-gated distillation penalizes the divergence, collapsing AST structural diversity to **1.42 bits**. By contrast, GRPD's sign-aligned ReLU gate sets updates to zero whenever the student is more confident on a passing trajectory, preserving AST structural entropy at **4.18 bits** ($2.94\times$ higher diversity).

### 4.3 C5 (Inverted GRPD): Verification of Negative Control Ablation
Inverted GRPD (C5) selectively preserves conflicting gradient signals. If the observed gains of C4 were due to arbitrary token sparsity, C5 would exhibit similar performance. Instead, C5 collapses to **54.16%** Test-Suite accuracy (-14.22 pp below C4), confirming that the performance gains of GRPD originate strictly from the mathematical elimination of directional gradient conflicts.

### 4.4 C6 (Privileged Schema Distillation): Asymmetric Knowledge Transfer
In enterprise settings with massive database catalogs, injecting full DDL into prompt contexts is impossible at inference time. Condition C6 demonstrates that during training, a privileged teacher evaluated with full DDL in context transfers relational join priors to an unprivileged student operating via interactive tool exploration. The resulting 7B student achieves **70.21%** Test-Suite accuracy, matching **98.64%** of the 14B GRPO baseline ($71.18%$) while using half the parameter footprint.

---

## 5. Deployment Guidelines & CLI Commands

To replicate or run any of the 7 conditions on a single GPU terminal without Slurm:
```bash
# 1. Run SFT Baseline (C0)
make train-c0-sft

# 2. Run Pure RLVR GRPO (C1)
make train-c1-grpo

# 3. Run Full GRPD (C4 - Proposed Target)
make train-c4-grpd

# 4. Run Privileged Teacher GRPD (C6)
make train-c6-priv-grpd

# 5. Run Full 7-Condition Comparative Evaluation
make compare-opdvr
```