# OPDVR Research & Feasibility Track

This directory contains the independent research documentation, technical feasibility study, comparative assessment, and engineering implementation specifications for **OPDVR (On-Policy Distillation with Verifiable Reward)** ([arXiv:2608.24696](https://arxiv.org/abs/2608.24696), August 2026) within the `AskTheDatabase` ecosystem.

---

## Documents in this Track

1. 📄 **[OPDVR Feasibility Study](feasibility_study.md)**:
   - Comprehensive technical breakdown of the OPDVR algorithm, its underlying motivation, and its mathematical formulation.
   - Analysis of the post-training dilemma: sparse outcome rewards in RLVR (credit assignment problem) vs. purely distributional objectives in On-Policy Distillation (OPD).
   - Detailed answers to the 7 core feasibility questions: problem, applicability, added value, complementarity, implementation feasibility, PoC design, and failure modes.
   - Deconstruction of the parameter-free sign-aligning ReLU gate:
     $$R_{\text{OPDVR}}(o_t) = R(o) \cdot \mathrm{ReLU}\left( R(o) \cdot \log \frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})} \right)$$
   - Formulation of **Group Relative Policy Distillation (GRPD)** combining group-relative advantages $\hat{A}_i$ with token-level teacher log-ratios.
   - Strategic verdict: **ADOPT (High Priority)**.

2. 📊 **[Comparative Analysis](comparison.md)**:
   - 10-dimensional architectural comparison matrix evaluating **OPDVR** alongside **Harness-Zero**, **LLM-as-a-Verifier**, **HarnessSQL**, and **SQL-Zero**.
   - In-depth pairwise comparative breakdowns:
     - *Harness-Zero vs. OPDVR*: Sequential post-training stages (procedural SFT warm-start $\to$ online policy distillation).
     - *LLM-as-a-Verifier vs. OPDVR*: Train-time policy optimization vs. test-time tournament reranking (Theorem 4.1 candidate pool quality multiplier).
     - *HarnessSQL vs. OPDVR*: Replacing heavyweight teacher rejection sampling and Slime/Megatron distributed RL with lightweight on-policy token distillation.
     - *SQL-Zero vs. OPDVR*: Unifying synthetic task proposal with verifiable policy distillation.
   - Concrete walkthrough of Type I (valid novel syntax) and Type II (hallucinated column) conflict resolution in SQL generation.
   - Synthesis of the **AskTheDatabase Unified Trinity**.

3. 🏗️ **[System Architecture & Mathematical Formulation](architecture.md)**:
   - Formal POMDP formulation for multi-turn tool interaction (`inspect_schema`, `sample_data`, `execute_sql`, `final_answer`).
   - Strict action-token loss masking specification (ignoring SQLite sandbox observation outputs).
   - The **Privileged Schema Teacher** paradigm: providing complete DDL schema context to the teacher while the student explores under zero-knowledge tool constraints.
   - Mathematical proofs: Proposition C.1 (directional alignment $\langle \Delta_{\text{OPDVR}}, \Delta_{\text{RLVR}} \rangle \ge 0$), Proposition C.2 (orthogonal conflict decomposition), and Theorem A.2 (monotonic joint improvement).
   - End-to-end Mermaid sequence and dataflow diagrams.
   - Modular codebase mapping and standalone PyTorch implementation of `GRPDLoss`.

4. 🧪 **[Experimental Plan & Proof of Concept](experimental_plan.md)**:
   - Concrete proof-of-concept specifications to empirically isolate and validate OPDVR.
   - Four formal scientific hypotheses (H1: Credit Assignment, H2: Exploration Plateau Elimination, H3: Syntactic Diversity, H4: Privileged Schema Distillation) with quantitative success criteria.
   - 7-condition experimental matrix (C0: SFT Baseline, C1: Pure RLVR, C2: Vanilla OPD+GRPO, C3: Binary OPDVR, C4: Full GRPD, C5: Inverse-Gated Control, C6: Privileged GRPD).
   - Diagnostic telemetry metrics: tracking Zero-Gated Token Ratio ($r_{\text{zero}}$) within the theoretical $40\% - 50\%$ band, gradient norms, and tool interaction errors.
   - Hyperparameters, compute resource budget (LoRA weight sharing on single GPU), and definition of done.

5. 🛠️ **[Implementation Roadmap & Engineering Plan](implementation_plan.md)**:
   - 4-phase Work Breakdown Structure (WBS):
     - *Phase 1*: Core Mathematical Engine & Invariant Verification (`losses.py`, unit tests).
     - *Phase 2*: Privileged Teacher & Action Token Masking (`privileged_teacher.py`, `trl_grpo_adapter.py`).
     - *Phase 3*: RLVR Trainer Integration & DVC Pipeline (`rlvr.yaml`, `train_rlvr.py`).
     - *Phase 4*: Observability, Diagnostics & Benchmark Validation (Weave/WandB streaming, micro-harness smoke tests).
   - Detailed component specifications, interfaces, and risk fallback matrix.
   - VRAM allocation profile demonstrating single-GPU (A100 / RTX 4090) feasibility.

---

## High-Level Verdict Summary

| Framework | Primary Regime | Supervisory Signal | Multi-Turn Tool Support | Credit Assignment | Exploration Stall Protection | Empirical Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **Harness-Zero** | SFT Cold-Start | In-flight procedural reviewer ($\mathcal{K}_{\text{SQL}}$) | ✅ Native | Step-level tool rules | ✅ Provides valid trajectory priors | SOTA SFT baseline (zero gold SQL leakage) |
| **HarnessSQL** | SFT + RLVR | Teacher Rejection Sampling + Binary Execution | ✅ Native | Coarse trajectory (16 turns) | ⚠️ Partial (DAPO dynamic sampling) | SOTA on Spider 2.0-SQLite; high compute & teacher cost |
| **SQL-Zero** | Data Synthesis | Self-play proposer–solver execution match | ❌ Static DDL | Coarse trajectory | ⚠️ Partial (Difficulty gating $d(k)$) | +6.6 to +7.3 on BIRD; collapses on 7B transfer |
| **LLM-as-a-Verifier** | Test-Time Search | Decomposed continuous logit expectation | ✅ Native | Trajectory candidate level | N/A (Inference reranking) | +8.13 pp test-time boost; Pareto-dominant |
| **OPDVR / GRPD** | Online RLVR | Verifiable Mutation $\mathcal{R}_{\text{TS}}$ $\times$ Teacher Log-Ratio | ✅ Native | **Dense token-level credit** | **Absolute (Active gradient when $\mathrm{std}(R)=0$)** | SOTA on reasoning benchmarks; parameter-free |

**Core Recommendation**: **Adopt OPDVR and GRPD to power the online RLVR post-training phase in AskTheDatabase. Pair an unprivileged multi-turn student agent with a Privileged Schema Teacher (full DDL access) and our 4-tier Test-Suite Mutation Verifier ($\mathcal{R}_{\text{TS}}$). Maintain Harness-Zero for SFT initialization and LLM-as-a-Verifier for test-time tournament search.**
