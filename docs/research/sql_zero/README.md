# SQL-Zero Research & Feasibility Track

This directory contains the independent research documentation, feasibility analysis, and comparative assessment for **SQL-Zero: Self-Evolving Text-to-SQL** ([arXiv:2609.04697](https://arxiv.org/abs/2609.04697), September 2026) within the `AskTheDatabase` ecosystem.

---

## Documents in this Track

1. 📄 **[SQL-Zero Feasibility Study](feasibility_study_sql_zero.md)**:
   - Comprehensive theoretical and empirical analysis of SQL-Zero's zero-annotation proposer–solver self-play spiral.
   - Deconstruction of the Challenger reward mechanics: format integrity gate ($f$), difficulty gating function ($d(k)$ peaking at $k=1$ of $n=5$), and masked-SQL-template repetition penalty ($\rho = \lambda |c| / |B|$).
   - In-depth investigation of empirical scaling dynamics and systemic failure modes:
     - The **7B Transfer Erosion** phenomenon (in-domain BIRD gains accompanied by $-2.5\,\text{pp}$ collapse on Spider test transfer).
     - **Policy Entropy Decay** ($0.34 \to 0.06$) and token length shrinkage under un-reset alternating GRPO.
     - **Diversity Exhaustion** and candidate yield collapse under fixed proposal budgets.
     - The **Reasoning Trace Ablation** showing $\langle\text{think}\rangle$ tokens provide no net accuracy gain at $2.4\times$ token cost.
   - Comparative 8-dimensional architectural matrix contrasting **SQL-Zero**, **HarnessSQL**, and **AskTheDatabase**.
   - Strategic blueprint for adopting SQL-Zero's Challenger mechanics to synthesize training tasks over **unlabeled enterprise schemas** (Spider 2.0-DBT SQLite databases) paired with AskTheDatabase's **Test-Suite Mutation Equivalence** and **Harness-Zero Procedural Reviewer**.
   - Phased 4-stage implementation roadmap.

---

## High-Level Verdict Summary

| Framework | Core Alignment Paradigm | Enterprise Schemas (>20 tables) | Test-Time Verification | Spurious Match Protection | Empirical Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **SQL-Zero** | Zero-Annotation Proposer–Solver Self-Play | ❌ Incompatible (Monolithic DDL) | ❌ None (Greedy single-pass) | ❌ Low (Single DB state) | +6.6 to +7.3 on BIRD; plateaus after It 1; collapses on 7B transfer |
| **HarnessSQL** | Teacher Rejection SFT + Distributed RL | ✅ Supported (DSH-SQL POMDP) | ❌ None (Greedy single-pass) | ❌ Low (Single DB state) | SOTA on Spider 2.0-SQLite; high compute & teacher cost |
| **AskTheDatabase** | Harness-Zero Procedural Distillation + RLVR | ✅ Supported (Tool Registry POMDP) | ✅ Continuous Logit Tournaments | ✅ Absolute (Mutation Equivalence $\mathcal{R}_{\text{TS}}$) | SOTA TS Frontier (61.8%); zero-budget micro-harness; +8.13 pp test-time boost |

**Recommendation**: **Selectively adapt SQL-Zero's Adversarial Challenger difficulty gating and template deduplication to autonomously generate training corpora over unlabeled enterprise databases; preserve AskTheDatabase's procedural non-leaking distillation, hierarchical mutation-verified rewards, and compound trajectory verifiers.**
