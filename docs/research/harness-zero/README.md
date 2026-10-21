# Harness-Zero Research & Distillation Track

This directory contains the authoritative research documentation, feasibility analysis, and empirical findings for integrating **Harness-Zero** (Agent-as-Harness Distillation) into the `AskTheDatabase` post-training ecosystem.

---

## Documents in this Track

1. 📄 **[Harness-Zero Feasibility Study](feasibility_study.md)**:
   - Comprehensive theoretical and practical analysis of the Harness-Zero paradigm (*Ye et al., 2026*).
   - Differentiates answer distillation from procedural harness distillation.
   - Evaluates action spaces, token costs, architectural trade-offs, and synergy with multi-turn RLVR.
   - Formulates the pre-experimental 4-condition design and success criteria.

2. 📊 **[Harness-Zero MVE Final Research Report](mve_final_report.md)**:
   - Single consolidated empirical report for the 4-condition Minimum Viable Experiment (MVE).
   - Ingests empirical benchmarks across $N=500$ unseen schemas on Spider Dev and Spider Realistic.
   - Documents the four deterministic procedural rules of reference harness $\mathcal{K}_{\text{SQL}}$ (`RequireSchemaInspection`, `RequireRowSampling`, `EnforceDryRunBeforeFinish`, `DiagnosticRecovery`).
   - Adjudicates all 3 pre-registered decision gates with empirical margins (**Verdict: `GO`**).
   - Deep-dive mathematical and mechanistic analysis of GRPO convergence speed (zero-variance plateau bypass) and asymptotic correctness (prevention of policy collapse via procedural inductive bias).
   - Provides concrete operational commands to reproduce all results.

---

## Key Experimental Results Summary

| Metric | Condition 1: Base Student | Condition 2: Conventional SFT | Condition 3: Filtered Rollouts | Condition 4: Harness-Zero SFT | Gain (C4 vs C2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test-Suite Accuracy (`spider_dev`)** | 38.2% | 51.6% | 53.9% | **58.4%** | **+6.8 pp** |
| **Test-Suite Accuracy (`spider_realistic`)** | 31.5% | 42.1% | 44.7% | **51.2%** | **+9.1 pp** |
| **Execution Accuracy (`spider_dev`)** | 44.1% | 57.3% | 59.5% | **64.2%** | **+6.9 pp** |
| **Multi-Turn Error Recovery Rate** | 14.2% | 22.8% | 26.5% | **38.4%** | **+68.4% rel** |
| **Schema Pre-Inspection Rate** | 31.0% | 46.2% | 52.8% | **84.6%** | **+38.4 pp** |
| **Adjudication Verdict** | — | — | — | **`GO`** | All gates cleared |

For the operational execution manual for training, evaluation, and distributed sweeps, see [`docs/pipeline_execution_guide.md`](../../pipeline_execution_guide.md).
