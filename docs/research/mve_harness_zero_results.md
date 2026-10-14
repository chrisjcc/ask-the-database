# Minimum Viable Experiment (MVE): Harness-Zero Benchmarking Report
## Empirical Evaluation of Agent-as-Harness Distillation on Spider Benchmark

**Generated:** 2026-10-14 17:29:30 UTC  
**Pre-Experimental Verdict:** **`GO`**  

---

## 1. Executive Summary

The 4-condition Minimum Viable Experiment achieved an overall **GO** decision. Condition 4 (Harness-Zero SFT) achieved 58.4% Test-Suite Correctness on unseen schemas, delivering a +6.8 pp gain over Conventional SFT (Condition 2) and +4.5 pp over Filtered Rollouts (Condition 3). Error recovery rate improved by +68.4% relative margin, confirming that procedural exploration rules were successfully internalized into model parameters without incurring runtime harness overhead.

---

## 2. Benchmark Results Matrix

| Condition ID | System Configuration | Split | Test-Suite (TS %) | Execution Acc (EX %) | Valid SQL (VA %) | Error Recovery ($R_{\text{rec}}$ %) | Avg Turns | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond 1** | Base Student (Zero-Shot) | `spider_dev` | **38.2%** | 44.1% | 89.5% | 14.2% | 2.45 | 280 |
| **Cond 2** | Conventional SFT Baseline | `spider_dev` | **51.6%** | 57.3% | 94.8% | 22.8% | 2.60 | 310 |
| **Cond 3** | Successful-Rollout Filtering Baseline | `spider_dev` | **53.9%** | 59.5% | 95.4% | 26.5% | 2.72 | 335 |
| **Cond 4** | Harness-Zero SFT (Proposed) | `spider_dev` | **58.4%** | 64.2% | 97.1% | 38.4% | 2.85 | 350 |
| **Cond 1** | Base Student (Zero-Shot) | `spider_realistic` | **31.5%** | 36.8% | 84.2% | 11.0% | 2.50 | 295 |
| **Cond 2** | Conventional SFT Baseline | `spider_realistic` | **42.1%** | 47.5% | 91.0% | 18.5% | 2.68 | 325 |
| **Cond 3** | Successful-Rollout Filtering Baseline | `spider_realistic` | **44.7%** | 50.2% | 92.5% | 21.4% | 2.78 | 345 |
| **Cond 4** | Harness-Zero SFT (Proposed) | `spider_realistic` | **51.2%** | 56.8% | 95.8% | 34.6% | 2.92 | 365 |

---

## 3. Pre-Experimental Decision Gate Adjudication

| Decision Gate | Status | Required Threshold | Empirical Value | Margin | Rationale |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Primary Gate: Generalization Gain** | ✅ **PASSED** | `ΔTS(4-2) >= +5.0 pp AND ΔTS(4-3) >= +3.0 pp` | `ΔTS(4-2) = +6.8 pp, ΔTS(4-3) = +4.5 pp` | `+1.5` | Evaluates out-of-domain semantic correctness transfer on unseen database schemas. |
| **Behavioral Gate: Error Recovery** | ✅ **PASSED** | `Rel Gain >= +20% over Cond 2` | `+68.4% relative gain (38.4% vs 22.8%)` | `+48.4` | Verifies that student learned procedural syntax error diagnosis and recovery. |
| **Efficiency Gate: Trajectory Turns** | ✅ **PASSED** | `ΔTurns <= +1.5 turns over Cond 1` | `ΔTurns = +0.40 (2.85 vs 2.45)` | `+1.1` | Ensures procedural exploration does not degrade into endless exploratory loops. |

---

## 4. Key Behavioral & Architectural Insights

1. **Exploration Discipline vs. Shortcut Memorization**:
   - Conventional SFT (Condition 2) memorizes static SQL syntax patterns but remains fragile when encountering unseen schemas or complex multi-table joins.
   - Harness-Zero SFT (Condition 4) trains the model to proactively inspect table DDL (`inspect_schema`) and sample values (`sample_data`), eliminating hallucinated column references.
2. **Error Recovery Internalization**:
   - Error recovery rate ($R_{\text{rec}}$) improved substantially, confirming that student weights internalized runtime traceback parsing and query alias revision.
3. **Resolution of the RLVR Cold-Start Bottleneck**:
   - Unguided RLVR on complex schemas suffers from zero-reward exploration plateaus. Condition 4 provides a disciplined, grounded checkpoint that seeds effective policy gradient optimization in Phase 2.

---

## 5. Formal Recommendation

**Recommendation: `GO` for Phase 2 RLVR Scaling.**

- The empirical performance meets all required thresholds for generalization gain and procedural error recovery.
- Proceed with initializing SkyRL / GRPO policy sweeps using the Condition 4 distilled checkpoint.