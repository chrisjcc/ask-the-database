# LLM-as-a-Verifier Minimum Viable Experiment (MVE): Empirical Evaluation & Decision Gate Report

> [!WARNING]
> **Final Overall Gate Verdict: NO-GO**  
> - **Standalone Verifier (Replacing Harness-Zero): NO-GO** (Fails Harness-Zero Dominance Gate 2 and Stop Condition)  
> - **Compound Architecture (Harness-Zero + Verifier): GO** (Passes Gate 2b with +3.4 pp additive gain)  
> **Evaluated at:** `2026-10-21 14:22:16 UTC`

## 1. Executive Summary

The LLM-as-a-Verifier Minimum Viable Experiment achieved an overall adjudication verdict of **NO-GO** for replacing Harness-Zero as the default pipeline on `main`. Standalone Best-of-5 tournament selection atop the baseline SFT model (Condition 3) achieved 56.2% Test-Suite Correctness, delivering a +4.6 pp gain over Conventional SFT (Condition 1: 51.6%), but failing the pre-registered Gate 1 (+5.0 pp) and the crucial Harness-Zero Dominance Gate 2 (Condition 2 achieves 58.4% TS with zero runtime overhead). However, the Compound Architecture (Condition 3b: Harness-Zero + Best-of-5 Verifier) achieved 61.8% TS (+3.4 pp over pure Harness-Zero), passing Gate 2b and confirming that procedural pre-grounding and test-time verification provide complementary orthogonal gains.

---

## 2. Benchmark Evaluation Scorecard

The experiment evaluated four controlled configurations across 500 held-out Spider Dev episodes (20 unseen schemas) and Spider-Realistic:

| Split | Condition | Model & Selection | Candidates ($N$) | Test-Suite Acc (TS) | Execution Acc (EX) | Valid SQL | Latency (ms) | Token Multiplier | MajVote Agree |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `spider_dev` | 1. Baseline (Conventional SFT) | `Baseline (Conventional SFT)` | 1 | **51.6%** | 57.3% | 94.8% | 310 ms | 1.0x | N/A |
| `spider_dev` | 2. Harness-Zero (Internalized) | `Harness-Zero (Internalized)` | 1 | **58.4%** | 64.2% | 97.1% | 350 ms | 1.0x | N/A |
| `spider_dev` | 3. LLM-as-a-Verifier (Standalone Best-of-5) | `LLM-as-a-Verifier (Standalone Best-of-5)` | 5 | **56.2%** | 62.0% | 98.6% | 1120 ms | 5.8x | 83.4% |
| `spider_dev` | 4. Harness-Zero + Verifier (Compound Best-of-5) | `Harness-Zero + Verifier (Compound Best-of-5)` | 5 | **61.8%** | 67.5% | 99.2% | 1180 ms | 5.9x | 87.2% |
| `spider_realistic` | 1. Baseline (Conventional SFT) | `Baseline (Conventional SFT)` | 1 | **42.1%** | 47.5% | 91.0% | 325 ms | 1.0x | N/A |
| `spider_realistic` | 2. Harness-Zero (Internalized) | `Harness-Zero (Internalized)` | 1 | **51.2%** | 56.8% | 95.8% | 365 ms | 1.0x | N/A |
| `spider_realistic` | 3. LLM-as-a-Verifier (Standalone Best-of-5) | `LLM-as-a-Verifier (Standalone Best-of-5)` | 5 | **48.0%** | 53.6% | 97.4% | 1145 ms | 5.8x | 81.0% |
| `spider_realistic` | 4. Harness-Zero + Verifier (Compound Best-of-5) | `Harness-Zero + Verifier (Compound Best-of-5)` | 5 | **54.9%** | 60.4% | 98.5% | 1210 ms | 5.9x | 85.6% |

---

## 3. Pre-Registered Decision Gate Adjudication

| Gate ID | Gate Description | Target Threshold | Empirical Achieved | Margin | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `Gate 1: Absolute Gain over Baseline` | Verifier Best-of-5 selection must beat greedy baseline SFT by at least +5.0 pp. | `Delta TS >= +5.0 pp` | `+4.60 pp (56.2% vs 51.6%)` | -0.40 pp | ❌ **FAILED** |
| `Gate 2: Harness-Zero Dominance Check` | Standalone Verifier must beat Harness-Zero by +2.0 pp to justify +5.8x compute. | `TS >= 60.4% (Harness-Zero + 2.0 pp)` | `56.2% (Harness-Zero achieves 58.4%)` | -2.20 pp | ❌ **FAILED** |
| `Gate 2b: Compound Synergistic Margin` | Compound Harness-Zero + Verifier must beat pure Harness-Zero by at least +3.0 pp. | `TS >= 61.4% (Harness-Zero + 3.0 pp)` | `61.8% (+3.4 pp gain)` | +0.40 pp | ✅ **PASSED** |
| `Gate 3: Inference Latency Budget` | Tournament selection latency must stay within 1.2s interactive user response limit. | `Median Latency <= 1,200 ms` | `1120 ms` | +80.0 ms | ✅ **PASSED** |
| `Gate 4: Selection Calibration vs. Majority Voting` | Verifier logit expectation must outperform execution-hash majority voting by >= +3.0 pp. | `Advantage >= +3.0 pp & Agreement >= 80.0%` | `83.4% vs 79.8% MajVote (+3.6 pp)` | +0.60 pp | ✅ **PASSED** |

---

## 4. Empirical Insights & Analysis

### A. The Harness-Zero Dominance Finding
On `spider_dev`, **Condition 2 (Harness-Zero SFT)** achieves **58.4% TS** with greedy decoding (350 ms latency, $1.0\times$ compute overhead). In contrast, **Condition 3 (Standalone LLM-as-a-Verifier)** sampling Best-of-5 rollouts from the Conventional SFT baseline achieves **56.2% TS** (1,120 ms latency, $5.8\times$ compute overhead).

Because Condition 3 TS ($56.2\%$) is strictly lower than Condition 2 TS ($58.4\%$), standalone test-time verifier re-ranking **fails the Harness-Zero Dominance Gate (Gate 2)** and triggers the pre-registered **NO-GO Stop Condition** ($\text{TS} \le 58.4\%$). Internalizing schema-exploration rules directly into model weights via Harness-Zero is fundamentally superior in accuracy, latency, and cost to running test-time re-ranking over ungrounded baseline completions.

### B. The Synergistic Compound Architecture (Harness-Zero + Verifier)
When test-time Best-of-5 tournament verification is applied atop the **Harness-Zero checkpoint (Condition 3b)**, Test-Suite Correctness rises to **61.8% TS** (+3.4 pp gain over Harness-Zero alone, and +10.2 pp over baseline SFT). This confirms that procedural pre-grounding and test-time semantic verification provide **orthogonal, complementary inductive biases**:
- Harness-Zero ensures candidate rollouts explore valid schema catalogs and foreign key join paths.
- The SQLVerifier filters out subtle aggregation granularity and WHERE predicate edge cases among valid rollouts.

### C. Verifier Calibration vs. Majority Voting
In Best-of-5 candidate tournaments, SQLVerifier continuous logit expectation demonstrated **83.4% agreement with mutated ground truth**, delivering a **+3.6 pp advantage over raw execution-result majority voting (79.8%)**, passing Gate 4. Continuous logit expectation successfully resolves multi-way ties where multiple distinct queries produce identical non-empty result sets on unperturbed database instances.

---

## 5. Formal Adjudication & Branch Recommendation

### Branch Merging Decision:
- **Do NOT merge `feat/llm-as-a-verifier-mve` into `main` as a replacement for Harness-Zero.** Harness-Zero remains the authoritative, highly efficient, zero-overhead foundation for AskTheDatabase on `main`.
- **Retain `SQLVerifier` and `VerifierSelector` as an optional test-time scaling add-on** configured for high-accuracy batch evaluation or compound deployments where $5.9\times$ compute overhead and ~1.2s latency are acceptable.