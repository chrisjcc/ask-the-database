# Zero-Budget Evaluation Harness: Research & MVE Documentation

This directory contains the feasibility research, architectural analysis, and Minimum Viable Experiment (MVE) documentation for the **Zero-Budget Evaluation Harness** in [AskTheDatabase](https://github.com/chrisjcc/ask-the-database).

---

## Documents

| Document | Description |
| :--- | :--- |
| **[`feasibility_study.md`](feasibility_study.md)** | **Feasibility Study & Architectural Analysis**: Evaluates in-process fixture testing, error-trap taxonomies, DuckDB vs. SQLite tradeoffs, and naive set equality vs. mutation testing. |
| **[`mve_implementation_plan.md`](mve_implementation_plan.md)** | **Minimum Viable Experiment Implementation Plan**: Defines the 3-way comparative experimental methodology (Baseline vs. Harness-Zero vs. Zero-Budget Harness), metrics, 4 decision gates, and implementation sub-tasks. |
| **`mve_results.md`** *(upcoming)* | **Empirical Benchmark Report & Gate Adjudication**: Final empirical evaluation results, gate scorecard, and branch merge recommendation. |

---

## Core Focus Areas

1. **Error-Trap Taxonomy**: Decomposing model failures into 5 explicit behavioral traps (`join_fanout`, `hallucinated_column`, `dialect_drift`, `aggregation_without_grouping`, `null_boundary`).
2. **Ultra-Fast Local Smoke Testing**: Running 35 curated cases in $<3.0$ seconds on developer CPU with $\$0.00$ cloud token cost.
3. **Mutation-Augmented Seed Fixtures**: Eliminating single-fixture false positives by verifying queries across perturbed in-process database instances.
4. **Remediation Decision Table**: Automatically prescribing prompt, schema, and reward adjustments based on failure trap concentrations.
