# LLM-as-a-Verifier Research & Distillation Track

This directory contains the feasibility study and Minimum Viable Experiment (MVE) implementation plan for evaluating **LLM-as-a-Verifier** (continuous logit-expectation verification, test-time scaling, and process reward modeling) for the `AskTheDatabase` project.

---

## Documents in this Track

1. 📄 **[LLM-as-a-Verifier Feasibility Study](feasibility_study.md)**:
   - Evaluates the theoretical foundation of continuous logit-expectation verification (*arXiv:2607.05391*).
   - Formulates the mathematical expectation over scoring logits ($\mathbb{E}[S]$).
   - Outlines the motivation: solving production test-time candidate selection when no ground-truth SQL exists.
   - Compares the architectural trade-offs against **Harness-Zero** (internalized zero-overhead exploration vs. test-time verifier re-ranking).
   - Establishes the 3-way comparative evaluation framework and pre-registered Go/No-Go decision gates.

2. 🛠️ **[Minimum Viable Experiment (MVE) Implementation Plan](mve_implementation_plan.md)**:
   - Detailed engineering roadmap for executing the 3-configuration comparative evaluation:
     - **Configuration 1 (Baseline)**: Conventional SFT baseline with greedy single-pass decoding ($N=1$).
     - **Configuration 2 (Harness-Zero)**: Harness-Zero distilled model with internalized procedural exploration ($N=1$).
     - **Configuration 3 (LLM-as-a-Verifier)**: Baseline model with $N=5$ parallel rollouts re-ranked via `llm_verifier.select()`.
     - **Configuration 3b (Compound)**: Harness-Zero model with $N=5$ rollouts re-ranked via `llm_verifier.select()`.
   - Technical specifications for `SQLVerifier`, decomposed criteria (`configs/verification/sql_verifier_rules.yaml`), and `scripts/run_verifier_mve.py`.
   - Quantitative decision gates and formal stop conditions.

---

## Git Branch & Merge Governance

All work on this research track is isolated on feature branch:
```bash
feat/llm-as-a-verifier-mve
```
The branch will be merged into `main` **only if** the experiment passes all pre-registered decision gates and proves measurable value over Harness-Zero. If the adjudication results in a `NO-GO`, the branch will be preserved as an archived research record.
