# Comparative Analysis: OPDVR across the Post-Training & Verification Ecosystem

**Document Version**: 1.0.0  
**Date**: November 9, 2026  
**Target Paper**: *On-Policy Distillation with Verifiable Reward* (arXiv:2608.24696, August 2026)  
**Assessing Project**: **AskTheDatabase**  
**Comparative Baselines**: **Harness-Zero**, **LLM-as-a-Verifier**, **HarnessSQL**, and **SQL-Zero**  

---

## Executive Overview

The design space for post-training and inference-time scaling in autonomous Text-to-SQL agents is multidimensional, spanning data generation, supervised warm-starting, reinforcement learning, and test-time verification. 

This document provides a systematic, rigorous comparative assessment of **OPDVR (On-Policy Distillation with Verifiable Reward)** against the four foundational frameworks in our research ecosystem:
1. **Harness-Zero**: In-flight procedural distillation with zero gold SQL leakage (AskTheDatabase SFT foundation).
2. **LLM-as-a-Verifier**: Compound test-time candidate generation, logit tournaments, and trajectory reranking (AskTheDatabase inference engine).
3. **HarnessSQL**: Offline teacher rejection sampling SFT + distributed execution RL via DAPO/GSPO in Slime/Megatron (Yang et al., 2026).
4. **SQL-Zero**: Zero-annotation proposer–solver self-play with format gating and masked template deduplication (ArXiv:2609.04697, 2026).

---

## 1. Multi-Dimensional Comparison Matrix

The table below contrasts the five frameworks across ten critical architectural, algorithmic, and operational dimensions:

| Architectural Dimension | Harness-Zero (AskTheDatabase SFT) | LLM-as-a-Verifier (AskTheDatabase Test-Time) | HarnessSQL (Yang et al., 2026) | SQL-Zero (ArXiv:2609.04697) | OPDVR / GRPD (Lin et al., 2026) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Primary Operational Phase** | Supervised Fine-Tuning (SFT Cold-Start) | Inference / Test-Time Search | SFT + Online RLVR | Synthetic Data Generation + RL | Online Policy Optimization (RLVR) |
| **2. Core Algorithmic Engine** | In-flight procedural reviewer ($\mathcal{K}_{\text{SQL}}$) with active repair | Probabilistic Pivot Tournaments & Decomposed Logits $\mathbb{E}[S]$ | Teacher Rejection Sampling + DAPO / GSPO in Slime | Proposer–Solver Self-Play Spiral with Difficulty Gating | Parameter-Free ReLU Gated Log-Ratio Distillation |
| **3. Supervisory Signal** | Step-level procedural violations (no gold SQL) | Decomposed continuous logit expectations | Coarse binary terminal execution match | Execution match against self-generated reference | Verifier outcome $R$ $\times$ Teacher confidence log-ratio |
| **4. Credit Assignment Granularity** | Step-level (tool call validation) | Candidate-level / Trajectory-level | Trajectory-level (binary scalar over 16 turns) | Trajectory-level (binary scalar over full query) | **Token-level dense credit assignment** |
| **5. Exploration Collapse Mitigation** | Provides valid structural trajectory priors | Overcomes single-candidate stochastic errors | Dynamic Sampling ($\mathcal{Q}_{\text{eff}}$ filters $\mathrm{std}(R)=0$) | Difficulty gating ($d(k)$ targets $k=1$ of $n=5$) | **Continuous token gradients even when $\mathrm{std}(R)=0$** |
| **6. Teacher Dependency & Bias** | Oracle reviewer rules; zero proprietary teacher | Frozen verifier model scoring logits | High (Requires GPT-4o rejection sampling) | Zero teacher (Fully autonomous self-play) | Dual-model forward pass; immune to teacher bias via ReLU |
| **7. Multi-Turn Tool Interaction** | Native (4-tool registry: schema, sample, exec, submit) | Native (Trajectory-level multi-turn ranking) | Native (DSH-SQL 4-tool contract in Node.js) | None (Static single-turn monolithic DDL) | **Fully compatible with multi-turn tool trajectories** |
| **8. Syntactic Diversity Handling** | High (Encourages tool exploration variants) | High (Evaluates execution equivalence) | Low (Rejection sampling collapses onto teacher style) | Moderate (Enforces masked template deduplication) | **Absolute (ReLU gate protects passing novel queries)** |
| **9. Test-Suite Mutation Protection** | N/A (Evaluates procedural rules) | Integrates mutation equivalence checks | None (Single SQLite state vulnerable to false matches) | None (Single synthetic DB state) | **Directly driven by mutation-verified reward $\mathcal{R}_{\text{TS}}$** |
| **10. Compute & Infrastructure Footprint** | Lightweight PyTorch / TRL / PEFT | Lightweight inference logit extraction | Heavyweight (Slime / Megatron / SGLang / Ray) | Medium (Alternating proposer–solver rollouts) | **Lightweight (Shared base weights with LoRA)** |

---

## 2. In-Depth Pairwise Comparative Breakdown

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ARCHITECTURAL RELATIONSHIPS                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│       ┌────────────────────────────────────────────────────────┐                                 │
│       │                      SQL-Zero                          │                                 │
│       │   (Unlabeled Database Synthesis & Proposer Spiral)     │                                 │
│       └───────────────────────────┬────────────────────────────┘                                 │
│                                   │ Generates Task Blueprints                                    │
│                                   ▼                                                              │
│       ┌────────────────────────────────────────────────────────┐                                 │
│       │                    Harness-Zero                        │                                 │
│       │   (Procedural SFT Warm-Start: Zero Gold SQL Leakage)   │                                 │
│       └───────────────────────────┬────────────────────────────┘                                 │
│                                   │ Initializes Base Policy \pi_0                                │
│                                   ▼                                                              │
│       ┌────────────────────────────────────────────────────────┐      HarnessSQL Alternative     │
│       │                    OPDVR / GRPD                        │ <─── Replaces Heavyweight       │
│       │ (Dense Token Credit Assignment + Verifiable Mutation)  │      Slime / Rejection Sampling │
│       └───────────────────────────┬────────────────────────────┘                                 │
│                                   │ Produces High-Quality Policy Checkpoint                      │
│                                   ▼                                                              │
│       ┌────────────────────────────────────────────────────────┐                                 │
│       │                 LLM-as-a-Verifier                      │                                 │
│       │ (Probabilistic Pivot Tournaments & Trajectory Reranking│                                 │
│       └────────────────────────────────────────────────────────┘                                 │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.1 OPDVR vs. Harness-Zero: Sequential Post-Training Stages

#### Algorithmic Comparison
* **Harness-Zero**: Targets the **Cold-Start SFT Stage**. Without prior trajectory demonstrations, an agent placed in an interactive database environment collapses due to unguided exploration. Harness-Zero deploys an in-flight reviewer ($\mathcal{K}_{\text{SQL}}$) that intervenes during student exploration—instructing the agent to inspect schemas before executing queries and catching syntax errors *mid-flight* with **zero ground-truth SQL leakage**.
* **OPDVR**: Targets the **Online RLVR Stage**. Once the model understands the tool contract and basic interaction discipline, it must learn to generalize to novel relational schemas and solve complex multi-table joins. OPDVR provides fine-grained token rewards based on teacher-student confidence discrepancies filtered by execution correctness.

#### Synergy & Complementarity
* **Why Harness-Zero is required before OPDVR**: On-policy distillation requires the student to generate coherent action tokens. If an uninitialized model emits gibberish tool calls or invalid JSON syntax, the teacher log-ratio $\log(\pi_T / \pi_\theta)$ provides little useful gradient structure. Harness-Zero guarantees that the student's initial policy $\pi_0$ is already procedurally disciplined.
* **Why OPDVR completes Harness-Zero**: Offline SFT suffers from distribution shift: as soon as the student takes a novel action not present in the SFT corpus, it drifts into unrecoverable failure states. OPDVR operates *on-policy*, continuously supervising the student's own generated trajectories with dense token guidance.

---

### 2.2 OPDVR vs. LLM-as-a-Verifier: Train-Time vs. Test-Time Scaling

#### Algorithmic Comparison
* **LLM-as-a-Verifier**: Operates entirely at **Test-Time (Inference)**. It assumes the model weights are fixed. Given a user query $q$, the system samples $G$ independent candidate trajectories, evaluates decomposed logit expectations ($\mathbb{E}[S]$) across relational criteria, and executes a Probabilistic Pivot Tournament to select the winning candidate.
* **OPDVR**: Operates entirely at **Train-Time (Post-Training)**. It updates the parameter weights $\theta$ of the student policy via gradient descent, aligning the model's generation distribution with verifiable task execution and teacher knowledge.

#### Synergy & Theorem 4.1 Proof
In Section 4 of the AskTheDatabase white paper, we proved **Theorem 4.1 (*Candidate Pool Quality Bottleneck*)**:
$$\mathcal{V}(G, \tau) \le \text{Pass@}G$$
* Even the most sophisticated tournament verifier cannot select a correct query if zero correct candidates exist in the sampled pool of size $G$.
* By providing dense token-level credit assignment, **OPDVR directly increases raw candidate accuracy and pool density ($\text{Pass@}G$)**.
* Consequently, pairing an OPDVR-trained policy with our LLM-as-a-Verifier tournament reranker creates a compounding performance multiplier: higher pool quality directly elevates the final test-time accuracy frontier.

---

### 2.3 OPDVR vs. HarnessSQL: Elegance vs. Brute-Force Infrastructure

#### Algorithmic Comparison
* **HarnessSQL**:
  - Relies on **Teacher Rejection Sampling** using proprietary high-capacity models (GPT-4o) over thousands of episodes, discarding failed rollouts (survivorship bias).
  - Trains via distributed reinforcement learning (DAPO/GSPO) inside the heavyweight Slime / Megatron-LM / SGLang / Ray framework.
  - Relies on **sparse, binary terminal execution rewards** ($R \in \{0, 1\}$). On 16-turn trajectories, the credit assignment signal is exceptionally weak.
* **OPDVR**:
  - Replaces external teacher trajectory generation with **on-policy student sampling**.
  - Provides **dense token-level credit assignment** via teacher log-probabilities, eliminating the need to wait for full trajectory rewards.
  - Requires only standard PyTorch / Hugging Face / PEFT infrastructure, avoiding the operational complexity of Slime and Megatron-LM.

#### Key Takeaway
HarnessSQL demonstrates that harness-native training is essential for realistic database environments. However, its training methodology is brute-force and resource-intensive. OPDVR achieves superior credit assignment through a mathematically principled, parameter-free gate that runs within standard academic compute budgets.

---

### 2.4 OPDVR vs. SQL-Zero: Policy Optimization vs. Data Generation

#### Algorithmic Comparison
* **SQL-Zero**: Focuses on the **zero-annotation data synthesis challenge**. It uses a dual-agent proposer–solver spiral: a Challenger proposes queries from database schemas, and a Solver attempts to answer them. Tasks are filtered using format gating ($f$) and difficulty gating ($d(k)$ peaking at $k=1$ of $n=5$), while masked template deduplication ($\rho = \lambda |c| / |B|$) prevents linguistic collapse.
  However, SQL-Zero's solver is trained using standard single-turn GRPO with binary outcome rewards.
* **OPDVR**: Focuses on the **policy gradient formulation**. It does not generate synthetic database tasks; rather, it governs how a policy updates its weights given any task and verifiable reward.

#### Integration Opportunity
SQL-Zero and OPDVR are completely orthogonal and can be combined into a state-of-the-art autonomous self-evolving stack:
1. **Task Generation**: SQL-Zero's Challenger generates diverse, difficulty-gated tasks over unlabeled enterprise databases.
2. **Interactive Environment**: AskTheDatabase's multi-turn SQLite sandbox executes the agent's exploratory tool calls.
3. **Reward Verification**: AskTheDatabase's mutation test-suite engine ($\mathcal{R}_{\text{TS}}$) validates candidate equivalence across perturbed database states.
4. **Policy Optimization**: OPDVR / GRPD trains the agent with dense token credit assignment and teacher confidence guidance.

---

## 3. Detailed Analysis of Token-Level Conflict Dynamics

The decisive scientific advantage of OPDVR over all other distillation frameworks is its **token-level conflict removal**.

To demonstrate this concretely in Text-to-SQL, consider an agent generating an action token $a_t$ during a multi-turn episode:

### Scenario A: Correct Multi-Turn Trajectory ($R = +1$), Alternative Valid Join
* **Context**: The user asks for "customers with total orders above $1,000".
* **Student Generation**: The student decides to write a Common Table Expression: `WITH totals AS (...) SELECT ...`.
* **Teacher Preference**: The teacher model strongly prefers an inline subquery: `SELECT ... FROM (SELECT ...)`.
* **Discrepancy**: At the token `WITH`, $\pi_\theta(\text{"WITH"}) = 0.85$, whereas $\pi_T(\text{"WITH"}) = 0.10$.
* **The Conflict**:
  - $\log(\pi_T / \pi_\theta) = \log(0.10 / 0.85) = -2.14 < 0$.
  - Under **Vanilla OPD**: Token update is $-2.14 \cdot \nabla_\theta \log \pi_\theta$. **The student is actively penalized for choosing a valid CTE syntax!**
  - Under **HarnessSQL / Pure RLVR**: The token receives a coarse trajectory bonus, but no fine-grained guidance.
  - Under **OPDVR**: The ReLU gate evaluates $\mathrm{ReLU}(\log(\pi_T / \pi_\theta)) = \mathrm{ReLU}(-2.14) = 0.0$.
  - **Result**: The student is **completely unpenalized**. It preserves its confidence in the valid alternative formulation.

### Scenario B: Incorrect Multi-Turn Trajectory ($R = -1$), Hallucinated Column Name
* **Context**: The agent queries table `customer_orders`.
* **Student Generation**: The student hallucinates the column name `customer_id` (the actual column is `cust_id`).
* **Discrepancy**: The student is overconfident: $\pi_\theta(\text{"customer_id"}) = 0.90$. The teacher, having inspected the DDL, assigns $\pi_T(\text{"customer_id"}) = 0.02$.
* **The Conflict**:
  - $\log(\pi_\theta / \pi_T) = \log(0.90 / 0.02) = +3.80 > 0$.
  - Under **Vanilla OPD**: The update is $+\log(\pi_T / \pi_\theta) = -3.80$. However, on other intermediate tokens where $\pi_T > \pi_\theta$, vanilla OPD applies positive updates, confusing the credit assignment.
  - Under **Pure RLVR**: Every token in the 1,000-token trajectory receives an identical small negative penalty, diluting the blame across table names, tool commands, and syntax.
  - Under **OPDVR**: The token update is exactly:
    $$- \mathrm{ReLU}\left( \log \frac{\pi_\theta}{\pi_T} \right) \nabla_\theta \log \pi_\theta = -3.80 \cdot \nabla_\theta \log \pi_\theta$$
  - **Result**: The hallucinated token is **heavily and specifically suppressed**, directly attacking the root cause of the execution failure.

---

## 4. Synthesis: The AskTheDatabase Unified Trinity

By integrating OPDVR alongside Harness-Zero and LLM-as-a-Verifier, AskTheDatabase establishes an unmatched end-to-end framework covering the full lifecycle of an autonomous Text-to-SQL agent:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE ASK-THE-DATABASE TRINITY                                       │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  1. STAGE 1: IN-FLIGHT PROCEDURAL SFT (Harness-Zero)                                             │
│     • Teaches operational tool discipline and procedural recovery                                │
│     • Enforces schema inspection before query execution                                          │
│     • Zero gold SQL leakage; eliminates survivorship bias                                        │
│                                                                                                  │
│  2. STAGE 2: ON-POLICY DISTILLATION WITH VERIFIABLE REWARDS (OPDVR / GRPD)                       │
│     • Solves multi-turn temporal credit assignment via dense token log-ratios                    │
│     • Bypasses zero-variance exploration collapse (std(R) -> 0)                                  │
│     • ReLU gate ensures immunity to teacher bias and protects valid alternative SQL               │
│     • Supervised by Privileged Schema Teacher with full DDL access                               │
│                                                                                                  │
│  3. STAGE 3: COMPOUND TEST-TIME VERIFICATION (LLM-as-a-Verifier)                                 │
│     • Multi-candidate trajectory rollouts evaluated over decomposed relational criteria          │
│     • Continuous logit expectation E[S] and Probabilistic Pivot Tournaments                      │
│     • Capitalizes on higher candidate pass rates (Pass@G) generated by OPDVR policy              │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

This three-stage progression resolves the train-deploy mismatch, the credit assignment bottleneck, and the test-time search challenge in a unified, theoretically sound, and computationally accessible architecture.
