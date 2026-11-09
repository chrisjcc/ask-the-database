# Technical Feasibility & Comparative Assessment Study: OPDVR (On-Policy Distillation with Verifiable Reward)

**Document Version**: 1.0.0  
**Date**: November 9, 2026  
**Target Paper**: *On-Policy Distillation with Verifiable Reward* (arXiv:2608.24696, August 2026)  
**Authors**: Wenze Lin, Jiale Zhao, Xitai Jiang, Songde Rao, Yining Li, Shenzhi Wang, Bingxiang He, Gao Huang (LeapLab, Tsinghua University; Qiuzhen College; Beihang University; Peking University)  
**Target Repository**: [https://github.com/LeapLabTHU/OPDVR](https://github.com/LeapLabTHU/OPDVR)  
**Assessing Project**: **AskTheDatabase** ([White Paper Draft](../../../papers/draft/main.pdf) & [Architecture Repo](https://github.com/chrisjcc/ask-the-database))  
**Classification**: Independent Technical Feasibility & Research Positioning Study  

---

## Executive Summary

This feasibility study conducts an in-depth, rigorous scientific and technical analysis of **OPDVR (On-Policy Distillation with Verifiable Reward)** ([arXiv:2608.24696](https://arxiv.org/abs/2608.24696), August 2026) in the context of the **AskTheDatabase** project. 

The central objective is to evaluate whether and how OPDVR—and its group-relative extension **GRPD (Group Relative Policy Distillation)**—can resolve the fundamental **credit assignment bottleneck** and **exploration collapse** encountered during multi-turn, tool-augmented reinforcement learning with verifiable rewards (RLVR) in Text-to-SQL.

### The Core Algorithmic Insight of OPDVR
Post-training of reasoning models currently sits between two competing paradigms:
1. **Reinforcement Learning with Verifiable Rewards (RLVR)** (e.g., GRPO, PPO): Optimizes policies against objective execution feedback (e.g., unit tests, SQL execution against databases). While outcome-aligned and capable of superhuman exploration, rewards are **sparse and trajectory-level**, leading to severe credit assignment challenges across long multi-turn rollouts and causing **zero-variance gradient stalls** ($\mathrm{std}(R) \to 0 \implies A_i \to 0$) when exploration fails.
2. **On-Policy Distillation (OPD)**: Optimizes policies against a teacher model's token distribution over student-sampled on-policy rollouts. While providing **dense, token-level supervisory signals**, OPD is **purely distributional**: it forces the student to mimic the teacher regardless of whether the generated trajectory is correct or incorrect, capping student capability at teacher proficiency and penalizing valid alternative solutions.

Prior attempts to combine RLVR and OPD rely on heuristic loss interpolations ($\mathcal{L}_{\text{RL}} + \lambda \mathcal{L}_{\text{OPD}}$) or hard trajectory filtering, introducing brittle hyperparameters and failing to resolve token-level gradient conflicts.

**OPDVR resolves this trade-off via a parameter-free, sign-aligning ReLU gate.**
By reinterpreting sampled-token OPD from an RLVR advantage perspective, the authors reveal that the implicit token reward of OPD is the log-probability ratio $r_t = \log(\pi_T(o_t) / \pi_\theta(o_t))$. When conditioned on trajectory correctness $R \in \{+1, -1\}$, standard OPD creates two catastrophic gradient conflicts:
* **Type I Conflict (Correct Trajectory, $\pi_\theta > \pi_T$)**: The student discovers a correct solution with high confidence, but the teacher assigns lower probability. Standard OPD penalizes the student ($\Delta_{\text{OPD}} < 0$), pulling down the probability of correct tokens.
* **Type II Conflict (Incorrect Trajectory, $\pi_T > \pi_\theta$)**: The student produces an erroneous rollout, but the teacher assigns higher probability to some intermediate tokens. Standard OPD rewards the student ($\Delta_{\text{OPD}} > 0$), reinforcing error-prone reasoning patterns.

OPDVR eliminates these conflicts by applying a simple sign-consistency gate:
$$R_{\text{OPDVR}}(o_t) = R(o) \cdot \mathrm{ReLU}\left( R(o) \cdot \log \frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})} \right)$$
* On **correct trajectories** ($R = +1$), tokens receive $+ \mathrm{ReLU}(\log(\pi_T / \pi_\theta))$, rewarding tokens where the teacher is more confident than the student, and zeroing out tokens where the student is already confident.
* On **incorrect trajectories** ($R = -1$), tokens receive $- \mathrm{ReLU}(\log(\pi_\theta / \pi_T))$, penalizing overconfident student errors where $\pi_\theta > \pi_T$, and zeroing out tokens where the teacher was also uncertain or wrong.

---

### Strategic Recommendation: **ADOPT FOR MULTI-TURN RLVR IN ASK-THE-DATABASE**
* **Verdict**: **GO (High Value, High Compatibility)**.
* **Primary Integration Point**: Integrate **Group Relative Policy Distillation (GRPD)** directly into AskTheDatabase's multi-turn RLVR training loop (`TRLGRPOAdapter` / SkyRL environment), replacing pure GRPO with GRPD powered by our **Hierarchical Mutation Test-Suite Verifier ($\mathcal{R}_{\text{TS}}$)**.
* **Teacher Architecture Formulation**: Deploy a **Privileged Schema Teacher** (a 7B or 14B model provided with full flattened schema DDL in context) to supervise an **Unprivileged Student Agent** that must explore zero-knowledge enterprise databases via tool actions (`inspect_schema`, `sample_data`, `execute_sql`). This achieves dense token supervision without requiring expensive proprietary teacher APIs.
* **Complementarity**: OPDVR does not replace our existing pillars; it unites them:
  - **Harness-Zero**: Remains the foundational cold-start SFT stage (procedural distillation with zero SQL leakage).
  - **OPDVR / GRPD**: Powers the online RLVR stage, resolving the 27-step exploration freeze and providing token-level credit assignment across multi-turn tool trajectories.
  - **LLM-as-a-Verifier**: Remains the inference-time tournament reranking engine, capitalizing on the dramatically higher candidate quality ($\text{Pass@}G$) produced by the OPDVR-trained policy.

---

## 1. Problem and Motivation

### 1.1 The Post-Training Dilemma in Complex Reasoning
Large language model post-training has increasingly bifurcated into two dominant methodologies:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE POST-TRAINING METHODOLOGY SPLIT                             │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│   Reinforcement Learning with            │   On-Policy Distillation                    │
│   Verifiable Rewards (RLVR)              │   (OPD)                                     │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Objective: Maximize expected reward    │ • Objective: Minimize reverse KL divergence │
│   $\mathbb{E}_{o \sim \pi_\theta}[R(o)]$ │   $D_{\mathrm{KL}}(\pi_\theta \| \pi_T)$   │
│ • Supervision: Sparse, trajectory-level  │ • Supervision: Dense, token-level log-ratio │
│ • Credit Assignment: Challenging (all    │ • Distributional: Mimics teacher blindly,   │
│   tokens share trajectory scalar)        │   agnostic to ground-truth correctness      │
│ • Exploration: Unbounded, self-evolving  │ • Upper Bound: Strictly capped at teacher   │
│ • Failure Mode: Zero-variance gradient   │ • Failure Mode: Distillation of teacher     │
│   stall ($\mathrm{std}(R) = 0$)          │   errors and syntactic over-constraint      │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

In mathematical reasoning and code generation, RLVR methods such as PPO, GRPO (DeepSeekMath, DeepSeek-R1), and DAPO have proven exceptionally powerful because objective verifiers (e.g., Python execution, SymPy symbolic equivalence) prevent reward hacking and drive out-of-distribution generalization. 

However, RLVR suffers from the classical **temporal credit assignment problem**:
When an agent generates a reasoning sequence of hundreds of tokens—or an interactive agent executes a 5-turn database exploration episode—a binary scalar reward $R \in \{0, 1\}$ at the terminal state treats every intermediate decision identically. In an episode where an agent correctly inspects table schemas, performs sample data probing, constructs a complex multi-table join, but mistypes a single literal in the `WHERE` clause, standard RLVR applies a uniform negative penalty to all generated tokens, actively unlearning valid exploratory behavior.

Conversely, On-Policy Distillation (OPD) samples rollouts from the student policy $\pi_\theta$ and supervises each token using the teacher's conditional distribution $\pi_T(o_t \mid q, o_{<t})$. Because the teacher evaluates tokens generated on the student's own trajectory, OPD avoids the severe off-policy distribution shift of standard SFT. However, OPD minimizes reverse KL divergence:
$$\min_\theta D_{\mathrm{KL}}(\pi_\theta \parallel \pi_T) = \mathbb{E}_{o \sim \pi_\theta} \left[ \sum_{t=1}^{|o|} \log \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_T(o_t \mid q, o_{<t})} \right]$$
This objective is completely agnostic to whether the rollout actually solved the problem. If the teacher has blind spots, biases, or suboptimal habits, the student is forced to inherit them.

### 1.2 Limitations of Prior Hybrid Methods
Recognizing this complementarity, several recent works attempted to combine OPD and RLVR:
1. **Linear Loss Interpolation** (e.g., Hubotter et al., 2026; Xiao et al., 2026):
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RLVR}} + \lambda \mathcal{L}_{\text{OPD}}$$
   *Failure Mode*: Introduces a sensitive weighting hyperparameter $\lambda$. When RL and OPD gradients point in opposite directions on specific tokens, they cancel out or destabilize optimization.
2. **Trajectory-Level Gating** (e.g., Uni-OPD, RG-OPD; Hou et al., 2026; Akhondzadeh et al., 2026): Discards entire episodes where the teacher's preference disagrees with the verifier.
   *Failure Mode*: Excessively coarse. Discarding whole trajectories wastes valuable gradient signals from valid intermediate reasoning steps.
3. **Token-Level Scaling** (e.g., SG-OPD; Xu et al., 2026b): Multiplies conflicting tokens by heuristic attenuation factors.
   *Failure Mode*: Does not completely eliminate conflicting gradient updates and requires hand-tuned scaling thresholds.

### 1.3 The Underlying Motivation of OPDVR
The authors of OPDVR decompose the sampled-token OPD gradient:
$$\nabla_\theta \mathcal{L}_{\text{OPD}} = - \mathbb{E}_{o \sim \pi_\theta} \left[ \sum_{t=1}^{|o|} \log \left( \frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})} \right) \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) \right]$$
Comparing this directly with the REINFORCE policy gradient:
$$\nabla_\theta \mathcal{L}_{\text{RLVR}} = - \mathbb{E}_{o \sim \pi_\theta} \left[ \sum_{t=1}^{|o|} R(o) \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) \right]$$
reveals that **sampled-token OPD implicitly assigns every token an intrinsic reward**:
$$R_{\text{OPD}}(o_t) = \log \frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})}$$
In classical RLVR, the fundamental sign convention requires that:
* Every token in a verified correct trajectory should receive a **non-negative advantage** ($\ge 0$).
* Every token in a verified incorrect trajectory should receive a **non-positive advantage** ($\le 0$).

Sampled-token OPD constantly violates this convention. When conditioned on trajectory correctness $R(o) \in \{+1, -1\}$:
$$\text{Implicit OPD Gradient Weight} = \begin{cases} +\log \frac{\pi_T(o_t)}{\pi_\theta(o_t)}, & \text{if } R(o) = +1 \\ -\log \frac{\pi_\theta(o_t)}{\pi_T(o_t)} = +\log \frac{\pi_T(o_t)}{\pi_\theta(o_t)}, & \text{if } R(o) = -1 \end{cases}$$
Because the log-ratio can be positive or negative independently of $R(o)$, OPD introduces two severe pathological failure modes:
1. **Type I Conflict (Correct Trajectory, $\pi_\theta > \pi_T$)**: The student generates a valid, correct token with high confidence, but the teacher's probability is lower ($\log(\pi_T / \pi_\theta) < 0$). Standard OPD applies a **negative reward**, actively depressing the probability of a correct answer!
2. **Type II Conflict (Incorrect Trajectory, $\pi_T > \pi_\theta$)**: The student generates an incorrect trajectory, but along the path, some tokens have $\pi_T > \pi_\theta$. Standard OPD applies a **positive reward**, actively reinforcing tokens that led to a verifiably failed outcome!

OPDVR replaces this unconstrained log-ratio with a **parameter-free ReLU gate**, ensuring the token-level update direction is strictly aligned with the verifier outcome while preserving the teacher's relative confidence as the magnitude of the update.

---

## 2. Applicability to AskTheDatabase

AskTheDatabase addresses multi-turn, tool-augmented natural-language-to-SQL generation over relational databases. We now evaluate the technical and operational applicability of OPDVR to this domain.

### 2.1 The Multi-Turn Tool POMDP Formulation
In AskTheDatabase, an interaction episode is formalized as an interactive decision process:
* **Task Context**: $x = (q, \mathcal{D})$, where $q$ is the natural language question and $\mathcal{D}$ is an unseen relational database.
* **State $s_t$**: Dialogue history up to turn $t$, containing system instructions, tool calls, and execution feedback: $s_t = (x, a_1, o_1, \dots, a_{t-1}, o_{t-1})$.
* **Action Space $\mathcal{A}$**: 4 operational tool calls:
  1. `inspect_schema(table_name)`: Retrieves table DDL and column data types.
  2. `sample_data(table_name, limit)`: Samples actual rows to observe value formatting and categorical values.
  3. `execute_sql(sql_query)`: Executes trial SQL against an isolated SQLite sandbox to observe results or runtime errors.
  4. `final_answer(sql_query)`: Submits the final candidate SQL query for evaluation.
* **Trajectory Structure**:
  $$\tau = (a_1, o_1, a_2, o_2, \dots, a_T, o_T)$$
  where agent action tokens $a_t$ are subject to policy loss, while environment observation tokens $o_t$ are masked out ($\text{mask} = 0$).

### 2.2 Feasibility of Dual-Model Interaction in Multi-Turn Tool Environments
To apply OPDVR to AskTheDatabase:
1. **Student Policy $\pi_\theta$**: Generates on-policy multi-turn rollouts by interacting dynamically with the `SQLiteEnvironment`. The student decides when to call tools and when to emit final SQL.
2. **Trajectory Verifier $\mathcal{R}_{\text{TS}}$**: Evaluates the terminal action `final_answer` against mutated test-suite database states ($\mathcal{D}_0, \dots, \mathcal{D}_K$), producing a verified reward $R(\tau) \in \{0, 1\}$ or scalar continuous reward $\mathcal{R}_{\text{TS}} \in [0, 1]$.
3. **Teacher Policy $\pi_T$**: Given the student's completed trajectory $\tau$, the teacher performs a single forward pass over the token sequence to compute teacher log-probabilities $\log \pi_T(a_t \mid s_t)$ exclusively on the agent-generated action tokens $a_t$.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    OPDVR MULTI-TURN TOOL-AUGMENTED ROLLOUT FLOW                             │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [Question q + DB]                                                                          │
│         │                                                                                   │
│         ▼                                                                                   │
│  ┌──────────────┐   Turn 1 Action: inspect_schema   ┌─────────────────┐                     │
│  │   Student    │ ────────────────────────────────> │ SQLite Sandbox  │                     │
│  │ Policy \pi_\theta │ <──────────────────────────────── │   Environment   │                │
│  │              │   Turn 1 Obs: CREATE TABLE ...    └─────────────────┘                     │
│  │              │                                                                           │
│  │              │   Turn 2 Action: execute_sql                                              │
│  │              │ ────────────────────────────────> [Executes trial query]                  │
│  │              │ <──────────────────────────────── [Returns 3 rows]                       │
│  │              │                                                                           │
│  │              │   Turn 3 Action: final_answer(SELECT ...)                                 │
│  └──────┬───────┘                                                                           │
│         │                                                                                   │
│         ├──────────────────────────────┬───────────────────────────────┐                    │
│         ▼                              ▼                               ▼                    │
│  ┌──────────────┐              ┌──────────────┐                ┌──────────────┐             │
│  │   Student    │              │ Test-Suite   │                │   Teacher    │             │
│  │ Log-Probs    │              │ Mutation     │                │  Evaluation  │             │
│  │ \log \pi_\theta(a_t)│       │ Verifier R_TS│                │ \log \pi_T(a_t)            │
│  └──────┬───────┘              └──────┬───────┘                └──────┬───────┘             │
│         │                             │                               │                     │
│         └──────────────────────┐      │      ┌────────────────────────┘                     │
│                                ▼      ▼      ▼                                              │
│                     ┌────────────────────────────────────┐                                  │
│                     │       OPDVR ReLU Gating Engine     │                                  │
│                     │  R_OPDVR = R * ReLU(R * log(T/\theta))│                               │
│                     └─────────────────┬──────────────────┘                                  │
│                                       ▼                                                     │
│                     ┌────────────────────────────────────┐                                  │
│                     │ Backpropagate Policy Gradient      │                                  │
│                     │ Dense token-level update into \pi_\theta│                             │
│                     └────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Privileged Schema Teacher Formulation (Breakthrough Advantage)
In standard mathematical reasoning (e.g., AIME, MATH), teacher models are typically simply larger versions of the student (e.g., 32B teacher distilling into 7B student).

In **AskTheDatabase**, we can introduce a vastly more potent paradigm: **The Privileged Schema Teacher**.
* **The Asymmetry**: In production Text-to-SQL, real-world schemas have dozens or hundreds of tables. The student agent is forced to operate under a **zero-knowledge regime**, spending turns exploring catalog metadata. However, during training, the training environment has access to the full, verified DDL and gold reference queries!
* **Privileged Teacher Setup**: We construct a teacher prompt that includes the complete, flattened schema DDL and domain foreign-key constraints directly in context.
* **Dense Supervision**: When evaluating the student's trajectory, the privileged teacher already possesses the complete schema picture. It assigns high log-probability $\log \pi_T(a_t)$ to correct table joins, valid column selections, and optimal filter predicates, providing dense, oracle-informed token supervision to the student's zero-knowledge exploration steps!

---

## 3. Potential Added Value for AskTheDatabase

Integrating OPDVR and GRPD directly addresses the four most critical bottlenecks in AskTheDatabase's current architecture:

### 3.1 Resolving the Multi-Turn Temporal Credit Assignment Bottleneck
* **Current Limitation**: In AskTheDatabase's GRPO pipeline (`src/ask_the_database/training/train_rlvr.py`), rollouts consist of 500 to 1,500 tokens across multiple interaction turns. If a query achieves test-suite equivalence ($\mathcal{R}_{\text{TS}} = 1.0$), all 1,500 tokens receive a positive advantage. If it fails ($\mathcal{R}_{\text{TS}} = 0.0$), all 1,500 tokens receive a negative advantage.
* **OPDVR Solution**: OPDVR decomposes the trajectory into token-level advantages. Even if the overall rollout fails ($R = -1$):
  - Tokens where the student chose a standard, reasonable join condition (where the teacher was equally confident) receive $\mathrm{ReLU}(\log(\pi_\theta / \pi_T)) \approx 0$ (no penalty).
  - Tokens where the student hallucinated a non-existent column name or syntax error (where student confidence drastically exceeded teacher confidence: $\pi_\theta \gg \pi_T$) receive a **heavy penalty** ($-\mathrm{ReLU}(\log(\pi_\theta / \pi_T)) \ll 0$).
  - Credit assignment is localized precisely to the erroneous tokens, preserving valid exploratory turns.

### 3.2 Bypassing the Zero-Variance Exploration Collapse ($\mathrm{std}(R) \to 0$)
* **Current Limitation**: In Section 5.3 of our white paper, we proved that during the cold-start phase of GRPO, when all $G$ sampled rollouts fail to execute or produce wrong results ($R_i = 0$ for all $i \in \{1, \dots, G\}$), the intra-group standard deviation becomes zero:
  $$\mathrm{std}(\{R_i\}) = 0 \implies \hat{A}_i = \frac{R_i - \bar{R}}{\mathrm{std}(R) + \epsilon} \to 0$$
  This causes policy gradient updates to completely vanish, stalling exploration for up to 27 training steps on hard queries.
* **OPDVR Solution**: Under OPDVR, even when all rollouts in a group fail ($R = -1$), the token-level reward is non-zero:
  $$R_{\text{OPDVR}}(o_t) = -\mathrm{ReLU}\left( \log \frac{\pi_\theta(o_t)}{\pi_T(o_t)} \right)$$
  The model receives active gradient signals penalizing overconfident hallucinations, reshaping the policy distribution toward the teacher even when terminal execution rewards are strictly zero!

### 3.3 Immunity to Syntactic Over-Constraint (Preserving Alternative Valid SQL)
* **Current Limitation of Vanilla Distillation**: In Text-to-SQL, there are often dozens of semantically equivalent SQL formulations for the same query (e.g., using a Common Table Expression `WITH ...` vs. an inline subquery; `WHERE col IN (SELECT ...)` vs. `INNER JOIN`; explicit `JOIN ... ON` vs. comma join).
  Under standard OPD, if the teacher prefers CTEs while the student generates an inline subquery, the teacher assigns low probability to the student's tokens ($\pi_T < \pi_\theta$). Standard OPD **severely penalizes the student**, forcing it to abandon valid SQL syntax.
* **OPDVR Solution**: Under OPDVR, if the student's subquery executes and passes test-suite verification ($R = +1$), the token reward is:
  $$R_{\text{OPDVR}}(o_t) = +\mathrm{ReLU}\left( \log \frac{\pi_T(o_t)}{\pi_\theta(o_t)} \right) = 0 \quad (\text{since } \pi_T < \pi_\theta)$$
  **The student is completely exempt from penalty!** The ReLU gate guarantees that as long as the generated query is verifiably correct, the student is free to discover alternative syntactic formulations and tool usage patterns that the teacher never considered.

### 3.4 Synergistic Amplification of Test-Time Verification
* **Connection to Theorem 4.1 (*Candidate Pool Quality Bottleneck*)**: In our white paper, we proved that test-time verification yield $\mathcal{V}$ is strictly bounded by the policy's raw generation diversity:
  $$\mathcal{V}(G, \tau) \le \text{Pass@}G$$
  A policy trained with GRPD achieves higher token precision, fewer tool syntax errors, and higher candidate pass rates. When fed into our **Compound Trajectory Verifier** and **Probabilistic Pivot Tournaments**, the test-time accuracy ceiling rises proportionally.

---

## 4. Complementarity and Differentiation

To position OPDVR rigorously within the research ecosystem, we compare it against the four landmark frameworks investigated in our research track:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               POST-TRAINING & VERIFICATION TAXONOMY                              │
├───────────────────┬─────────────────────────┬──────────────────────────┬─────────────────────────┤
│ Framework         │ Learning Phase          │ Supervisory Signal       │ Algorithmic Engine      │
├───────────────────┼─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ **Harness-Zero**  │ SFT Cold-Start          │ Procedural Environment   │ Procedural Distillation │
│                   │ (Pre-RL)                │ Feedback ($\mathcal{K}$) │ (Zero Gold Leakage)     │
├───────────────────┼─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ **HarnessSQL**    │ SFT + Policy RL         │ Offline Teacher Rollouts │ Rejection SFT +         │
│                   │ (Full Pipeline)         │ + Binary Execution       │ DAPO / GSPO in Slime    │
├───────────────────┼─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ **SQL-Zero**      │ Self-Play Data          │ Execution Match against  │ Proposer–Solver Spiral  │
│                   │ Synthesis + RL          │ Synthetic Reference      │ Format/Difficulty Gate  │
├───────────────────┼─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ **LLM-as-a-       │ Test-Time Inference     │ Decomposed Logit Post.   │ Tournaments & Trajectory│
│ Verifier**        │ (Post-Training)         │ $\mathbb{E}[S]$          │ Reranking               │
├───────────────────┼─────────────────────────┼──────────────────────────┼─────────────────────────┤
│ **OPDVR / GRPD**  │ Online Policy RLVR      │ Verifier Outcome $R$     │ Parameter-Free ReLU     │
│                   │ (Fine-Tuning)           │ $\times$ Teacher Ratio   │ Log-Ratio Gating        │
└───────────────────┴─────────────────────────┴──────────────────────────┴─────────────────────────┘
```

### 4.1 OPDVR vs. Harness-Zero
* **Differentiation**: Harness-Zero is an **offline procedural distillation** methodology for SFT. An oracle reviewer ($\mathcal{K}_{\text{SQL}}$) monitors intermediate tool execution and provides procedural guidance without leaking gold SQL. OPDVR is an **on-policy reinforcement learning** objective that operates *during* online RLVR exploration.
* **Relationship**: **Strictly Complementary (Sequential Pipeline)**.
  Harness-Zero provides the ideal warm-started policy $\pi_0$. Once $\pi_0$ is initialized, GRPD replaces standard GRPO during the RL exploration phase, preventing distributional collapse and accelerating sample efficiency.

### 4.2 OPDVR vs. HarnessSQL
* **Differentiation**: HarnessSQL relies on **rejection sampling** from proprietary teachers (GPT-4o) to generate offline SFT datasets, followed by distributed RL (DAPO/GSPO) with coarse binary execution rewards. It suffers from survivorship bias and severe credit assignment issues across its 16-turn trajectories.
* **Relationship**: **Algorithmic Leap Forward**.
  OPDVR replaces HarnessSQL's brute-force rejection sampling and sparse binary RL with on-policy distillation aligned with verifiable test-suite execution, eliminating the need for expensive proprietary teacher APIs during rollout generation.

### 4.3 OPDVR vs. SQL-Zero
* **Differentiation**: SQL-Zero focuses on **zero-annotation task generation** via proposer–solver self-play over unlabeled schemas, using difficulty gating ($d(k)$) and masked template deduplication ($\rho$). However, SQL-Zero's solver is trained using standard single-turn GRPO with binary outcome rewards.
* **Relationship**: **Orthogonal and Combinable**.
  SQL-Zero solves the *data generation problem* (synthesizing diverse SQL tasks without human labels). OPDVR solves the *policy optimization problem* (distilling teacher guidance into the solver policy under verifiable execution). They can be directly unified: SQL-Zero acts as the proposer, and OPDVR trains the solver.

### 4.4 OPDVR vs. LLM-as-a-Verifier
* **Differentiation**: LLM-as-a-Verifier operates strictly at **test-time (inference)**. It samples $G$ candidates from a frozen model, scores trajectories via continuous logit expectation $\mathbb{E}[S]$, and conducts Probabilistic Pivot Tournaments to select the best output. OPDVR operates strictly at **train-time (post-training)** to update the weights of the policy.
* **Relationship**: **Mutually Reinforcing**.
  A policy trained with OPDVR generates higher-density candidate pools (higher $\text{Pass@}G$). As established by Theorem 4.1 in our white paper, the ceiling of tournament verification $\mathcal{V}$ scales monotonically with pool pass rate.

---

## 5. Feasibility and Implementation Architecture

We now evaluate the concrete engineering feasibility of incorporating OPDVR into AskTheDatabase.

### 5.1 Mathematical Formulation of GRPD in AskTheDatabase
In AskTheDatabase, rollouts are generated in groups of size $G$ (typically $G \in \{4, 8\}$) for each prompt $q$:
$$\{ \tau_1, \tau_2, \dots, \tau_G \} \sim \pi_\theta(\cdot \mid q)$$
Each trajectory $\tau_i$ consists of an alternating sequence of action tokens $a_{i,t}$ and environment observation tokens $o_{i,t}$.

#### 1. Verifiable Test-Suite Reward Calculation
Each trajectory $\tau_i$ is evaluated by our test-suite mutation engine across $K$ perturbed database states:
$$\mathcal{R}_{\text{TS}}(\tau_i) = \prod_{k=0}^K \mathbb{I}\left( Q_{\text{gen}}(\mathcal{D}_k) \equiv Q_{\text{gold}}(\mathcal{D}_k) \right) \in \{0, 1\}$$
Optionally incorporating our 4-tier rubric bonus ($\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{syntax}} + \mathcal{R}_{\text{schema}} + \mathcal{R}_{\text{TS}} + \mathcal{R}_{\text{eff}}$).

#### 2. Group Relative Advantage Estimation
The group-relative advantage $\hat{A}_i$ for trajectory $i$ is computed via GRPO normalization:
$$\hat{A}_i = \frac{\mathcal{R}_{\text{total}}(\tau_i) - \mathrm{mean}\left(\{\mathcal{R}_{\text{total}}(\tau_j)\}_{j=1}^G\right)}{\mathrm{std}\left(\{\mathcal{R}_{\text{total}}(\tau_j)\}_{j=1}^G\right) + \epsilon}$$

#### 3. Token-Level GRPD Reward
For each action token $a_{i,t}$ in trajectory $\tau_i$, the GRPD reward is formulated as:
$$R_{\text{GRPD}}(a_{i,t}) = \hat{A}_i \cdot \mathrm{ReLU}\left( \mathrm{sign}(\hat{A}_i) \cdot \log \frac{\pi_T(a_{i,t} \mid s_{i,t})}{\pi_\theta(a_{i,t} \mid s_{i,t})} \right)$$

#### 4. GRPD Policy Optimization Loss
The sequence-level loss for the group is:
$$\mathcal{L}_{\text{GRPD}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|\mathcal{A}_i|} \sum_{t \in \mathcal{A}_i} R_{\text{GRPD}}(a_{i,t}) \cdot \log \pi_\theta(a_{i,t} \mid s_{i,t})$$
where $\mathcal{A}_i$ is the set of indices corresponding to agent-generated action tokens (excluding environment observation tokens).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          GRPD REWARD BEHAVIOR MATRIX                                        │
├──────────────────────────┬──────────────────────────────┬───────────────────────────────────┤
│ Relative Advantage       │ Relative Confidence          │ Resulting GRPD Token Reward       │
├──────────────────────────┼──────────────────────────────┼───────────────────────────────────┤
│ **Positive ($\hat{A}_i > 0$)** │ Teacher > Student ($\pi_T > \pi_\theta$) │ $+ \hat{A}_i \log(\pi_T / \pi_\theta)$ (Strong Reward) │
│ (Above Group Average)    │ Student $\ge$ Teacher ($\pi_\theta \ge \pi_T$) │ **$0.0$ (Zero Gated - No Penalty)**       │
├──────────────────────────┼──────────────────────────────┼───────────────────────────────────┤
│ **Negative ($\hat{A}_i < 0$)** │ Student > Teacher ($\pi_\theta > \pi_T$) │ $- |\hat{A}_i| \log(\pi_\theta / \pi_T)$ (Heavy Penalty) │
│ (Below Group Average)    │ Teacher $\ge$ Student ($\pi_T \ge \pi_\theta$) │ **$0.0$ (Zero Gated - No Misleading Boost)** │
└──────────────────────────┴──────────────────────────────┴───────────────────────────────────┘
```

### 5.2 Computational Overhead & Memory Profiling
A common concern with on-policy distillation is the computational cost of running two models simultaneously. We analyze the VRAM and compute budget:

1. **Rollout Generation**: Only the **student model** $\pi_\theta$ generates tokens. Generation runs at full inference speed (via vLLM or Hugging Face generate). The teacher model is completely idle during rollouts.
2. **Teacher Forward Pass**: Once rollouts are generated, the teacher model $\pi_T$ performs a single batched **forward pass** (evaluation mode, no gradients, no KV cache accumulation) to compute $\log \pi_T(a_t \mid s_t)$.
   - For a sequence of length $L = 1,024$, a forward pass takes $\approx 15\%$ of the compute required for autoregressive generation.
3. **Student Backward Pass**: The student computes $\log \pi_\theta(a_t \mid s_t)$, calculates $R_{\text{GRPD}}(a_t)$, and backpropagates gradients.
4. **Memory Footprint with PEFT / LoRA**:
   - In our current training stack, the student is trained using **PEFT LoRA** (or rsLoRA as added in PR #100).
   - The student base model weights are frozen and shared or quantized.
   - If using a **Same-Architecture Teacher** (e.g., student = Qwen2.5-Coder-7B with active LoRA; teacher = Qwen2.5-Coder-7B base or a frozen privileged checkpoint), the base model weights are **100% shared in GPU VRAM**!
   - Under weight sharing, the teacher requires **0 MB additional parameter VRAM**, needing only memory for its intermediate activations during the forward pass!

---

## 6. Experimental Proposal: Proof of Concept

To validate OPDVR in AskTheDatabase empirically without confounding variables, we specify a clean, controlled Proof of Concept (PoC).

### 6.1 Core Hypotheses
* **H1 (Credit Assignment)**: GRPD achieves $\ge 4.0\,\text{pp}$ higher Test-Suite Accuracy ($\mathcal{R}_{\text{TS}}$) than standard GRPO within the first 100 training steps, demonstrating superior sample efficiency on multi-turn trajectories.
* **H2 (Plateau Mitigation)**: On hard queries with initial zero-pass rates ($\text{Pass@}G = 0$), GRPD maintains non-zero gradient norms ($||\nabla_\theta \mathcal{L}|| > 0.05$) and reduces the exploration stall from 27 steps to $< 5$ steps.
* **H3 (Syntactic Flexibility)**: Compared to vanilla OPD, GRPD preserves valid alternative SQL query structures, exhibiting $\ge 2.5\times$ higher syntax entropy on passing queries.
* **H4 (Privileged Distillation)**: A privileged 7B teacher (with full DDL) distills effective schema retrieval habits into an unprivileged 7B student, matching the execution performance of a 14B unprivileged baseline.

### 6.2 Experimental Conditions
We specify 5 conditions evaluated on the Spider 1.0 Dev split (1,034 queries across 20 databases):

| Condition ID | Algorithm | Initialization | Verifier / Feedback | Teacher Specification |
| :--- | :--- | :--- | :--- | :--- |
| **C0 (Baseline SFT)** | SFT | Qwen2.5-Coder-7B | None (Static supervised) | N/A |
| **C1 (Pure RLVR)** | GRPO | C0 Checkpoint | Test-Suite Mutation $\mathcal{R}_{\text{TS}}$ | None |
| **C2 (Vanilla OPD+GRPO)**| OPD + GRPO | C0 Checkpoint | $\mathcal{R}_{\text{TS}}$ + Un-gated Sampled OPD | Qwen2.5-Coder-7B (Base or SFT) |
| **C3 (Binary OPDVR)** | OPDVR | C0 Checkpoint | Binary $\mathcal{R}_{\text{TS}} \in \{0, 1\}$ + ReLU Gate | Same Teacher as C2 |
| **C4 (Full GRPD)** | GRPD | C0 Checkpoint | Group Relative $\hat{A}_i$ + ReLU Gate | Same Teacher as C2 |
| **C5 (Inverse-Gated Abl)**| Inv-GRPD | C0 Checkpoint | Inverted ReLU Gate | Same Teacher as C2 |

### 6.3 Evaluation Metrics
1. **Test-Suite Accuracy (TS)**: Percentage of queries passing mutation equivalence across $K=5$ perturbed database states.
2. **Execution Accuracy (EX)**: Classical single-database execution match.
3. **Exploration Efficiency**: Average turns to final answer; tool error frequency (syntax errors, non-existent columns).
4. **Zero-Gated Token Ratio ($r_{\text{zero}}$)**: Percentage of tokens masked out by the ReLU gate throughout training (expected band: $40\% - 50\%$).
5. **Zero-Variance Stall Steps**: Consecutive training steps where policy gradients vanish due to $\mathrm{std}(R) = 0$.

---

## 7. Risks, Failure Modes, and Mitigation Strategies

| Risk / Failure Mode | Root Cause | Impact | Mitigation Strategy in AskTheDatabase |
| :--- | :--- | :--- | :--- |
| **1. Environment Token Leakage** | Distillation loss accidentally applied to SQLite observation tokens | Policy attempts to predict environment outputs rather than actions | Enforce strict token-type masking: compute log-ratios and loss **only on agent action tokens** ($a_t$). |
| **2. Teacher Hallucination Amplification** | Teacher places high confidence on invalid column names | False positive rewards on incorrect trajectories | Mitigated by design: OPDVR's ReLU gate zeroes out tokens where $\pi_T > \pi_\theta$ on failed trajectories ($R = -1$). |
| **3. Teacher-Student Vocabulary / Tokenizer Mismatch** | Using cross-family teacher models (e.g., Llama-3 teacher with Qwen-2.5 student) | Incompatible token IDs and logit shapes | Restrict teacher-student pairs to the **same tokenizer family** (e.g., Qwen2.5-Coder-7B / 14B / 32B). |
| **4. GPU VRAM Exhaustion** | Loading full 32B teacher alongside 7B student on single GPU | Out-of-memory during batched forward pass | Use **LoRA Weight Sharing** (same 7B base model) or host teacher on a separate vLLM inference instance over local HTTP. |
| **5. Sequence Length Explosion** | Unconstrained multi-turn rollouts exceeding context window | Truncation of critical final SQL submission | Enforce strict turn budgets (max 5 turns) and truncate intermediate tool responses to 400 tokens via SQLite sandbox guards. |

---

## 8. Strategic Recommendation & Research Roadmap

### 8.1 Final Verdict: **ADOPT (HIGH PRIORITY)**
OPDVR provides the exact theoretical and algorithmic mechanism missing from AskTheDatabase's multi-turn RLVR pipeline. It solves the temporal credit assignment dilemma and exploration collapse without introducing sensitive balancing hyperparameters or compromising the student's ability to discover valid alternative SQL formulations.

### 8.2 Research & Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             OPDVR INTEGRATION ROADMAP                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  PHASE 1: Core Mathematical Engine (Week 1)                                                 │
│  • Implement `OPDVRLoss` and `GRPDLoss` in `src/ask_the_database/training/losses.py`        │
│  • Add unit tests verifying Proposition C.1 (directional alignment) and gradient masking    │
│                                                                                             │
│  PHASE 2: Trainer Integration & Action Masking (Week 2)                                     │
│  • Extend `TRLGRPOAdapter` to support teacher log-probability scoring on student rollouts   │
│  • Implement strict action-token masking (ignoring SQLite sandbox observation tokens)       │
│                                                                                             │
│  PHASE 3: Privileged Teacher Formulation (Week 3)                                           │
│  • Implement `PrivilegedSchemaTeacher` providing full DDL context to teacher forward passes  │
│  • Verify memory efficiency using PEFT LoRA base model weight sharing                       │
│                                                                                             │
│  PHASE 4: Controlled Empirical PoC & Benchmark Evaluation (Week 4)                          │
│  • Execute 5-condition PoC on Spider 1.0 Dev split                                          │
│  • Measure Test-Suite Accuracy, Zero-Gated Token Ratio, and Stall Step Elimination          │
│  • Integrate findings into the AskTheDatabase White Paper                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

1. Lin, W., Zhao, J., Jiang, X., Rao, S., Li, Y., Wang, S., He, B., & Huang, G. (2026). *On-Policy Distillation with Verifiable Reward*. arXiv preprint [arXiv:2608.24696](https://arxiv.org/abs/2608.24696).
2. Guo, D., et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv preprint arXiv:2501.12948.
3. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.
4. Yang, H., Zhang, J., Qian, Z., Xie, J., Gong, S., Guo, Y., & Han, S. (2026). *HarnessSQL: Harness-Native Training for SQL Agents in Realistic Database Environments*. arXiv preprint arXiv:2610.12274.
5. AskTheDatabase Team (2026). *AskTheDatabase: Internalized Schema Exploration via Procedural Harness Distillation and Verifiable Multi-Turn Trajectory Verification*. Technical White Paper Draft.
