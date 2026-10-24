# Stop Guessing the Schema: Why LLMs Need Internalized Exploration and How Harness Distillation Solves Text-to-SQL

### Teaching language models to autonomously inspect database catalogs, eliminate cold-start RL collapse, and scale relational reasoning with compound verification.

---

Every developer who has built an autonomous Text-to-SQL agent has witnessed this frustrating failure mode:

A user asks a straightforward business question:  
> *"Show me the top 5 customers with recurring monthly subscriptions who haven't logged in over the last 30 days."*

The model confidently emits a syntactically pristine SQL query. It features subqueries, window functions, and proper aliases. It looks magnificent—until you execute it against the database:

```sql
SELECT c.customer_id, c.customer_name, s.mrr
FROM customers c
JOIN subscriptions s ON c.id = s.customer_id
WHERE s.plan_type = 'monthly'
  AND c.last_login_date < DATE('now', '-30 days')
ORDER BY s.mrr DESC
LIMIT 5;
```

`OperationalError: no such column: c.id`  
`OperationalError: no such column: c.last_login_date`

The model hallucinated column names. In reality, the table used `account_id` instead of `id`, stored login timestamps in an `auth_events` audit table, and represented plan durations as an integer enum rather than the string `'monthly'`.

Why did this happen? Because modern frontier models have been trained on static text-to-SQL corpora where the prompt contains an abbreviated schema snippet, and the target is a single-turn gold SQL string. When deployed against real-world relational databases containing dozens of tables and hundreds of attributes, the model doesn't know how to **explore**. It guesses based on surface linguistic associations.

In this deep dive, we present **AskTheDatabase**, an end-to-end framework that solves this fundamental bottleneck. We explain:
1. Why conventional agent harnesses and prompt scaffolding are too slow and expensive for interactive production systems.
2. How **Harness-Zero** procedural distillation transfers multi-turn exploratory habits directly into model parameters without gold query leakage.
3. How internalized exploration solves the **zero-variance cold-start plateau** in Reinforcement Learning with Verifiable Rewards (RLVR / GRPO).
4. Why standalone test-time verification is **strictly Pareto-dominated**, and how pairing it with Harness-Zero establishes a state-of-the-art **Compound Architecture** reaching **61.8% Test-Suite accuracy** on Spider.
5. How we built an in-memory **Zero-Budget Evaluation Micro-Harness** that executes in **126.8 ms** with a perfect monotonic correlation ($\rho = 1.000$) to full benchmark evaluations.

---

## The Three Dilemmas of Production Text-to-SQL

Building autonomous data agents reveals three stark engineering trade-offs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       The Text-to-SQL Trilemma                              │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ 1. The Exploration Dilemma     │ Single-turn guessing hallucinates schemas;  │
│                               │ Multi-turn harnesses incur 5-10x latency.   │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 2. The Verification Dilemma   │ Production queries have no ground truth;    │
│                               │ Ungrounded candidate pools fool verifiers.  │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 3. The Evaluation Bottleneck  │ Full test-suite suites take 3-6 minutes;    │
│                               │ Slow CI cycles paralyze agent development.  │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### 1. The Multi-Turn Latency Tax
The standard industry response to schema hallucination has been **agentic prompting harnesses** like ReAct, Reflexion, or DIN-SQL. You equip the model with tool definitions (`inspect_schema`, `sample_data`, `execute_sql`) and let it loop iteratively.

While these harnesses improve accuracy, they introduce a massive **latency and compute tax**. A 5-turn trajectory requires 5 sequential round-trips to the LLM. In user-facing analytics, waiting 4 to 8 seconds for a dashboard query is unacceptable.

### 2. The Survivorship Bias of Rejection Sampling
A common alternative is *rejection sampling* (best-of-$N$ filtering): sample trajectories from a base student, execute them, discard the failures, and fine-tune on the successful rollouts.

However, rejection sampling suffers from severe **survivorship bias**. By discarding every failed rollout, the model never observes *how to recover* from errors. When it encounters an unexpected SQLite operational exception in production, it lacks the procedural muscle memory to diagnose the issue and falls into repetitive retry loops.

### 3. The Cold-Start Gradient Collapse in RLVR
When training multi-turn agents with Group Relative Policy Optimization (GRPO), the policy gradient is driven by intra-group advantage:

$$A_i = \frac{R_i - \operatorname{mean}(\{R_j\}_{j=1}^G)}{\operatorname{std}(\{R_j\}_{j=1}^G) + \epsilon}$$

If an ungrounded model explores an unseen database schema without prior exploratory habits, **all $G$ sampled rollouts in the group frequently fail simultaneously** ($R_1 = R_2 = \dots = R_G = -1.0$). 

When every rollout fails:

$$\operatorname{std}(\{R_j\}_{j=1}^G) = 0 \implies A_i = 0 \quad \forall i$$

The advantage vanishes, the policy gradient collapses to zero ($\nabla_\theta \mathcal{J} = 0$), and the model enters an extended exploration plateau where it receives no directional learning signal whatsoever.

---

## Pillar 1: Harness-Zero (Harness Distillation via Agent-as-Harness)

The foundational premise of **Harness-Zero** is simple yet radical:

> **Agentic discipline should be baked into neural weights, not orchestrated through runtime prompt scaffolding.**

```
                      Harness-Zero Distillation Pipeline
                      
  Natural Language Prompt (x)
             │
             ▼
   Student Model (π_θ) ──[ Emits Action a_t ]──► Environment Sandbox
             ▲                                            │
             │           Procedural Reviewer              ▼
             │                (K_SQL)              Observation o_t
             │         [Zero Gold Leakage!]               │
             │                   │                        │
             └──[ Corrective Action z_t ]◄────────────────┘
                                 │
                                 ▼
                     Repaired Trajectory T_rep
                                 │
                                 ▼
                    Standard SFT on Agent Steps
                    (External Reviewer Discarded)
```

### Answer Distillation vs. Harness Distillation
Most distillation approaches perform *answer distillation*: a giant teacher model observes the ground-truth SQL query and guides the student directly to the correct output.

As established by Ye et al. (2026), answer distillation produces fragile policies. When a teacher has oracle access to the answer, it teaches the student non-generalizable shortcuts. In contrast, when the teacher enforces strictly **procedural guidance ($\mathcal{K}$)** with zero access to the gold query, downstream test generalization doubles.

In AskTheDatabase, the procedural reviewer $\mathcal{K}_{\text{SQL}}$ monitors student rollouts in real time and enforces four deterministic protocols:

1. **`RequireSchemaInspection`**: If the student attempts to generate SQL querying `orders` and `line_items` without first calling `inspect_schema(["orders", "line_items"])`, $\mathcal{K}_{\text{SQL}}$ intercepts the call and forces catalog inspection.
2. **`RequireRowSampling`**: If a query filters on a categorical string column (e.g. `WHERE status = 'Active'`), $\mathcal{K}_{\text{SQL}}$ intercepts the submission and requires a `sample_data("orders", limit=3)` call to inspect actual string casing and formatting.
3. **`EnforceDryRunBeforeFinish`**: Any query emitted as `final_answer` must have passed at least one trial execution via `execute_sql` without runtime exceptions.
4. **`DiagnosticRecovery`**: When an execution fails with an SQLite error (e.g., `no such column`), $\mathcal{K}_{\text{SQL}}$ blocks repeated blind retries and forces a targeted schema lookup.

### Three Fundamental Distillation Invariants
1. **Zero Gold Leakage**: $\mathcal{K}_{\text{SQL}}$ has zero access to `gold_sql`. It operates strictly on database catalogs and execution exceptions.
2. **Action Space Invariance**: All interventions occur strictly within the student's native toolset (`inspect_schema`, `sample_data`, `execute_sql`, `final_answer`).
3. **Zero Deployment Overhead**: Once fine-tuned on the repaired trajectories, $\mathcal{K}_{\text{SQL}}$ is **completely discarded**. At inference time, the student model executes in standard autoregressive greedy mode.

---

## Pillar 2: Resolving the RLVR Cold-Start Plateau

With procedural exploration internalized into the base weights, we align the agent using multi-turn Reinforcement Learning with Verifiable Rewards (RLVR / GRPO).

Because Harness-Zero pre-fine-tuning elevates the model's schema pre-inspection rate from **46.2% to 84.6%**, at least one rollout in every GRPO sampling group ($G=4$) successfully inspects the catalog and finds valid joins from Step 1.

```
                  GRPO Intra-Group Reward Variance
                  
  Conventional SFT Rollouts:
  [Fail: -1.0]  [Fail: -1.0]  [Fail: -1.0]  [Fail: -1.0]  ──► std(R) = 0.0  ──► Advantage = 0 (STALL)
  
  Harness-Zero Pre-Grounded Rollouts:
  [Fail: -0.4]  [Syntax: +0.2]  [TS Match: +1.0]  [Err: -0.5]  ──► std(R) = 0.68 ──► High Policy Gradient!
```

This guarantees divergent intra-group returns ($\operatorname{std}(R) > 0$), providing high-variance advantage estimates that eliminate the cold-start exploration plateau.

### The Hierarchical Multi-Tier Reward Rubric
To prevent reward hacking, we implement a strict 4-tier verifiable reward rubric:

$$\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{syntax}} + \mathcal{R}_{\text{schema}} + \mathcal{R}_{\text{TS}} + \mathcal{R}_{\text{efficiency}}$$

* **Tier 1: Syntax Validity ($\mathcal{R}_{\text{syntax}} \in [-1.0, 0.0]$)**: AST validation via SQLGlot. Syntax errors receive $-1.0$.
* **Tier 2: Schema Conformance ($\mathcal{R}_{\text{schema}} \in [-0.5, +0.2]$)**: Penalizes hallucinated columns ($-0.5$) and awards $+0.2$ for exception-free execution.
* **Tier 3: Test-Suite Mutation Equivalence ($\mathcal{R}_{\text{TS}} \in [0.0, 1.0]$)**: To eliminate spurious matches where a broken query accidentally returns the same table on small databases, candidate queries must match the reference query across $K$ mutated database states:

$$\mathcal{R}_{\text{TS}} = \prod_{k=0}^K \mathbb{I} \left( Q_{\text{gen}}(\mathcal{D}_k) \equiv Q_{\text{gold}}(\mathcal{D}_k) \right)$$

* **Tier 4: Efficiency & Autonomous Recovery ($\mathcal{R}_{\text{efficiency}}$)**: Imposes a turn discount penalty ($-0.05 \times t$) to prevent infinite loops, while awarding a $+0.3$ recovery bonus when a runtime error is resolved in a subsequent turn.

---

## Pillar 3: LLM-as-a-Verifier & The Pareto Frontier

In production, there is no ground-truth oracle ($\text{gold\_sql}$). If you sample 5 candidate queries and all 5 execute without errors but produce different result sets, how do you decide which one satisfies the user's intent?

We incorporated the **LLM-as-a-Verifier** framework, which replaces noisy categorical ratings with **continuous logit expectation** over normalized score vocabularies, combined with a **Probabilistic Pivot Tournament** that reduces pairwise comparison costs by up to 70%.

However, our empirical investigations revealed a profound architectural lesson:

### Why Standalone Verification Fails: The Quality Bottleneck
What happens when you deploy an LLM-as-a-Verifier directly over candidate queries generated by an ungrounded baseline model?

A verifier is fundamentally a **re-ranker and selector**. It cannot invent SQL out of thin air; it can only select the best candidate from the pool.

When the generator is an ungrounded baseline model that never learned procedural catalog inspection, **all 5 sampled candidates often hallucinate schema columns simultaneously**. The verifier is placed in an impossible situation: forced to choose the "least bad" broken query.

```
                       The Candidate Pool Bottleneck
                       
  Ungrounded SFT Generator (5 Candidates)
  ┌─────────────────────────────────────────────────────────────────┐
  │ Candidate 1: Hallucinates column 'c.last_login'                 │
  │ Candidate 2: Joins through non-existent foreign key 'order_ref' │
  │ Candidate 3: Cartesian fanout cross-join                        │
  │ Candidate 4: Syntax error in HAVING clause                      │
  │ Candidate 5: Hallucinates column 'c.signup_date'                │
  └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
               LLM-as-a-Verifier: Forced to pick "least bad"
                                   │
                                   ▼
               Result: 56.2% TS @ 1,120 ms & 5.8× Compute
```

### The Pareto Dominance of Harness-Zero
When we evaluated Standalone Verification against pure Harness-Zero on held-out Spider Dev benchmarks, the results were unequivocal:

* **Accuracy Advantage**: Pure Harness-Zero achieves **58.4% TS**, outperforming the Standalone Verifier (**56.2% TS**) by **+2.2 pp**.
* **Speed & Latency Advantage**: Harness-Zero responds in **350 ms**, which is **3.2× faster** than the Standalone Verifier (**1,120 ms**).
* **Compute & Token Advantage**: Harness-Zero operates with **1.0× compute**, making it **5.8× cheaper** than sampling 5 rollouts and running tournament verification (**5.8× compute**).

Because Harness-Zero is simultaneously **more accurate, 3.2× faster, and 5.8× cheaper**, Standalone Verification is **strictly Pareto-dominated**.

```
  Accuracy (TS %)
      │
 65%  │                                      ● Point D: Compound Architecture (61.8%)
      │                                        [Harness-Zero + Verifier]
 60%  │                  ● Point C: Harness-Zero (58.4%)
      │                    [Pareto-Dominant Real-Time]
 55%  │                                      ▲ Point B: Standalone Verifier (56.2%)
      │                                        [Pareto-Dominated!]
 50%  │   ● Point A: Baseline SFT (51.6%)
      │
      └─────────────────────────────────────────────────────────────► Compute Cost
         1.0×                                5.9×
```

### Where the Verifier Wins: The Compound Regime
Does this mean test-time verification is useless? **No!**

When we deploy the verifier atop **Harness-Zero** (the **Compound Architecture**), the dynamics invert completely:

1. **Harness-Zero's Role**: Ensures that candidate generation is grounded from turn 1. Because the model autonomously executes `inspect_schema` and `sample_data`, all 5 candidates navigate valid tables and foreign keys.
2. **The Verifier's Role**: Operating over a candidate pool of structurally sound queries, the verifier exercises its true superpower: breaking subtle ties in `WHERE` clause boundaries, aggregate granularity (`COUNT` vs. `SUM`), and three-valued logic edge cases.

In the compound regime, performance jumps to **61.8% Test-Suite accuracy**—an additive **+3.4 pp boost** over Harness-Zero alone, and a massive **+10.2 pp boost** over the baseline SFT model!

---

## Pillar 4: The Zero-Budget Evaluation Micro-Harness

Evaluating models across 1,034 benchmark queries on Spider Dev requires between 3 and 6 minutes of GPU cluster compute over disk-bound SQLite files. 

For developers iterating on agent prompts, loss functions, or reward rubrics, a 6-minute feedback cycle is paralyzing.

To solve this, we engineered the **Zero-Budget Evaluation Micro-Harness**:

```
                       Zero-Budget Micro-Harness
                       
               35 In-Memory SQLite Test Fixtures (:memory:)
              ┌───────────────────────────────────────────┐
              │ State A: Canonical Benchmark Relational   │
              │ State B: Adversarial Mutant State         │
              │          (Nulls, inverted PKs, boundary)  │
              └───────────────────────────────────────────┘
                                   │
                       Dual-Fixture Execution Gate
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
     Execution Verification                   Diagnostic Trap Profiler
     Match across both states?                AST & Plan Classification:
     (Eliminates 100% of coincidences)        • hallucinated_column
                                              • aggregation_without_grouping
                                              • dialect_drift
                                              • join_fanout
                                              • null_boundary
                                   │
                                   ▼
             Total Run Time: 126.8 ms (23.4 ms in Pre-Commit)
             Spearman Rank Correlation: ρ = 1.000 vs. Spider Dev
```

### Key Engineering Features:
* **Dual In-Memory Fixtures**: Loaded into SQLite `:memory:`. Each fixture pairs a base relational state with an adversarial mutant state (inserting null boundaries, duplicating foreign keys, or inverting numerical distributions). This eliminates 100% of coincidental query passes.
* **Sub-Second Execution**: Evaluates 35 cases across 5 passes in **126.8 ms** total CPU time (averaging **0.66 ms per case**).
* **Perfect Monotonic Correlation**: Achieves a perfect Spearman rank correlation ($\rho = 1.000$) with Spider Dev benchmark standings. If a checkpoint improves on the micro-harness, it improves on Spider.
* **Pre-Commit Gate (< 0.2s)**: Packaged directly into a native git pre-commit hook (`make setup-hooks`). Before any code is committed, the entire test suite and error-attribution classifier run in 23 ms!

---

## Comprehensive Empirical Benchmark Results

We evaluated Qwen2.5-Coder-7B-Instruct across five conditions on $N=500$ held-out episodes of Spider Dev (20 unseen schemas) and Spider-Realistic:

| Metric | Condition 1: Base Student | Condition 2: Baseline SFT | Condition 3: Rejection Sampling | Condition 4: Harness-Zero SFT | Condition 5: Compound (H-0 + Verifier) | $\Delta$ (C5 vs C2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spider Dev Test-Suite (TS)** | 38.2% | 51.6% | 53.9% | **58.4%** | **61.8%** | **+10.2 pp** |
| **Execution Accuracy (EX)** | 44.1% | 57.3% | 59.5% | **64.2%** | **67.5%** | **+10.2 pp** |
| **Valid SQL Rate** | 89.5% | 94.8% | 95.4% | **97.1%** | **99.2%** | **+4.4 pp** |
| **Error Recovery ($R_{\text{rec}}$)** | 14.2% | 22.8% | 26.5% | **38.4%** | **38.4%** | **+68.4% rel** |
| **Schema Pre-Inspection Rate** | 31.0% | 46.2% | 52.8% | **84.6%** | **84.6%** | **+38.4 pp** |
| **Median Latency** | **280 ms** | **310 ms** | 335 ms | **350 ms** | 1,180 ms | +870 ms |
| **Compute Multiplier** | **1.0×** | **1.0×** | 1.0× | **1.0×** | 5.9× | +4.9× |
| **Spider-Realistic TS** | 31.5% | 42.1% | 44.7% | **51.2%** | **54.9%** | **+12.8 pp** |

### Stratification by Difficulty
On complex queries containing nested subqueries, compound `HAVING` filters, and multi-table joins, the performance margin widens dramatically:

* **Hard Queries**: 48.0% (Baseline SFT) $\to$ **59.8% (Compound)** (**+11.8 pp**)
* **Extra Hard Queries**: 35.0% (Baseline SFT) $\to$ **48.2% (Compound)** (**+13.2 pp**)

---

## Production Deployment: The Dual-Regime Architecture

Because AskTheDatabase disentangles internalized grounding from test-time verification, it offers operators an elegant deployment choice:

```
                          Production Dispatcher
                                    │
                    What is the query SLA requirement?
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          Real-Time Interactive            High-Accuracy Batch
          (User Chat / Dashboards)         (Reporting / Financial Audits)
                    │                               │
                    ▼                               ▼
           Pure Harness-Zero               Compound Architecture
           • Latency: 350 ms               • Latency: 1,180 ms
           • Compute: 1.0×                 • Compute: 5.9×
           • Accuracy: 58.4% TS            • Accuracy: 61.8% TS
```

1. **Real-Time Interactive Serving**: Use **Pure Harness-Zero**. In a single forward pass ($350\,\text{ms}$, $1.0\times$ compute), the model delivers grounded, schema-accurate SQL for live user interactions.
2. **High-Accuracy Offline Batch**: Use the **Compound Architecture**. For scheduled reporting, ETL audits, and executive analytics, Best-of-5 tournament verification pushes accuracy to **61.8% TS**, eliminating subtle filter discrepancies.

---

## Key Takeaways for AI Practitioners

1. **Prompt Scaffolding Belongs in Model Weights**: Multi-turn agent loops are wonderful for data collection, but terrible for inference latency. Distilling procedural exploration rules into weights gives you multi-turn intelligence at single-turn speed.
2. **Never Distill Answers When You Can Distill Procedures**: Giving teachers oracle answer access teaches students non-generalizable shortcuts. Distilling procedural protocols ($\mathcal{K}$) with zero gold leakage forces models to learn genuine relational problem-solving.
3. **Verification Requires Grounded Candidates**: A test-time verifier cannot save an ungrounded generator. Standalone verification over ungrounded rollouts is strictly Pareto-dominated; verification only compounds value when the candidate pool is already schema-valid.
4. **Pre-Grounding Eliminates the RLVR Cold-Start**: If your RL policy gradients are vanishing due to zero reward variance, don't just tweak the learning rate—warm-start the policy with procedural distillation so rollouts explore successfully from Step 1.
5. **Invest in In-Memory Micro-Harnesses**: Waiting minutes for benchmark results destroys developer velocity. A sub-second in-memory fixture suite running in pre-commit hooks transforms agent engineering into a rapid, test-driven science.

---

## Try It Yourself

The AskTheDatabase project is fully open-source. Code, dataset processing scripts, training configurations, and evaluation suites are available on GitHub and Hugging Face:

* 🌐 **GitHub Repository**: [chrisjcc/ask-the-database](https://github.com/chrisjcc/ask-the-database)
* 📚 **Research Paper**: [AskTheDatabase White Paper](https://github.com/chrisjcc/ask-the-database/blob/main/papers/draft/main.pdf)
* 🤗 **Hugging Face Space Demo**: [AskTheDatabase Interactive Demo](https://huggingface.co/spaces/chrisjcc/ask-the-database-demo)
* 🤗 **Hugging Face Model Card**: [chrisjcc/ask-the-database](https://huggingface.co/chrisjcc/ask-the-database)
* 🤗 **Hugging Face Dataset**: [AskTheDatabase Data](https://huggingface.co/datasets/chrisjcc/ask-the-database)

---

## References

1. **Ye, H., et al. (2026)**. *Harness-Zero: Distilling Agentic Scaffolds into Model Parameters via Procedural Guidance*. arXiv preprint.
2. **Yu, T., Zhang, R., Yang, K., et al. (2018)**. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. In *EMNLP*.
3. **Zhong, R., Choi, E., et al. (2020)**. *Semantic Evaluation for Text-to-SQL with Test Suite Based Execution*. In *EMNLP*.
4. **Nova, A., et al. (2025)**. *SkyRL-SQL: Multi-Turn Relational Reinforcement Learning with Verifiable Rewards*. arXiv preprint.
5. **Shao, Z., Wang, P., et al. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
6. **Yao, S., Zhao, J., Yu, D., et al. (2023)**. *ReAct: Synergizing Reasoning and Acting in Language Models for General Task Solving*. In *ICLR*.
7. **Shinn, N., Cassano, F., Labash, B., et al. (2024)**. *Reflexion: Language Agents with Verbal Reinforcement Learning*. In *NeurIPS*.
8. **Pourreza, M., & Rafiei, D. (2024)**. *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction*. In *NeurIPS*.
9. **Contreras Campana, C. (2026)**. *AskTheDatabase: Harness Distillation and Verifiable Feedback for Multi-Turn Data Agents with Zero-Budget Evaluation*. Technical Report.
10. **Toby Mao. (2023)**. *SQLGlot: An In-Python SQL Parser, Transpiler, and Optimizer*. GitHub repository.
