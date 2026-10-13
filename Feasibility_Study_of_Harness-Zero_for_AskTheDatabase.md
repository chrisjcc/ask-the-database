# Task: Feasibility Study of Harness-Zero for AskTheDatabase

## Objective

Conduct a rigorous, evidence-based feasibility study to determine whether the research presented in **Harness-Zero: Harness Distillation via Agent-as-Harness** can improve the AskTheDatabase (`ask-the-database`) project, which investigates post-training large language models for natural-language-to-SQL translation using supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards (RLVR).

The goal is to identify actionable research directions, assess their technical feasibility and expected value, and propose a minimal, reproducible experiment to validate the most promising approach.

## Research sources

Read and cross-reference all three primary sources before drawing conclusions:

1. **Research paper:** https://huggingface.co/papers/2609.24974
2. **arXiv manuscript:** https://arxiv.org/pdf/2609.24974
3. **Official implementation:** https://github.com/metaevo-ai/harness-zero

Inspect the paper's methodology, experimental setup, results, limitations, and appendices where available. Examine the official repository's source code, training pipeline, data structures, evaluation procedures, configuration, and dependencies to distinguish the actual implementation from a high-level description.

Also inspect the current AskTheDatabase (`ask-the-database`) project repository, including its research documentation, implementation plan, experiment configurations, reward functions, datasets, evaluation framework, and existing training pipeline. Treat the repository's current state as the source of truth for project-specific details. Do not assume that proposed features have already been implemented.

If the AskTheDatabase repository is not available in the current environment, identify the missing access explicitly and distinguish verified findings from assumptions. Do not fabricate implementation details.

## Research questions

### 1. What does Harness-Zero actually demonstrate?

Explain the research problem, central hypothesis, algorithm, and experimental findings in sufficient technical detail to evaluate whether the method transfers to natural-language-to-SQL tasks.

In particular, investigate the following distinction:

**Does Harness-Zero primarily teach a smaller model by replacing an incorrect answer with a larger model's correct answer, or does it teach the student to internalize the capabilities of an optimized agent harness?**

Establish precisely what is corrected during training, what information the harnessing agent receives, what the student observes, which trajectories are retained, how supervised fine-tuning is performed, and which components are removed at deployment.

Clarify the relationship between:
- Conventional teacher-student knowledge distillation.
- Supervised fine-tuning on teacher-generated correct answers.
- Rejection sampling and filtering successful trajectories.
- Critique-and-revision or response correction.
- Agent-harness distillation, where tool-use strategies, execution procedures, memory, skills, and other harness-induced behaviors are transferred into model weights.
- Reinforcement learning with execution-based or otherwise verifiable rewards.

Determine whether Harness-Zero uses a larger model simply to provide ground-truth answers, or whether its key contribution is to transform the behavior of an optimized harness into training demonstrations that are valid in the student's target action space.

Use concrete examples from the paper and implementation to support the explanation.

### 2. What are the paper's most important findings?

Extract the principal experimental results and explain what they establish, rather than merely repeating the authors' claims.

At a minimum, investigate the reported results:

- Base-model macro-average task success: 23.3%.
- Base model with the evolved, specialized harness: 41.7%.
- Distilled model under the minimal target harness: 44.3%.
- Agent-as-harness versus code-as-harness: 81.1% versus 78.1% average performance on the reported comparison.
- Recovery of harness-exclusive behavior patterns: 82.3% across 28 patterns.

Verify these numbers and their definitions against the original paper and repository. Identify the relevant benchmarks, student model, evaluation metrics, experimental conditions, and comparisons.

Explain the significance of the distilled model outperforming the base model with the specialized harness still attached. Also discuss what the results do not establish, including potential limitations in statistical confidence, generalization, training cost, evaluation scope, and reproducibility.

Do not treat these results as evidence that Harness-Zero will necessarily improve text-to-SQL accuracy.

### 3. How does the research map to AskTheDatabase?

Inspect the actual AskTheDatabase implementation and determine which parts of Harness-Zero are directly applicable, which require adaptation, and which are unlikely to provide meaningful benefits.

Evaluate at least the following potential applications.

**A. SQL generation and correction**

Use a stronger model, an optimized SQL agent, or a reference harness to inspect the student's proposed SQL before execution. Where appropriate, correct the SQL, its execution strategy, or the sequence of actions to produce a better trajectory in the student's available action space.

Investigate whether this could teach the student to avoid recurring SQL errors, choose appropriate joins, handle schema ambiguity, use aggregation correctly, and recover from execution failures.

Distinguish correction of SQL syntax from correction of semantics and from correction of the overall agent's decision-making process.

**B. Harness distillation for database agents**

Determine what an optimized SQL-agent harness could contribute beyond a correct final SQL query. Consider schema inspection, table and column selection, query planning, tool selection, execution-feedback interpretation, iterative debugging, retry policies, and stopping criteria.

Assess which of these behaviors can be represented in the student's training trajectories and learned through supervised fine-tuning.

Determine whether the current AskTheDatabase action space and agent architecture can support this approach without major redesign.

**C. Execution-based rewards and RLVR**

Assess how Harness-Zero could complement the existing or planned execution-based reward pipeline.

Consider using corrected trajectories as supervised fine-tuning data before reinforcement learning, or using them to initialize a stronger policy for subsequent RLVR training.

Compare these approaches with training directly from execution rewards, filtering successful rollouts, and conventional teacher-generated SQL demonstrations.

Do not assume Harness-Zero itself is an RLVR algorithm. Explain the methodological distinction and evaluate whether combining the two methods is justified.

**D. Preference optimization and failure-driven training**

Determine whether corrected and uncorrected student responses could yield useful preference pairs or failure-analysis data.

Assess whether this would add value beyond existing SFT, DPO/ORPO, or GRPO experiments, and whether it introduces risks such as reward hacking, learning superficial correction patterns, or overfitting to the reference model.

**E. Deployment implications**

Determine which capabilities could plausibly be internalized in model weights and which still require external infrastructure, such as a SQL execution engine, database schema access, runtime validation, sandboxing, or access controls.

Assess whether the main practical benefit would be improved task success, reduced inference-time overhead, fewer model calls, more reliable recovery, simpler deployment, or some combination of these.

Explicitly distinguish removing a specialized reasoning or orchestration harness from removing essential database execution and safety infrastructure.

### 4. Design a technically credible adaptation

Propose a concrete adaptation of Harness-Zero for AskTheDatabase.

Define:

1. **Student model:** A suitable model already used by the project, or a justified alternative.
2. **Reference harness or teacher:** The source of improved SQL-agent behavior and how its guidance will be obtained.
3. **Target harness:** The fixed action space and runtime interface used by the student during training and deployment.
4. **Training tasks:** Existing text-to-SQL datasets, database environments, and task splits that are suitable for the experiment.
5. **Correction mechanism:** How the reference agent reviews student responses, decides whether to accept or replace them, and generates valid student-action-space demonstrations.
6. **Trajectory construction:** Which student actions, corrections, tool observations, execution results, and intermediate reasoning traces are retained or masked.
7. **Training objective:** The proposed SFT procedure and its relationship to subsequent RLVR or preference optimization.
8. **Evaluation:** Metrics, baselines, ablations, and held-out tests required to establish a meaningful improvement.
9. **Compute and cost:** Estimated teacher inference, student rollout, training, storage, and evaluation requirements, with explicit assumptions.
10. **Reproducibility:** Seeds, configuration management, data provenance, experiment tracking, and protection against benchmark leakage.

Pay particular attention to the fact that a corrected final SQL query is not necessarily sufficient supervision for an agent that must learn multi-step database interaction. Determine what trajectory representation is required for the intended capability.

Avoid leaking test answers or allowing the reference harness to access information unavailable to the student at inference time, except where that difference is explicitly part of the method being evaluated. Document any privileged teacher information and how it is used.

### 5. Design the minimum viable experiment

Recommend the smallest experiment that can establish whether harness distillation is promising before committing substantial GPU resources or restructuring the project.

Where feasible, compare the following conditions using the same base model, training tasks, evaluation split, and comparable training budgets:

1. **Baseline:** The existing student model and current inference-time harness.
2. **SFT baseline:** Conventional SFT on available ground-truth SQL or successful trajectories.
3. **Harness-Zero-style SFT:** SFT on student trajectories corrected by the reference harness, using the method's key design principles.
4. **Successful-rollout filtering:** SFT on successful student trajectories, to distinguish the effect of correction from the effect of selecting successful examples.
5. **Optional RLVR continuation:** Apply the project's existing execution-based RL training to the baseline and/or distilled checkpoint to determine whether the gains complement reinforcement learning.

Adapt this comparison to the project's actual implementation and resource constraints. Clearly mark optional experiments and avoid unnecessary duplication.

Measure, as appropriate:

- Execution accuracy and exact-match accuracy.
- Semantic correctness of query results.
- Execution success rate and SQL syntax error rate.
- Performance on unseen databases and schemas.
- Generalization to compositional and difficult queries.
- Recovery from execution errors.
- Number of tool calls, model calls, tokens, latency, and inference cost.
- Robustness to ambiguous or incomplete requests.
- Performance with the specialized reference harness removed.
- Statistical uncertainty across seeds and evaluation samples.

Use execution-based correctness as a primary signal where the benchmark permits it. Account for SQL queries that differ syntactically but produce equivalent results, database state, nondeterminism, and potentially unsafe statements.

State the success criteria before running experiments. Recommend a go/no-go decision rule based on measurable gains, computational cost, and robustness, rather than choosing thresholds after observing the results.

### 6. Assess limitations and research risks

Discuss the most relevant threats to validity:

- Teacher-model errors and correlated teacher/student failures.
- Incorrect corrections that nevertheless appear plausible.
- Distribution shift between the reference harness and the student's deployment environment.
- Training data contamination and benchmark leakage.
- Overfitting to specific databases, schemas, or query templates.
- Distillation of superficial response patterns instead of generalizable problem-solving strategies.
- Loss of useful exploration behavior caused by training only on corrected trajectories.
- Cost of teacher-assisted rollouts and correction.
- Differences between the paper's action space and the database agent's action space.
- Whether the reported gains survive across multiple seeds and model sizes.

Determine which risks can be addressed through the proposed experiment and which remain open research questions.

### 7. Inspect the implementation and ecosystem

Inspect the official Harness-Zero repository and determine:

- Which components can be reused directly.
- Which are tightly coupled to its current benchmarks, harness abstraction, rollout infrastructure, or training provider.
- Whether the code can be adapted without adopting its entire framework.
- Licensing and attribution requirements.
- Python and framework compatibility.
- Maintenance, dependency, and integration risks.

Prefer minimal, modular reuse over adding an entire new framework unless the benefits justify the complexity.

For AskTheDatabase, preserve the project's existing separation of concerns, reproducible configurations, experiment tracking, and single source of truth. Avoid introducing duplicated configuration or a second experiment-management system.

Clearly distinguish repository facts from recommendations and assumptions.

## Required deliverable

Create a Markdown report named:

`docs/research/harness-zero-feasibility-study.md`

If the repository uses a different established documentation structure, follow its conventions and state the actual output path.

The report must be a standalone research document suitable for future implementation planning and for generating actionable GitHub issues.

Include these sections:

1. **Executive summary**
2. **Research question and scope**
3. **Paper summary and core methodological contribution**
4. **Clarification: answer distillation versus harness distillation**
5. **Key findings and quantitative results**
6. **Relevance to AskTheDatabase**
7. **Proposed adaptation and architecture**
8. **Minimum viable experimental plan**
9. **Baselines, metrics, and ablations**
10. **Implementation feasibility and dependencies**
11. **Compute, engineering effort, and expected costs**
12. **Limitations, risks, and threats to validity**
13. **Go/no-go recommendation**
14. **Proposed follow-up experiments and implementation issues**
15. **References**

Use Markdown tables where they make comparisons clearer. Include a proposed experiment matrix with hypotheses, controlled variables, metrics, and decision criteria. Add a Mermaid diagram if it materially clarifies the proposed training and deployment workflow.

Cite the paper, official repository, and any additional sources using direct URLs. Cite specific claims and numerical findings close to the relevant discussion. Separate results reported by the paper from findings established by inspecting AskTheDatabase and from hypotheses proposed by the agent.

Do not fabricate measurements, benchmark results, runtime estimates, or implementation details. When information is unavailable, label it as unknown and explain how to obtain it.

## Expected conclusion

The report must give an explicit answer to the following question:

**Is Harness-Zero a promising research direction for AskTheDatabase, and what is the strongest reason to adopt or reject it?**

The recommendation must distinguish between:

- Using a stronger model to generate correct SQL answers.
- Using a stronger model or optimized harness to correct the student's actions and produce better training trajectories.
- Distilling the behavior of an optimized SQL-agent harness into model weights.
- Combining harness-distilled SFT with execution-based RLVR.

If the approach is promising, identify the single highest-value experiment to run first, its required changes, its expected evidence of success, and the conditions under which the project should stop or expand the investigation.

## Execution instructions

- Inspect the actual repositories and source materials before making project-specific recommendations.
- Prefer the primary paper and official implementation over secondary summaries.
- Do not modify the existing training pipeline or launch expensive experiments unless explicitly authorized.
- Do not create speculative implementation changes merely to complete the study.
- Save the completed report to the specified Markdown path.
- Validate the Markdown structure, links, and any Mermaid syntax where tooling permits.
- Finish with a concise summary of the report's location, the principal conclusion, and the recommended next experiment.
