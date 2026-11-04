---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
- Qwen/Qwen2.5-Coder-7B-Instruct
datasets:
- chrisjcc/ask-the-database
tags:
- text-to-sql
- reinforcement-learning
- rlvr
- harness-zero
- spider
- verifiable-rewards
- sqlite
- zero-budget-eval
- qwen2.5-coder
---

# AskTheDatabase: Multi-Turn Verifiable Text-to-SQL 🤖🗄️

> Autonomous relational schema exploration and natural language to SQL translation with verifiable execution rewards.

Translating complex natural language questions into accurate SQL across unfamiliar schemas is fraught with schema hallucinations, join fanouts, and dialect drift. Instead of generating brittle queries in a single greedy step, **AskTheDatabase** operates as an interactive agent that explores schemas, samples table records, dry-runs trial queries in a secure sandbox, and self-corrects execution errors.

---

## Model Details

- **Model Name:** `chrisjcc/ask-the-database`
- **Model Variant:** `harness_zero_grpo` (Harness-Zero GRPO Policy (Multi-Turn RLVR with Verifiable Rewards))
- **Base Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Training Paradigm:** Multi-Turn Agentic Exploration with Verifiable Environment Feedback
- **Primary Dataset:** `chrisjcc/ask-the-database`
- **License:** Apache 2.0

---

## Key Capabilities & Interactive Tools

The model is trained to interleave reasoning thoughts with procedural tool calls:
1. `\schema [table_name]`: Inspect exact DDL schemas, primary keys, and foreign key relations.
2. `\sample [table_name] [column_name]`: Inspect distinct categorical values to prevent value hallucination.
3. `\sql [query]`: Execute trial queries against an in-process SQLite sandbox to inspect execution plans and catch runtime errors before final commitment.

---

## Empirical Benchmark & Decision Gate Results

### 1. Benchmark Execution Accuracy
| Benchmark | Total Samples | Execution Accuracy (EX) | Test Suite Accuracy (TS) | Valid Execution Rate (VA) |
| :--- | :--- | :--- | :--- | :--- |
| **Spider Dev** | 1034 | **20.99%** | **19.92%** | **30.56%** |


### 2. Pre-Registered Zero-Budget MVE Decision Gates (Live Cluster Telemetry)
The model was verified against the project's pre-registered Zero-Budget Evaluation Harness decision gates:
- **Gate 1 (Latency Budget):** 0.1268 s (126.80 ms) across 5 passes (target: ≤ 3.0 s) → **PASSED**
- **Gate 2 (Rank Correlation):** Spearman ρ = 1.0000 vs. Spider Dev (target: ρ ≥ 0.88) → **PASSED**
- **Gate 3 (Diagnostic Resolution):** 100.0% attribution, 100.0% hallucination reduction (10 → 0 misses) → **PASSED**
- **Gate 4 (Mutation Sensitivity):** 100.0% bug detection rate, 0.0% escape rate across dual fixtures → **PASSED**

**Official Verdict:** `MVE ADJUDICATION DECISION: GO`

---

## Quickstart & Inference Usage

### Using the High-Level Pipeline
```python
from ask_the_database.inference.pipeline import TextToSQLPipeline

pipeline = TextToSQLPipeline(
    model_name_or_path="chrisjcc/ask-the-database",
    db_path="data/databases/concert_singer/concert_singer.sqlite",
)

result = pipeline.run("How many singers are from France?")
print(f"Generated SQL: {result['final_sql']}")
print(f"Execution Output: {result['execution_result']}")
```

### Direct Transformers Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("chrisjcc/ask-the-database")
model = AutoModelForCausalLM.from_pretrained(
    "chrisjcc/ask-the-database",
    torch_dtype="auto",
    device_map="auto",
)

prompt = "<|im_start|>user\nDatabase: concert_singer\nQuestion: What is the highest stadium capacity?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Intended Use & Safety Sandbox
- **Target Use:** Translating complex business and analytical natural language questions into accurate, executable SQLite queries.
- **Sandboxing Guardrails:** All candidate queries run inside an isolated sandbox with query execution timeouts (< 2.0 s) and query plan inspection (`EXPLAIN QUERY PLAN`) to flag and reject unconstrained Cartesian joins (`SCAN TABLE ... WITHOUT INDEX`).
