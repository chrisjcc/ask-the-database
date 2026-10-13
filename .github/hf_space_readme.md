---
title: Ask The Database Demo
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# AskTheDatabase 🤖

> Multi-turn reinforcement learning with verifiable rewards for natural language to SQL translation across relational databases.

Translating complex natural language questions into executable, semantically accurate SQL across unfamiliar schemas is fraught with hallucinations and query errors. Instead of generating fragile queries in a single greedy step, **AskTheDatabase** operates as an autonomous agent that explores the database schema, samples table records, dry-runs trial queries in a secure sandbox, and self-corrects execution errors before returning verified SQL.

## 🚀 Application Details
This Hugging Face Space runs an interactive Streamlit UI deployed via a Docker container. It demonstrates the multi-turn agent exploration pipeline against relational databases.

### Features
- **Database Selection & Schema DDL Drawer:** Select across Spider relational databases and inspect raw DDL table schemas and foreign key relationships.
- **Multi-Turn Autonomous Exploration:** Watch the agent formulate hypotheses, inspect schemas (`inspect_schema`), sample categorical records (`sample_data`), and execute trial queries (`execute_sql`).
- **Verifiable Execution & Self-Correction:** Experience how intermediate runtime errors are diagnosed and recovered to produce correct SQL.
- **Interactive Result Preview:** View and inspect final executed query tabular results rendered directly in interactive dataframes.

For more information, research methodology, training pipelines, and evaluation benchmarks, please refer to the [main GitHub repository](https://github.com/chrisjcc/ask-the-database).
