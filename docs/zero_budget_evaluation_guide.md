# Zero-Budget Evaluation Harness & Golden Test Fixtures Guide

**Document Version**: 1.0.0  
**Target Subsystem**: Zero-Budget Evaluation Harness & Golden Sanity Verification  
**Location**: `docs/zero_budget_evaluation_guide.md`  

---

## 1. Executive Summary & Purpose

The **Zero-Budget Evaluation Harness** provides sub-second, zero-cloud-token evaluation for SQL generation pipelines in **AskTheDatabase**. It runs locally on developer CPUs, in GitHub Actions CI pipelines, and within Git pre-commit hooks to verify model and pipeline integrity before running expensive multi-GPU training or full benchmark evaluations.

The harness relies on **three lightweight, curated golden test fixtures (~33 KB total)** stored in `data/fixtures/`. These files are **strictly required to be tracked by the Git repository** to prevent test failures, CI breakages, or pipeline stalls across distributed environments and fresh clones.

---

## 2. Golden Test Fixtures Breakdown (`data/fixtures/`)

| File Path | Size | Description & Strategic Role |
| :--- | :--- | :--- |
| **`data/fixtures/seed_fixtures.sql`** | ~7.2 KB | **Base Deterministic SQLite Database**: Contains complete DDL schema definitions and seed data covering relational foreign keys, junction tables, nullable attributes, and boundary values. Used to instantiate fresh in-memory SQLite instances. |
| **`data/fixtures/seed_fixtures_mutant.sql`** | ~6.6 KB | **Mutant Evaluation State**: A controlled mutation-perturbed variant of the base database (injected with edge-case row deletions, boundary mutations, and constraint adjustments) used to evaluate test-suite mutation catching and false-positive elimination. |
| **`data/fixtures/smoke_cases.jsonl`** | ~19.3 KB | **Curated Error-Trap Golden Suite**: 35 hand-crafted benchmark queries with gold SQL answers, reference execution outputs, and explicit error-trap attributions across 5 core failure modes (7 cases per trap category). |

### Error-Trap Taxonomy Coverage (35 Cases)
Every case in `smoke_cases.jsonl` targets one of five pervasive SQL error modes:
1. `join_fanout`: Improper join predicates causing cartesian explosions or row duplications.
2. `dialect_drift`: Non-standard SQL syntax incompatible with SQLite (e.g. `ILIKE`, date syntax).
3. `hallucinated_column`: Referencing non-existent table attributes or misaligned column aliases.
4. `aggregation_without_grouping`: Un-grouped column projections alongside aggregate functions (`COUNT`, `SUM`, `AVG`).
5. `null_boundary`: Misinterpreting `NULL` values in `WHERE` clauses, three-valued logic, or `NOT IN (NULL)`.

---

## 3. Git Tracking & `.gitignore` Architecture

### Why Large Data is Ignored While Fixtures are Tracked
AskTheDatabase uses large external datasets (such as raw Spider databases, Spider 2.0-DBT dumps, and processed SFT corpora) which reside in `data/` and must never be committed to Git.

To enforce this, `.gitignore` uses an explicit wildcard exclusion:
```gitignore
# Project Artifacts and Hydra outputs
/data/*
!/data/fixtures/
!/data/fixtures/**
```

> [!IMPORTANT]
> **Never untrack `data/fixtures/`!**  
> Running commands like `git rm -r --cached data/` without excluding `data/fixtures/` will delete the golden test fixtures from the Git index. Fresh checkouts, GitHub Actions CI runners, and remote cluster nodes (e.g., DESY) will then fail with `FileNotFoundError: data/fixtures/smoke_cases.jsonl`.

---

## 4. System Dependency Graph & Operational Workflows

The golden fixtures in `data/fixtures/` are actively required across four distinct operational execution points:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA FIXTURES DEPENDENCY MAP                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   data/fixtures/seed_fixtures.sql                                                      │
│   data/fixtures/seed_fixtures_mutant.sql                                               │
│   data/fixtures/smoke_cases.jsonl                                                      │
│          │                                                                             │
│          ├──► [1] Git Pre-Commit Hook (.git/hooks/pre-commit)                          │
│          │        `python scripts/run_smoke_eval.py --golden` (< 25 ms sanity gate)     │
│          │                                                                             │
│          ├──► [2] Pytest Unit Test Suite (tests/unit/test_smoke_fixtures.py)           │
│          │        Validates schema integrity, in-memory isolation & trap coverage      │
│          │                                                                             │
│          ├──► [3] Zero-Budget Harness Engine (src/.../zero_budget_harness.py)          │
│          │        In-process micro-benchmarking and remediation decision tables        │
│          │                                                                             │
│          └──► [4] CI Pipeline & Remote Training Nodes (.github/workflows/ci.yml)       │
│                   Guarantees deterministic execution across headless runners & servers │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Git Pre-Commit Hook
Every commit executed in this repository automatically triggers `.git/hooks/pre-commit`:
```bash
python scripts/run_smoke_eval.py --golden
```
This executes all 35 cases against in-memory SQLite instances in ~25 milliseconds. If any case fails, the commit is aborted, guaranteeing that no broken SQL generation logic enters the repository.

### 2. Pytest Unit Testing
The test module `tests/unit/test_smoke_fixtures.py`:
- Asserts that all three files exist on disk.
- Asserts that all 35 cases load valid JSON and parse into valid SQLite queries.
- Tests the mutant seed database to verify that mutant catching logic operates deterministically.
- Verifies that loading and execution take $< 1.0$ second.

### 3. Core Zero-Budget Harness Runner
In Python code, `ZeroBudgetHarness`:
```python
from ask_the_database.evaluation.zero_budget_harness import ZeroBudgetHarness

harness = ZeroBudgetHarness()  # Automatically resolves data/fixtures/ paths
report = harness.run_golden_verification()
assert report.overall_pass_rate == 1.0
```

---

## 5. Maintenance & Restoration

If `data/fixtures/` is ever accidentally untracked or deleted:

1. **Verify `.gitignore` exception**:
   Ensure `.gitignore` contains:
   ```gitignore
   /data/*
   !/data/fixtures/
   !/data/fixtures/**
   ```
2. **Force-add the fixtures to Git**:
   ```bash
   git add -f data/fixtures/seed_fixtures.sql data/fixtures/seed_fixtures_mutant.sql data/fixtures/smoke_cases.jsonl
   ```
3. **Verify via smoke eval**:
   ```bash
   python scripts/run_smoke_eval.py --golden
   ```
