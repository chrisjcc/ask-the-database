# AskTheDatabase — Phase 1 Granular Implementation Plan
## Repository Foundation, SSoT Architecture & Database Sandboxing (Issues #1 to #5)

---

## 1. Phase 1 Overview & Scope

The objective of **Phase 1** is to construct the rock-solid, verifiable foundation of **`ask-the-database`**. Following the Single Source of Truth (SSoT) principles and lessons learned from `AskBeforeAnswer`, Phase 1 focuses entirely on:
1. Setting up the framework-agnostic repository skeleton, linting, typing, and configuration layers.
2. Establishing the deterministic DVC pipeline and artifact tracking structure.
3. Implementing the secure, read-only, ephemeral **SQLite Sandbox** with AST-level mutation blocking and hard execution timeouts.
4. Implementing automated ingestion and normalization of benchmark datasets (`NovaSky-AI/SkyRL-SQL-653-data-newfmt` and `xlangai/spider`).
5. Authoring a comprehensive unit test suite to rigorously verify sandbox safety and isolation.

> [!IMPORTANT]
> **Zero GPU Requirement for Phase 1**: All code in Phase 1 is 100% CPU-compatible, executing purely on standard Python 3.11+, `sqlite3`, `sqlglot`, `duckdb`, and `hydra-core`. It is developed and validated entirely on macOS local environments before cluster staging.

---

## 2. Directory Layout & Target File Inventory

```text
ask-the-database/
├── .env.example
├── .gitignore
├── .dvcignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── dvc.yaml
├── params.yaml
│
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── spider.yaml
│   │   └── skyrl_sql_653.yaml
│   └── environment/
│       └── sqlite_sandbox.yaml
│
├── data/
│   ├── raw/
│   ├── databases/
│   └── processed/
│
├── src/
│   └── ask_the_database/
│       ├── __init__.py
│       ├── environment/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sqlite_env.py
│       │   └── schema_manager.py
│       └── data/
│           ├── __init__.py
│           ├── downloader.py
│           └── preprocessor.py
│
├── scripts/
│   └── preprocess_data.py
│
└── tests/
    ├── conftest.py
    └── unit/
        ├── test_sandbox.py
        └── test_schema_manager.py
```

---

## 3. Granular Issue-by-Issue Implementation Specifications

---

### Issue #1: Repository Skeleton, SSoT Toolchain & Base Configurations

#### 1. Objectives & Responsibilities
- Establish the packaging metadata in `pyproject.toml` with strict `ruff` formatting/linting rules and `mypy` type checking.
- Lock exact runtime dependencies in `requirements.txt`.
- Author the root `Makefile` enforcing workflow commands (`make install`, `make lint`, `make test`, `make format`).
- Establish the root Hydra configuration (`configs/config.yaml`) ensuring strict hierarchical separation.

#### 2. Exact File Contracts

##### A. `pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ask-the-database"
version = "0.1.0"
description = "Multi-turn reinforcement learning with verifiable rewards for natural language to SQL translation."
readme = "README.md"
authors = [{ name = "Christian Contreras-Campana" }]
requires-python = ">=3.11"
dependencies = [
    "hydra-core>=1.3.2",
    "omegaconf>=2.3.0",
    "pydantic>=2.7.0",
    "sqlglot>=25.0.0",
    "duckdb>=1.1.0",
    "datasets>=3.0.0",
    "huggingface-hub>=0.25.0",
    "python-dotenv>=1.0.1",
    "tqdm>=4.66.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
    "types-tqdm",
    "types-requests"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
exclude = ["build/", "dist/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --strict-markers"
```

##### B. `Makefile`
```makefile
-include .env
export

# Dynamic Environment Isolation
export PYTHONNOUSERSITE := 1

ifneq ($(CONDA_PREFIX),)
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
endif

.PHONY: help install lint format test preprocess

help:
	@echo "AskTheDatabase - Operational Commands:"
	@echo "  make install     Install production & development dependencies"
	@echo "  make lint        Run ruff linter and mypy type checks"
	@echo "  make format      Auto-format code via ruff"
	@echo "  make test        Execute pytest test suite"
	@echo "  make preprocess  Download datasets and run normalization pipeline"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check src/ scripts/ tests/
	mypy src/

format:
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

test:
	pytest tests/

preprocess:
	python scripts/preprocess_data.py
```

##### C. `configs/config.yaml`
```yaml
defaults:
  - environment: sqlite_sandbox
  - data: spider
  - _self_

project_name: "ask-the-database"
seed: 42

paths:
  root_dir: ${oc.env:PWD}
  data_dir: ${paths.root_dir}/data
  raw_dir: ${paths.data_dir}/raw
  databases_dir: ${paths.data_dir}/databases
  processed_dir: ${paths.data_dir}/processed
```

##### D. `configs/environment/sqlite_sandbox.yaml`
```yaml
type: "sqlite"
timeout_seconds: 3.0
memory_limit_mb: 512
max_result_rows: 500
read_only: true
enforce_ast_safety: true
forbidden_tokens:
  - "DROP"
  - "DELETE"
  - "INSERT"
  - "UPDATE"
  - "ALTER"
  - "CREATE"
  - "ATTACH"
  - "DETACH"
  - "REINDEX"
  - "VACUUM"
```

#### 3. Definition of Done & Pass/Fail Verification Gate
Execute the verification suite locally:
```bash
make install
make format
make lint
python -c "import ask_the_database; print('AskTheDatabase imported successfully!')"
```
*Pass Criteria*: Clean imports, zero Ruff errors, zero Mypy type violations.

---

### Issue #2: DVC Pipeline & Remote Storage Setup

#### 1. Objectives & Responsibilities
- Configure `dvc.yaml` tracking the end-to-end data preparation stage (`preprocess`).
- Configure `params.yaml` mirroring Hydra parameters to establish the DVC dependency DAG.
- Configure `.dvcignore` to prevent staging SQLite ephemeral locks, logs, or temporary caches.

#### 2. Exact File Contracts

##### A. `params.yaml`
```yaml
preprocess:
  dataset_name: "spider"
  include_skyrl_653: true
  max_tokens: 2048
  train_split: 0.9
  val_split: 0.1
  seed: 42
```

##### B. `dvc.yaml`
```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess_data.py
    deps:
      - scripts/preprocess_data.py
      - src/ask_the_database/data/preprocessor.py
      - src/ask_the_database/data/downloader.py
      - configs/data/spider.yaml
    params:
      - preprocess.dataset_name
      - preprocess.include_skyrl_653
      - preprocess.seed
    outs:
      - data/processed/train.jsonl
      - data/processed/val.jsonl
```

##### C. `.dvcignore`
```gitignore
*.sqlite-journal
*.sqlite-wal
*.sqlite-shm
data/raw/
__pycache__/
*.pyc
.DS_Store
```

#### 3. Definition of Done & Pass/Fail Verification Gate
```bash
dvc doctor
dvc status
```
*Pass Criteria*: DVC initializes cleanly and reports `preprocess` stage correctly configured.

---

### Issue #3: SQLite Ephemeral Sandbox Implementation

#### 1. Objectives & Responsibilities
- Define the framework-agnostic `BaseSQLEnvironment` abstract interface.
- Implement `SQLiteEnvironment` providing:
  1. **Strict Read-Only Connection**: Using URI mode `file:{path}?mode=ro`.
  2. **AST Safety Interception**: Using `sqlglot` to parse AST, verify root is strictly `SELECT` or `WITH`, and reject any DDL/DML mutation attempts.
  3. **Execution Timeout**: Enforcing a strict 3.0-second walltime cutoff using Python thread/signal timers.
  4. **Memory/Row Bounding**: Capping query returns via `fetchmany(max_result_rows)` to prevent buffer exhaustion.
- Implement `SchemaManager` extracting table DDLs, column data types, and foreign key graphs for prompting.

#### 2. Exact File Contracts

##### A. `src/ask_the_database/environment/base.py`
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Standardized result returned by the database sandbox."""
    success: bool
    data: list[tuple[Any, ...]] | None
    columns: list[str] | None
    row_count: int
    execution_time_ms: float
    error_message: str | None
    error_type: str | None  # e.g., "SYNTAX", "SECURITY", "TIMEOUT", "EXECUTION"


class BaseSQLEnvironment(ABC):
    """Abstract interface for database execution sandboxes."""

    @abstractmethod
    def execute(self, query: str) -> ExecutionResult:
        """Execute a SQL query within the sandboxed environment."""
        pass

    @abstractmethod
    def get_schema_summary(self) -> str:
        """Return a formatted string representing the schema for LLM prompting."""
        pass
```

##### B. `src/ask_the_database/environment/sqlite_env.py`
Key technical capabilities implemented in this class:
- `_verify_ast_safety(query: str) -> None`: Parses with `sqlglot.parse_one(query, read="sqlite")`. Traverses nodes. If any node is an instance of `exp.Insert`, `exp.Update`, `exp.Delete`, `exp.Drop`, `exp.Create`, or `exp.Alter`, raises `SQLSecurityViolationError`.
- `execute(query: str) -> ExecutionResult`:
  - Opens connection via `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`.
  - Executes `PRAGMA query_only = ON;`.
  - Runs with a `concurrent.futures.ThreadPoolExecutor` with a timeout limit of `timeout_seconds`.
  - Captures runtime database exceptions (`sqlite3.OperationalError`, `sqlite3.DatabaseError`) and packages them cleanly into `ExecutionResult`.

##### C. `src/ask_the_database/environment/schema_manager.py`
- Connects to target `.sqlite` database.
- Queries `PRAGMA table_info(table_name)` for column names and types.
- Queries `PRAGMA foreign_key_list(table_name)` to build relational graphs.
- Generates standard markdown/DDL schema representations:
  ```sql
  CREATE TABLE singer (
      singer_id INTEGER PRIMARY KEY,
      name TEXT,
      country TEXT,
      song_name TEXT,
      song_release_year TEXT,
      age INTEGER,
      is_male BOOLEAN
  );
  ```

#### 3. Definition of Done & Pass/Fail Verification Gate
Run a scratch test verifying AST blocking and read-only execution against an ephemeral SQLite DB.
*Pass Criteria*: `SELECT * FROM table` succeeds; `DROP TABLE table` raises security error; recursive CTE `WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT * FROM cnt;` terminates at exactly 3.0s with `TIMEOUT` error type.

---

### Issue #4: Benchmark Data Ingestion & Preprocessing

#### 1. Objectives & Responsibilities
- Implement automated downloader fetching Spider (`xlangai/spider`) and SkyRL-SQL-653 (`NovaSky-AI/SkyRL-SQL-653-data-newfmt`) datasets and SQLite database binaries from Hugging Face Hub (`seeklhy/OmniSQL-datasets`).
- Implement the normalization pipeline in `src/ask_the_database/data/preprocessor.py` converting raw entries into a unified JSONL schema.
- Assert every sample is grounded: verify that its `db_id.sqlite` file physically exists and that the gold query executes cleanly in the SQLite sandbox during ingestion.

#### 2. Exact Data Schema Contract

Each processed JSON line in `data/processed/train.jsonl` and `data/processed/val.jsonl` must strictly adhere to this JSON schema:

```json
{
  "instance_id": "spider_train_0001",
  "source": "spider",
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "gold_sql": "SELECT count(*) FROM singer",
  "db_path": "data/databases/concert_singer/concert_singer.sqlite",
  "schema_ddl": "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, ...);",
  "difficulty": "easy",
  "gold_row_count": 1,
  "gold_execution_hash": "a1b2c3d4..."
}
```

#### 3. Executable Script Specification (`scripts/preprocess_data.py`)
- Reads Hydra configuration from `configs/data/spider.yaml`.
- Downloads missing `.sqlite` databases into `data/databases/<db_id>/<db_id>.sqlite`.
- Parses question-SQL pairs.
- Validates the gold query using `SQLiteEnvironment(db_path).execute(gold_sql)`.
- If gold SQL fails or returns an execution error, the sample is flagged and logged.
- Writes deterministic splits to `data/processed/train.jsonl` and `data/processed/val.jsonl`.

#### 4. Definition of Done & Pass/Fail Verification Gate
```bash
make preprocess
```
*Pass Criteria*: 
1. `data/databases/` contains extracted SQLite database folders.
2. `data/processed/train.jsonl` and `data/processed/val.jsonl` exist and contain verified, runnable question-SQL pairs.
3. 100% of gold queries execute successfully with 0 runtime exceptions.

---

### Issue #5: Sandbox Unit & Security Tests

#### 1. Objectives & Responsibilities
- Implement a comprehensive, hermetic pytest test suite in `tests/unit/test_sandbox.py` and `tests/unit/test_schema_manager.py`.
- Mock nothing for SQLite: create temporary in-memory or fixture SQLite databases on disk (`tmp_path`) with known schemas and data.
- Prove sandbox security: test mutation prevention, SQL injection bypasses, timeout killing, and memory limits.

#### 2. Exact Test Matrix (`tests/unit/test_sandbox.py`)

| Test Function | Input Query Scenario | Expected Outcome |
| :--- | :--- | :--- |
| `test_valid_select_query` | Simple `SELECT * FROM users WHERE age > 21` | `result.success == True`, correct rows returned. |
| `test_valid_join_aggregation` | Multi-table join with `GROUP BY` and `HAVING` | `result.success == True`, correct column names and data types. |
| `test_block_drop_table` | `DROP TABLE users;` | `result.success == False`, `error_type == "SECURITY"`. Table remains intact. |
| `test_block_delete_from` | `DELETE FROM users WHERE id = 1;` | `result.success == False`, `error_type == "SECURITY"`. Zero rows deleted. |
| `test_block_stacked_injection` | `SELECT * FROM users; DROP TABLE users;` | `result.success == False`, `error_type == "SECURITY"`. |
| `test_block_nested_cte_mutation` | `WITH cte AS (SELECT 1) INSERT INTO users VALUES (99);` | `result.success == False`, `error_type == "SECURITY"`. |
| `test_enforce_timeout` | Infinite loop: `WITH RECURSIVE cnt(x) AS (...)` | Process terminates at $3.0 \pm 0.2\text{s}$, `error_type == "TIMEOUT"`. |
| `test_result_row_limit` | Query returning 10,000 rows when limit is 500 | `result.row_count == 500`, no memory overflow. |
| `test_schema_extraction` | SQLite DB with composite primary keys and foreign keys | Schema DDL contains all tables, keys, and column types. |

#### 3. Definition of Done & Pass/Fail Verification Gate
```bash
pytest tests/unit/ -v --cov=src/ask_the_database
```
*Pass Criteria*: 100% pass rate across all tests with $>90\%$ code coverage on `src/ask_the_database/environment/`.

---

## 4. Execution Workflow: Immediate Next Step

Once you review and approve this implementation plan, we will immediately execute **Issue #1**:
1. Scaffold the root repository files (`pyproject.toml`, `Makefile`, `configs/config.yaml`, `configs/environment/sqlite_sandbox.yaml`).
2. Run `make install` and `make lint` to verify clean toolchain setup.
3. Commit and advance to Issue #2.
