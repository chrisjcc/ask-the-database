-include .env
export

# Dynamic Environment Isolation
export PYTHONNOUSERSITE := 1

ifneq ($(CONDA_PREFIX),)
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
endif

# ------------------------------------------------------------
# Environment & Tooling Resolution
# ------------------------------------------------------------

VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)
RUFF := $(if $(wildcard $(VENV)/bin/ruff),$(VENV)/bin/ruff,ruff)
MYPY := $(if $(wildcard $(VENV)/bin/mypy),$(VENV)/bin/mypy,mypy)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
DVC := $(if $(wildcard $(VENV)/bin/dvc),$(VENV)/bin/dvc,dvc)
STREAMLIT := $(if $(wildcard $(VENV)/bin/streamlit),$(VENV)/bin/streamlit,streamlit)

# Deployment & Container Configuration
APP_IMAGE ?= ask-the-database-app
MODEL_REPO ?= chrisjcc/ask-the-database
DATASET_REPO ?= chrisjcc/ask-the-database

.PHONY: help \
        install install-dev lint format test setup-hooks \
        download-spider1 preprocess collect-distillation download-spider2-dbt convert-spider2-dbt test-spider2-dbt \
        download-spider2-lite prepare-spider2-sqlite dvc-repro \
        train-sft train-filtered train-harness-zero train-grpo train-dapo train-gspo \
        sweep \
        smoke-eval evaluate-zero-budget-mve evaluate evaluate-spider2 test-spider2-eval evaluate-mve compare-rlvr evaluate-verifier-mve \
        compile-paper \
        generate-hf-cards publish-hf-cards publish-hf-model publish-hf-dataset publish-hf-release \
        run-app docker-build docker-run

# ------------------------------------------------------------
# Operational Help Command
# ------------------------------------------------------------

help:
	@echo "================================================================================"
	@echo "                      AskTheDatabase Operational Commands                       "
	@echo "================================================================================"
	@echo "Environment & Quality:"
	@echo "  make install                  Install all project dependencies (training, dev, UI)"
	@echo "  make install-dev              Install minimal development dependencies only"
	@echo "  make lint                     Run ruff linter and mypy static type checks"
	@echo "  make format                   Auto-format codebase via ruff"
	@echo "  make test                     Execute pytest unit and integration test suite"
	@echo "  make setup-hooks              Install native git pre-commit hook (< 0.2s gate)"
	@echo ""
	@echo "Data & Trajectory Distillation:"
	@echo "  make download-spider1         Download and extract Spider 1.0 SQLite databases"
	@echo "  make preprocess               Download and normalize Spider benchmarks into JSONL"
	@echo "  make collect-distillation     Collect K_SQL reviewed trajectories for Harness-Zero"
	@echo "  make download-spider2-dbt     Download Spider 2.0-DBT DuckDB enterprise databases from HF"
	@echo "  make convert-spider2-dbt      Convert Spider 2.0-DBT DuckDB enterprise databases to SQLite"
	@echo "  make test-spider2-dbt         Execute Spider 2.0-DBT pipeline unit and parity tests"
	@echo "  make download-spider2-lite    Download Spider 2.0-Lite SQLite databases and gold CSVs"
	@echo "  make prepare-spider2-sqlite   Prepare and validate 135 Spider 2.0-SQLite evaluation tasks"
	@echo "  make dvc-repro                Reproduce pipeline stages tracked in dvc.yaml"
	@echo ""
	@echo "Model Training & Alignment:"
	@echo "  make train-sft                Train SFT baseline on standard Spider gold pairs (Cond 2)"
	@echo "  make train-filtered           Train SFT on successful rollouts via rejection sampling (Cond 3)"
	@echo "  make train-harness-zero       Train SFT on K_SQL reviewed trajectories (Cond 4 - Proposed)"
	@echo "  make train-grpo               Execute multi-turn GRPO / RLVR verifiable training (Cond 5)"
	@echo "  make train-dapo               Execute multi-turn DAPO enterprise training (Cond 6)"
	@echo "  make train-gspo               Execute multi-turn GSPO enterprise training (Cond 7)"
	@echo ""
	@echo "Hyperparameter Optimization (W&B Sweeps):"
	@echo "  make sweep                    Launch W&B Bayesian hyperparameter sweep (FINE_TUNE_METHOD=<sft|grpo>)"
	@echo ""
	@echo "Evaluation & Decision Gates:"
	@echo "  make smoke-eval               Execute fast in-process zero-budget smoke check (< 38ms)"
	@echo "  make evaluate-zero-budget-mve Adjudicate 4-gate Zero-Budget Evaluation Micro-Harness"
	@echo "  make evaluate                 Evaluate full benchmark suites (Spider, DK, Syn, Realistic)"
	@echo "  make evaluate-spider2         Evaluate Spider 2.0-SQLite enterprise benchmark"
	@echo "  make test-spider2-eval        Execute unit test suite for Spider 2.0 evaluation"
	@echo "  make evaluate-mve             Adjudicate 4-condition MVE decision gates across 500 episodes"
	@echo "  make compare-rlvr             Compare sample efficiency & asymptotic accuracy (Run A vs B)"
	@echo "  make evaluate-verifier-mve    Adjudicate LLM-as-a-Verifier MVE decision gates & publish report"
	@echo ""
	@echo "Paper & Documentation:"
	@echo "  make compile-paper            Compile ICML white paper draft into PDF (papers/draft/main.pdf)"
	@echo ""
	@echo "Deployment & Hugging Face Hub:"
	@echo "  make generate-hf-cards        Generate Model Card & Dataset READMEs locally (--card-only)"
	@echo "  make publish-hf-cards         Publish Model & Dataset README cards to Hugging Face Hub"
	@echo "  make publish-hf-release       Deploy verified model checkpoints & datasets to Hugging Face"
	@echo ""
	@echo "Interactive Demo App & Docker:"
	@echo "  make run-app                  Launch local interactive Streamlit demo UI"
	@echo "  make docker-build             Build Streamlit Docker container image"
	@echo "  make docker-run               Run AskTheDatabase interactive demo in Docker"
	@echo "================================================================================"

# ------------------------------------------------------------
# 1. Environment & Code Quality
# ------------------------------------------------------------

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src/ scripts/ tests/ app/
	$(RUFF) format --check src/ scripts/ tests/ app/
	$(MYPY) src/

format:
	$(RUFF) check --fix src/ scripts/ tests/ app/
	$(RUFF) format src/ scripts/ tests/ app/

test:
	$(PYTEST) tests/

setup-hooks:
	@echo "Installing native git pre-commit hook..."
	@mkdir -p .git/hooks
	@printf '#!/usr/bin/env bash\nset -e\nif [ -d .venv/bin ]; then export PATH="$$PWD/.venv/bin:$$PATH"; fi\necho "[pre-commit] Running ruff lint & format checks..."\nruff check src/ scripts/ tests/ app/\nruff format --check src/ scripts/ tests/ app/\necho "[pre-commit] Running zero-budget evaluation smoke gate (< 3.0s)..."\npython scripts/run_smoke_eval.py --golden\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git pre-commit hook installed successfully! (Runs ruff and make smoke-eval in < 0.2s before each commit)"

# ------------------------------------------------------------
# 2. Data Preprocessing & Trajectory Distillation
# ------------------------------------------------------------

download-spider1:
	$(PYTHON) scripts/download_spider1.py $(ARGS)

preprocess:
	$(PYTHON) scripts/preprocess_data.py $(ARGS)

collect-distillation:
	$(PYTHON) scripts/collect_reviewed_trajectories.py $(ARGS)

download-spider2-dbt:
	$(PYTHON) scripts/download_spider2_dbt.py $(ARGS)

convert-spider2-dbt:
	$(PYTHON) scripts/convert_spider2_dbt.py $(ARGS)

test-spider2-dbt:
	$(PYTHON) -m pytest tests/unit/test_dbt_sqlite.py $(ARGS)

download-spider2-lite:
	$(PYTHON) scripts/download_spider2_lite.py $(ARGS)

prepare-spider2-sqlite:
	$(PYTHON) scripts/prepare_spider2_sqlite.py $(ARGS)

dvc-repro:
	$(DVC) repro $(ARGS)

# ------------------------------------------------------------
# 3. Model Training & Alignment
# ------------------------------------------------------------

train-sft:
	$(PYTHON) scripts/train_sft.py $(ARGS)

train-filtered:
	$(PYTHON) scripts/train_sft.py training.dataset_path=data/distillation/filtered_rollouts.jsonl training.output_dir=models/mve/filtered_rollouts $(ARGS)

train-harness-zero:
	$(PYTHON) scripts/train_sft.py training.dataset_path=data/distillation/reviewed_trajectories.jsonl training.output_dir=models/sft/harness_zero $(ARGS)

train-grpo:
	$(PYTHON) scripts/train_rlvr.py training=rlvr_skyrl_agent $(ARGS)

train-dapo:
	$(PYTHON) scripts/train_rlvr.py training=dapo_enterprise $(ARGS)

train-gspo:
	$(PYTHON) scripts/train_rlvr.py training=gspo_enterprise $(ARGS)

# ------------------------------------------------------------
# 4. Hyperparameter Optimization (W&B Sweeps)
# ------------------------------------------------------------

FINE_TUNE_METHODS := sft grpo
COUNT ?= 5

sweep:
	@if [ -z "$(FINE_TUNE_METHOD)" ]; then \
		echo "ERROR: FINE_TUNE_METHOD is required."; \
		echo "Usage: make sweep FINE_TUNE_METHOD=<sft|grpo> [COUNT=<n>]"; \
		exit 1; \
	fi
	@if ! echo "$(FINE_TUNE_METHODS)" | grep -qw "$(FINE_TUNE_METHOD)"; then \
		echo "ERROR: Unsupported fine-tuning method='$(FINE_TUNE_METHOD)'."; \
		echo "Supported fine-tuning methods: $(FINE_TUNE_METHODS)"; \
		exit 1; \
	fi
	@ENTITY=$${WANDB_ENTITY:-$(WANDB_ENTITY)}; \
	if [ -z "$$ENTITY" ]; then \
		echo "ERROR: WANDB_ENTITY is not set. Export WANDB_ENTITY or add to .env"; \
		exit 1; \
	fi; \
	PROJECT=$${WANDB_PROJECT:-$(WANDB_PROJECT)}; \
	if [ -z "$$PROJECT" ]; then \
		echo "ERROR: WANDB_PROJECT is not set. Export WANDB_PROJECT or add to .env"; \
		exit 1; \
	fi; \
	if [ ! -f "sweeps/$(FINE_TUNE_METHOD).yaml" ]; then \
		echo "ERROR: Sweep configuration not found: sweeps/$(FINE_TUNE_METHOD).yaml"; \
		exit 1; \
	fi; \
	echo "=========================================================="; \
	echo "Initializing W&B Sweep"; \
	echo "=========================================================="; \
	echo "Fine-tune method: $(FINE_TUNE_METHOD)"; \
	echo "Sweep config:     sweeps/$(FINE_TUNE_METHOD).yaml"; \
	echo "Entity:           $$ENTITY"; \
	echo "Project:          $$PROJECT"; \
	echo "Trial count:      $(COUNT)"; \
	echo "=========================================================="; \
	OUTPUT=$$($(PYTHON) -m wandb sweep sweeps/$(FINE_TUNE_METHOD).yaml 2>&1) || { \
		echo "$$OUTPUT"; \
		echo "ERROR: Failed to create W&B sweep."; \
		exit 1; \
	}; \
	echo "$$OUTPUT"; \
	SWEEP_ID=$$(echo "$$OUTPUT" | grep -oE "ID: [a-zA-Z0-9]+" | awk '{print $$2}' | tail -1); \
	if [ -z "$$SWEEP_ID" ]; then \
		echo "ERROR: Failed to extract Sweep ID from wandb output."; \
		exit 1; \
	fi; \
	echo "Parsed Sweep ID: $$SWEEP_ID"; \
	echo "Starting W&B sweep agent..."; \
	$(PYTHON) -m wandb agent $$ENTITY/$$PROJECT/$$SWEEP_ID --count $(COUNT) || { \
		echo "ERROR: W&B sweep agent failed."; \
		exit 1; \
	}; \
	echo "Generating sweep report..."; \
	$(PYTHON) scripts/generate_sweep_report.py \
		--fine-tune-method "$(FINE_TUNE_METHOD)" \
		--sweep-id "$$SWEEP_ID"

# ------------------------------------------------------------
# 5. Evaluation & Decision Gates
# ------------------------------------------------------------

smoke-eval:
	$(PYTHON) scripts/run_smoke_eval.py $(ARGS)

evaluate-zero-budget-mve:
	$(PYTHON) scripts/run_zero_budget_mve.py $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate_benchmark.py $(ARGS)

evaluate-spider2:
	$(PYTHON) scripts/evaluate_benchmark.py evaluation=spider2_sqlite $(ARGS)

test-spider2-eval:
	$(PYTHON) -m pytest tests/unit/test_spider2_eval.py $(ARGS)

evaluate-mve:
	$(PYTHON) scripts/run_mve_evaluation.py $(ARGS)

compare-rlvr:
	$(PYTHON) scripts/compare_rlvr_efficiency.py $(ARGS)

evaluate-verifier-mve:
	$(PYTHON) scripts/run_verifier_mve.py $(ARGS)

compile-paper:
	@echo "=========================================================="
	@echo "📄 Compiling ICML white paper draft (papers/draft/main.tex)"
	@echo "=========================================================="
	cd papers/draft && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	@echo "=========================================================="
	@echo "✅ White paper successfully compiled: papers/draft/main.pdf"
	@echo "=========================================================="

# ------------------------------------------------------------
# 6. Deployment & Hugging Face Hub
# ------------------------------------------------------------

generate-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --card-only $(ARGS)

publish-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --publish-cards $(ARGS)

publish-hf-model:
	@echo "=========================================================="
	@echo "🚀 Deploying model card and weights to Hugging Face Model Hub"
	@echo "➔ Model: https://huggingface.co/$(MODEL_REPO)"
	@echo "=========================================================="
	@test -n "$$HF_TOKEN" || ( \
		echo "ERROR: HF_TOKEN environment variable is not set."; \
		echo "Export HF_TOKEN=<your-token> or add it to .env"; \
		exit 1; \
	)
	$(PYTHON) scripts/release_to_hf.py --publish-cards \
		--model-repo $(MODEL_REPO) \
		--dataset-repo $(DATASET_REPO) \
		$(if $(WEIGHTS_DIR),--weights-dir $(WEIGHTS_DIR),) $(ARGS)
	@echo "=========================================================="
	@echo "✅ Model publication to Hugging Face complete."
	@echo "=========================================================="

publish-hf-dataset:
	@echo "=========================================================="
	@echo "🚀 Deploying dataset card and files to Hugging Face Dataset Hub"
	@echo "➔ Dataset: https://huggingface.co/datasets/$(DATASET_REPO)"
	@echo "=========================================================="
	@test -n "$$HF_TOKEN" || ( \
		echo "ERROR: HF_TOKEN environment variable is not set."; \
		echo "Export HF_TOKEN=<your-token> or add it to .env"; \
		exit 1; \
	)
	$(PYTHON) scripts/release_to_hf.py --publish-cards \
		--model-repo $(MODEL_REPO) \
		--dataset-repo $(DATASET_REPO) \
		$(if $(DATASET_DIR),--dataset-dir $(DATASET_DIR),) $(ARGS)
	@echo "=========================================================="
	@echo "✅ Dataset publication to Hugging Face complete."
	@echo "=========================================================="

publish-hf-release:
	@echo "=========================================================="
	@echo "🚀 Deploying verified production model & cards to Hugging Face"
	@echo "➔ Dataset: https://huggingface.co/datasets/$(DATASET_REPO)"
	@echo "➔ Model:   https://huggingface.co/$(MODEL_REPO)"
	@echo "=========================================================="
	@test -n "$$HF_TOKEN" || ( \
		echo "ERROR: HF_TOKEN environment variable is not set."; \
		echo "Export HF_TOKEN=<your-token> or add it to .env"; \
		exit 1; \
	)
	@RELEASE_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0") && \
	echo "Detected release tag: $$RELEASE_TAG" && \
	$(PYTHON) scripts/release_to_hf.py --publish-cards \
		--model-repo $(MODEL_REPO) \
		--dataset-repo $(DATASET_REPO) \
		--release-tag "$$RELEASE_TAG" $(ARGS)
	@echo "=========================================================="
	@echo "✅ Hugging Face deployment complete."
	@echo "=========================================================="

# ------------------------------------------------------------
# 7. Interactive Demo App & Docker
# ------------------------------------------------------------

run-app:
	$(STREAMLIT) run app/app.py $(ARGS)

docker-build:
	docker build -t $(APP_IMAGE) .

docker-run:
	docker run -p 8501:8501 --env-file .env $(APP_IMAGE)
