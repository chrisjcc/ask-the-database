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
        preprocess collect-distillation dvc-repro \
        train-sft train-filtered train-harness-zero train-grpo \
        sweep \
        smoke-eval evaluate-zero-budget-mve evaluate evaluate-mve compare-rlvr evaluate-verifier-mve \
        generate-hf-cards publish-hf-cards publish-hf-release \
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
	@echo "  make preprocess               Download and normalize Spider benchmarks into Parquet"
	@echo "  make collect-distillation     Collect K_SQL reviewed trajectories for Harness-Zero"
	@echo "  make dvc-repro                Reproduce pipeline stages tracked in dvc.yaml"
	@echo ""
	@echo "Model Training & Alignment:"
	@echo "  make train-sft                Train SFT baseline on standard Spider gold pairs (Cond 2)"
	@echo "  make train-filtered           Train SFT on successful rollouts via rejection sampling (Cond 3)"
	@echo "  make train-harness-zero       Train SFT on K_SQL reviewed trajectories (Cond 4 - Proposed)"
	@echo "  make train-grpo               Execute multi-turn GRPO / RLVR verifiable training (Cond 5)"
	@echo ""
	@echo "Hyperparameter Optimization (W&B Sweeps):"
	@echo "  make sweep                    Launch W&B Bayesian hyperparameter sweep (FINE_TUNE_METHOD=<sft|grpo>)"
	@echo ""
	@echo "Evaluation & Decision Gates:"
	@echo "  make smoke-eval               Execute fast in-process zero-budget smoke check (< 38ms)"
	@echo "  make evaluate-zero-budget-mve Adjudicate 4-gate Zero-Budget Evaluation Micro-Harness"
	@echo "  make evaluate                 Evaluate full benchmark suites (Spider, DK, Syn, Realistic)"
	@echo "  make evaluate-mve             Adjudicate 4-condition MVE decision gates across 500 episodes"
	@echo "  make compare-rlvr             Compare sample efficiency & asymptotic accuracy (Run A vs B)"
	@echo "  make evaluate-verifier-mve    Adjudicate LLM-as-a-Verifier MVE decision gates & publish report"
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

preprocess:
	$(PYTHON) scripts/preprocess_data.py $(ARGS)

collect-distillation:
	$(PYTHON) scripts/collect_reviewed_trajectories.py $(ARGS)

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

evaluate-mve:
	$(PYTHON) scripts/run_mve_evaluation.py $(ARGS)

compare-rlvr:
	$(PYTHON) scripts/compare_rlvr_efficiency.py $(ARGS)

evaluate-verifier-mve:
	$(PYTHON) scripts/run_verifier_mve.py $(ARGS)

# ------------------------------------------------------------
# 6. Deployment & Hugging Face Hub
# ------------------------------------------------------------

generate-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --card-only $(ARGS)

publish-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --publish-cards $(ARGS)

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
