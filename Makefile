-include .env
export

# Dynamic Environment Isolation
export PYTHONNOUSERSITE := 1

ifneq ($(CONDA_PREFIX),)
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
endif

# Detect virtual environment
VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)
RUFF := $(if $(wildcard $(VENV)/bin/ruff),$(VENV)/bin/ruff,ruff)
MYPY := $(if $(wildcard $(VENV)/bin/mypy),$(VENV)/bin/mypy,mypy)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
DVC := $(if $(wildcard $(VENV)/bin/dvc),$(VENV)/bin/dvc,dvc)
STREAMLIT := $(if $(wildcard $(VENV)/bin/streamlit),$(VENV)/bin/streamlit,streamlit)

APP_IMAGE ?= ask-the-database-app
MODEL_REPO ?= chrisjcc/ask-the-database
DATASET_REPO ?= chrisjcc/ask-the-database

.PHONY: help install install-dev lint format test preprocess train-sft train-filtered train-grpo collect-distillation train-harness-zero evaluate evaluate-mve compare-rlvr smoke-eval evaluate-zero-budget-mve setup-hooks generate-hf-cards publish-hf-cards publish-hf-release docker-build docker-run run-app

help:
	@echo "AskTheDatabase - Operational Commands:"
	@echo "  make install              Install all project dependencies (training, dev, UI)"
	@echo "  make install-dev          Install minimal development dependencies only"
	@echo "  make lint                 Run ruff linter and mypy type checks"
	@echo "  make format               Auto-format code via ruff"
	@echo "  make test                 Execute pytest test suite"
	@echo "  make smoke-eval           Execute fast in-process zero-budget smoke check (< 3.0s)"
	@echo "  make evaluate-zero-budget-mve Adjudicate 3-way Zero-Budget MVE decision gates"
	@echo "  make setup-hooks          Install local git pre-commit hook (runs smoke-eval & lint in < 0.2s)"
	@echo "  make preprocess           Download datasets and run normalization pipeline"
	@echo "  make train-sft            Execute SFT warmup training on base model (Condition 2)"
	@echo "  make train-filtered       Train SFT on filtered successful rollouts (Condition 3)"
	@echo "  make train-harness-zero   Train SFT on K_SQL reviewed trajectories (Condition 4)"
	@echo "  make train-grpo           Execute multi-turn GRPO / RLVR training (Condition 5)"
	@echo "  make collect-distillation Collect K_SQL reviewed trajectories for Harness-Zero"
	@echo "  make evaluate             Evaluate benchmark suites (Spider, DK, Syn, Realistic)"
	@echo "  make evaluate-mve         Adjudicate 4-condition MVE decision gates & publish report"
	@echo "  make compare-rlvr         Compare sample efficiency & asymptotic accuracy (Run A vs Run B)"
	@echo "  make generate-hf-cards    Generate Hugging Face Model & Dataset README cards"
	@echo "  make publish-hf-cards     Publish Model & Dataset README cards to Hugging Face Hub"
	@echo "  make publish-hf-release   Deploy verified model cards & datasets to Hugging Face"
	@echo "  make run-app              Run local interactive Streamlit demo app"
	@echo "  make docker-build         Build Streamlit Docker container image"
	@echo "  make docker-run           Run AskTheDatabase interactive demo in Docker"

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

preprocess:
	$(PYTHON) scripts/preprocess_data.py $(ARGS)

train-sft:
	$(PYTHON) scripts/train_sft.py $(ARGS)

train-filtered:
	$(PYTHON) scripts/train_sft.py training.dataset_path=data/distillation/filtered_rollouts.jsonl training.output_dir=models/mve/filtered_rollouts $(ARGS)

train-grpo:
	$(PYTHON) scripts/train_rlvr.py training=rlvr_skyrl_agent $(ARGS)

collect-distillation:
	$(PYTHON) scripts/collect_reviewed_trajectories.py $(ARGS)

train-harness-zero:
	$(PYTHON) scripts/train_sft.py training.dataset_path=data/distillation/reviewed_trajectories.jsonl training.output_dir=models/sft/harness_zero $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate_benchmark.py $(ARGS)

evaluate-mve:
	$(PYTHON) scripts/run_mve_evaluation.py $(ARGS)

compare-rlvr:
	$(PYTHON) scripts/compare_rlvr_efficiency.py $(ARGS)

smoke-eval:
	$(PYTHON) scripts/run_smoke_eval.py $(ARGS)

evaluate-zero-budget-mve:
	$(PYTHON) scripts/run_zero_budget_mve.py $(ARGS)

setup-hooks:
	@echo "Installing native git pre-commit hook..."
	@mkdir -p .git/hooks
	@printf '#!/usr/bin/env bash\nset -e\nif [ -d .venv/bin ]; then export PATH="$$PWD/.venv/bin:$$PATH"; fi\necho "[pre-commit] Running ruff lint & format checks..."\nruff check src/ scripts/ tests/ app/\nruff format --check src/ scripts/ tests/ app/\necho "[pre-commit] Running zero-budget evaluation smoke gate (< 3.0s)..."\npython scripts/run_smoke_eval.py --golden\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git pre-commit hook installed successfully! (Runs ruff and make smoke-eval in < 0.2s before each commit)"

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
# App & Docker
# ------------------------------------------------------------

docker-build:
	docker build -t $(APP_IMAGE) .

docker-run:
	docker run -p 8501:8501 --env-file .env $(APP_IMAGE)

run-app:
	$(STREAMLIT) run app/app.py $(ARGS)


