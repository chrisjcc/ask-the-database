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

.PHONY: help install install-all install-dev lint format test preprocess train-sft train-grpo collect-distillation train-harness-zero evaluate evaluate-mve

help:
	@echo "AskTheDatabase - Operational Commands:"
	@echo "  make install              Install development & training dependencies"
	@echo "  make install-all          Install all dependencies (dev, training, and UI)"
	@echo "  make install-dev          Install minimal development dependencies only"
	@echo "  make lint                 Run ruff linter and mypy type checks"
	@echo "  make format               Auto-format code via ruff"
	@echo "  make test                 Execute pytest test suite"
	@echo "  make preprocess           Download datasets and run normalization pipeline"
	@echo "  make train-sft            Execute SFT warmup training on base model"
	@echo "  make train-grpo           Execute multi-turn GRPO / RLVR training"
	@echo "  make collect-distillation Collect K_SQL reviewed trajectories for Harness-Zero"
	@echo "  make train-harness-zero   Train SFT on K_SQL reviewed trajectories"
	@echo "  make evaluate             Evaluate benchmark suites (Spider, DK, Syn, Realistic)"
	@echo "  make evaluate-mve         Adjudicate 4-condition MVE decision gates & publish report"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,training]"

install-all:
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
	$(PYTHON) scripts/preprocess_data.py

train-sft:
	$(PYTHON) scripts/train_sft.py

train-grpo:
	$(PYTHON) scripts/train_rlvr.py

collect-distillation:
	$(PYTHON) scripts/collect_reviewed_trajectories.py

train-harness-zero:
	$(PYTHON) scripts/train_sft.py training.dataset_path=data/distillation/reviewed_trajectories.jsonl training.output_dir=models/sft/harness_zero

evaluate:
	$(PYTHON) scripts/evaluate_benchmark.py

evaluate-mve:
	$(PYTHON) scripts/run_mve_evaluation.py
