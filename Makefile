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

.PHONY: help install lint format test preprocess

help:
	@echo "AskTheDatabase - Operational Commands:"
	@echo "  make install     Install production & development dependencies"
	@echo "  make lint        Run ruff linter and mypy type checks"
	@echo "  make format      Auto-format code via ruff"
	@echo "  make test        Execute pytest test suite"
	@echo "  make preprocess  Download datasets and run normalization pipeline"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src/ scripts/ tests/
	$(MYPY) src/

format:
	$(RUFF) check --fix src/ scripts/ tests/
	$(RUFF) format src/ scripts/ tests/

test:
	$(PYTEST) tests/

preprocess:
	$(PYTHON) scripts/preprocess_data.py
