# Contributing

Thank you for your interest in contributing to **AskTheDatabase**. We welcome contributions from the community, whether they involve fixing bugs, improving documentation, adding new features, or enhancing model training and evaluation.

Please read this guide before opening an issue or submitting a pull request.

---

# Code of Conduct

Be respectful and constructive in all interactions.

We aim to foster an inclusive and collaborative environment where discussions remain focused on improving the project.

---

# Ways to Contribute

Contributions are welcome in many forms, including:

* Bug fixes
* New features
* Documentation improvements
* Performance optimizations
* Model evaluation improvements
* Training pipeline improvements
* Tests
* Refactoring
* Examples and tutorials

If you are unsure whether a contribution is appropriate, feel free to open an issue first to discuss it.

---

# Before You Start

Before implementing a significant feature or architectural change:

1. Search existing issues to avoid duplicate work.
2. Open an issue describing your proposal.
3. Wait for discussion before investing significant implementation effort.

Small bug fixes and documentation improvements generally do not require prior discussion.

---

# Development Setup

Clone the repository:

```bash
git clone https://github.com/chrisjcc/ask-the-database.git
cd ask-the-database
```

Create a Python environment using your preferred environment manager.

For example:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

depending on the current repository setup.

If additional development dependencies exist, install them as well.

---

# Running the Project

Refer to the project README for the latest instructions.

Common development tasks are exposed through the project's `Makefile`.

Examples may include:

```bash
make lint
make format
make test
```

If you are modifying the training pipeline, ensure that your changes integrate cleanly with the existing Hydra, DVC, and Weights & Biases workflows.

---

# Coding Standards

Please follow these conventions.

## Formatting

Format code before submitting:

```bash
make format
```

or the equivalent formatting command used by the repository.

---

## Linting

Run the linter before opening a pull request.

```bash
make lint
```

Your contribution should pass all lint checks.

---

## Type Hints

Use Python type hints whenever practical.

Public functions should include type annotations.

---

## Documentation

New public APIs should include docstrings.

Complex algorithms should include explanatory comments where appropriate.

Prefer explaining *why* something is implemented rather than *what* the code is doing.

---

# Testing

All new functionality should include appropriate tests whenever possible.

Before submitting a pull request, ensure that existing tests continue to pass.

```bash
make test
```

Bug fixes should ideally include a regression test demonstrating the original issue.

---

# Commit Messages

Write clear and descriptive commit messages.

Good examples:

```
Fix dataset loading for local cache

Improve ambiguity detection evaluation

Add support for configurable LoRA rank
```

Avoid messages such as:

```
fix

update

changes
```

---

# Pull Requests

Please keep pull requests focused.

A pull request should ideally address a single logical change.

Include:

* a clear description of the change
* motivation
* implementation details
* testing performed

If your pull request closes an issue, reference it:

```
Closes #42
```

---

# Review Process

Maintainers will review contributions for:

* correctness
* readability
* maintainability
* documentation
* testing
* consistency with the project's design

Review feedback is a normal part of the contribution process. Please be responsive to comments and update your pull request as needed.

---

# Documentation Contributions

Documentation improvements are always welcome.

Examples include:

* fixing inaccuracies
* improving explanations
* adding examples
* expanding tutorials
* improving API documentation

Documentation-only pull requests are encouraged.

---

# Reporting Bugs

When opening a bug report, please include:

* operating system
* Python version
* installation method
* steps to reproduce
* expected behavior
* actual behavior
* relevant logs or error messages

Providing a minimal reproducible example greatly speeds up debugging.

---

# Feature Requests

When proposing a new feature, please describe:

* the problem being solved
* the proposed solution
* possible alternatives
* any implementation considerations

Discussion before implementation helps avoid duplicated effort.

---

# Large Contributions

If you plan to contribute a major feature, model architecture, or significant pipeline change, please open an issue first so that the design can be discussed before development begins.

This helps ensure that contributions align with the project's long-term direction.

---

# License

By submitting a contribution, you agree that your work will be distributed under the same license as this repository.
