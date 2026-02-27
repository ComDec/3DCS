# Contributing to 3DCS

Thank you for considering contributing to 3DCS! This document explains how to set up
a development environment and submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/EscheWang/3DBench.git
cd 3DBench

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install RDKit (required for molecule parsing)
pip install rdkit-pypi

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=three_dbench

# Skip slow tests (those requiring HF downloads)
pytest -m "not slow"
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

Key style rules:
- Line length: 120 characters
- Python 3.9+ syntax
- Imports sorted with `isort` rules (via ruff)

## Adding a New Benchmark

1. Create a subpackage under `src/three_dbench/` (e.g., `my_task/`)
2. Add dataset conversion in `src/three_dbench/datasets/my_task.py`
3. Add evaluation logic in `src/three_dbench/my_task/evaluation.py`
4. Add a benchmark entry point in `src/three_dbench/benchmarks/my_task.py`
5. Register the task in `src/three_dbench/__main__.py`
6. Add tests under `tests/`

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Run `make lint` and `make test` to verify
4. Submit a pull request with a clear description

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/EscheWang/3DBench/issues) to report
bugs or request features. Please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs. actual behavior
