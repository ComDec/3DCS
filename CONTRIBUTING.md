# Contributing to 3DCS

Thank you for considering contributing to 3DCS! This document explains how to set up
a development environment and submit changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ComDec/3DCS.git
cd 3DCS

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies (RDKit comes from the `rdkit` package on PyPI;
# do not install `rdkit-pypi`)
pip install -e ".[dev]"

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

CI runs `pytest -m "not slow"` on Python 3.9, 3.10, 3.11 and 3.12.

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues (the same paths as CI and `make lint`)
ruff check src/ tests/ examples/ reproduce/

# Auto-fix issues
ruff check --fix src/ tests/ examples/ reproduce/

# Format code
ruff format src/ tests/ examples/ reproduce/
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
7. Document the metric definitions under `docs/metrics/` and, if the task reproduces a paper table,
   add `reproduce/<table>/` with `run.sh` and `expected.csv` (see `reproduce/README.md`)

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Run `make lint` and `make test` to verify
4. Submit a pull request with a clear description

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/ComDec/3DCS/issues) to report
bugs or request features. Please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs. actual behavior
