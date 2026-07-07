# Repository Guidelines

## Project Structure & Module Organization

This repository is a from-scratch machine learning practice project. Each numbered directory is dedicated to one algorithm:

- `01_knn/` through `08_neural_network/`: algorithm-specific notes, implementation files, demos, and tests.
- `utils/`: shared helpers such as metrics, preprocessing, or dataset utilities.
- `datasets/`: small committed sample data only. Keep large or local data in ignored paths such as `datasets/raw/` and `datasets/processed/`.
- `main.py`: optional local scratch entry point.

Keep each algorithm self-contained. For example, KNN code should live under `01_knn/`, not in another algorithm directory.

## Build, Test, and Development Commands

Use a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Run a quick syntax check:

```bash
python -m compileall .
```

Run tests once they exist:

```bash
python -m pytest
```

Run a specific demo, for example:

```bash
python 01_knn/demo.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Prefer clear, educational code over clever abstractions. Name files by algorithm or purpose, such as `knn.py`, `linear_regression.py`, `demo.py`, and `test_knn.py`.

Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive names for mathematical values, such as `learning_rate`, `n_samples`, `weights`, and `bias`.

## Testing Guidelines

Use `pytest` for tests. Place tests near the algorithm being implemented, such as `01_knn/test_knn.py`, or move to a top-level `tests/` directory later if the project grows.

Tests should cover core behavior, edge cases, and expected shapes. For numerical algorithms, assert approximate values with tolerances instead of exact floating-point equality.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so use simple Conventional Commit-style messages going forward:

- `feat: implement knn classifier`
- `test: add linear regression convergence tests`
- `docs: update pca notes`

Pull requests should describe what algorithm or utility changed, include how it was tested, and mention any known limitations. Add plots or screenshots only when visual output is relevant.

## Agent-Specific Instructions

Do not add completed algorithm implementations unless explicitly requested. This project is intended for manual practice, so prefer guides, scaffolding, tests, and review comments over filling in solutions.
