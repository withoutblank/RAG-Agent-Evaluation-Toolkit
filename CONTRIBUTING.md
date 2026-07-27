# Contributing

Thank you for considering a contribution to the RAG Agent Evaluation Toolkit.
Keep changes small, testable, and connected to the project's portfolio-sized
RAG and evaluation scope.

## Development setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_all.py
```

Tests must not call paid APIs. Use the deterministic fake embedding and
generation providers.

## Workflow

1. Open or reference an issue when the change is more than a small correction.
2. Branch from `main` using `feat/`, `fix/`, `docs/`, `test/`, `chore/`, or
   `ci/`.
3. Use Conventional Commit prefixes.
4. Add or update tests and documentation with the behavior change.
5. Run `python scripts/check_all.py`.
6. Open a focused pull request with the template completed.

Do not include unrelated formatting churn.

## Required checks

```powershell
python -m pytest
python -m pytest --cov=rag_agent_eval_toolkit --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pip_audit
```

## Data and benchmark rules

- Never contribute private, customer, supplier, credential, or personal data.
- Do not put API keys in fixtures, logs, screenshots, or issue reports.
- New sample data must be original, fictional, or clearly licensed.
- Do not manually edit generated benchmark values.
- Rerun the benchmark when changing ingestion, chunking, embeddings, ranking,
  prompts, citation parsing, generation, labels, or metric logic.
- Include configuration, corpus and dataset checksums, model names, and the
  generating commit with results.
- Commit generated results only when they are the curated sample report.

## Proposing providers or metrics

Explain the problem, why the addition belongs in this compact project, its
runtime and licensing impact, how it remains injectable and offline-testable,
and how its outputs will be kept separate from existing deterministic metrics.
