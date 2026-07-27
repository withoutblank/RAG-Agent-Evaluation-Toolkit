# AGENTS.md

## Project mission

Build and maintain a compact, public, interview-ready Python project named `rag-agent-eval-toolkit`.

The project demonstrates document ingestion, chunking, embeddings, vector retrieval, cited RAG responses, tool-using AI agents, controlled experiments, benchmarking, testing, and GitHub automation.

## Binding working rules

- Read the relevant specification Markdown files before implementing or changing behavior.
- Prefer the smallest implementation that completely satisfies the acceptance criteria.
- Do not add a framework merely because it is popular.
- Do not fabricate benchmark results, test output, latency, token usage, cost, screenshots, or GitHub status.
- Do not claim a task is complete until the relevant commands have run successfully.
- Do not commit secrets, personal data, real customer data, supplier data, private documents, API keys, tokens, local database files, or `.env`.
- Use fictional or clearly licensed public sample data only.
- Keep the project Windows-friendly. All documented commands must work in PowerShell, or a PowerShell equivalent must be provided.
- Use Python 3.11 or newer unless a dependency requires a narrower supported range.
- Prefer standard library modules and small dependencies.
- Pin direct dependencies after confirming a working environment.
- Use type hints for public functions.
- Use Pydantic models or dataclasses for core domain objects.
- Use structured logging rather than scattered `print` statements in library code.
- Make stochastic operations reproducible where practical by supporting a seed.
- Separate deterministic metrics from LLM-as-judge metrics.
- Keep generated evaluation results out of version control except for one curated sample report.
- Never automatically delete repositories, remote branches, releases, tags, issues, or user data.
- Never change repository visibility to public without an explicit confirmation flag.
- Never force-push to `main`.
- Never commit directly to `main` after the initial bootstrap unless explicitly requested.

## Required architecture boundaries

Maintain clear modules for:

- configuration;
- document loading;
- chunking;
- embeddings;
- vector storage and similarity search;
- retrieval;
- answer generation;
- agent tools and orchestration;
- evaluation dataset loading;
- metrics;
- experiment execution;
- reporting;
- CLI;
- Streamlit UI.

Do not put all logic into `app.py`.

## Preferred implementation choices

- Packaging: `pyproject.toml`
- Source layout: `src/rag_agent_eval_toolkit/`
- CLI: Typer
- Configuration: Pydantic Settings or an equivalently small typed configuration layer
- PDF parsing: pypdf
- Tabular results: pandas
- Vector computation: NumPy cosine similarity
- API client: official OpenAI Python SDK
- Agent implementation: official OpenAI Agents SDK or official tool-calling primitives
- UI: Streamlit
- Testing: pytest
- Coverage: pytest-cov
- Lint/format: Ruff
- Type checking: mypy
- Security checks: pip-audit plus secret scanning instructions
- Pre-commit: optional but recommended after MVP stability

Avoid LangChain, LlamaIndex, hosted vector databases, Redis, Celery, Kubernetes, Docker Compose, and cloud deployment in the MVP unless the specifications are revised.

## Coding standards

- Use descriptive names and small functions.
- Public functions require docstrings.
- Raise domain-specific exceptions at module boundaries.
- Validate file types, paths, encoding, and empty content.
- Preserve source metadata through ingestion, chunking, retrieval, answering, and reporting.
- Every answer citation must map back to a source document and chunk identifier.
- Retrieval results must include similarity score and rank.
- Do not silently swallow API or parsing errors.
- Retry only transient API failures, with bounded exponential backoff.
- Tests must not make paid network calls.
- External API clients must be injectable or mockable.
- Use deterministic fake providers in unit tests.

## Git workflow

Initial bootstrap may occur on `main`. After the first clean commit:

1. create a branch named `feat/<short-description>`, `fix/<short-description>`, `docs/<short-description>`, or `chore/<short-description>`;
2. make focused commits;
3. run all required checks;
4. push the branch;
5. open a pull request with a clear summary and test evidence;
6. merge with squash when checks pass;
7. delete the remote branch after merge.

Use Conventional Commit style:

- `feat:`
- `fix:`
- `docs:`
- `test:`
- `refactor:`
- `chore:`
- `ci:`

## Required validation commands

Codex must create and maintain commands equivalent to:

```powershell
python -m pytest
python -m pytest --cov=rag_agent_eval_toolkit --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pip_audit
```

A single convenience command should be available, preferably:

```powershell
python scripts/check_all.py
```

or a documented Make/PowerShell equivalent.

## Code review rules

Flag changes that:

- permit unsupported file types without validation;
- lose source metadata;
- return uncited answers;
- mix evaluation labels into retrieval inputs;
- use LLM scoring without identifying it as non-deterministic;
- expose secrets or user content in logs;
- make tests depend on live APIs;
- change benchmark configurations without updating documentation;
- hard-code a personal filesystem path;
- claim performance improvements without reproducible results;
- weaken workflow permissions;
- auto-publish the repository without explicit confirmation;
- increase scope without a clear connection to the portfolio objective.

## Documentation obligations

When changing behavior, update:

- README usage;
- architecture or interface documentation;
- configuration examples;
- acceptance tests;
- sample output if the output schema changes;
- changelog for user-visible changes.

## Completion reporting

At the end of each task, report:

1. files changed;
2. behavior implemented;
3. checks executed;
4. checks passed or failed;
5. known limitations;
6. next recommended task.

Do not use vague claims such as “everything works” without command evidence.
