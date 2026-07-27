# Changelog

All notable user-visible changes to this project are documented here. The
format follows Keep a Changelog principles, and releases use Semantic
Versioning.

## [Unreleased]

### Added

- Markdown, plain-text, and text-based PDF ingestion with validation,
  normalisation, checksums, duplicate detection, and source metadata.
- Deterministic character chunking with stable identifiers and provenance.
- Injectable deterministic fake and OpenAI embedding/generation providers.
- Persisted exact-search NumPy vector index with compatibility metadata.
- Ranked retrieval and direct-RAG answers with context-only prompting,
  abstention behavior, evidence retention, and citation validation.
- Constrained agent mode with typed corpus-search and source-metadata tools plus
  captured tool traces.
- Versioned 20-question evaluation dataset for the eight-document fictional EV
  support corpus.
- Deterministic retrieval, required-fact, citation, abstention, and latency
  metrics with four controlled benchmark configurations.
- JSON, CSV, and Markdown reporting, including a real fake-provider sample
  benchmark with clean Git provenance and per-question failure analysis.
- Typer CLI commands for ingestion, direct and agent answers, evaluation,
  benchmarking, report generation, and environment diagnosis.
- One-page Streamlit interface for corpus/index status, cited answers, retrieved
  evidence, agent traces, and benchmark comparison.
- Offline unit and integration tests, coverage enforcement, Ruff, mypy,
  dependency auditing, GitHub Actions definitions, Dependabot, contribution
  templates, and security guidance.
- PowerShell-first setup, architecture, evaluation methodology, and benchmark
  reproduction documentation.
- MIT licensing and factual public attribution to Marshall.

### Changed

- Replaced planning-stage repository text with reader-facing MVP documentation
  grounded in the generated sample artifacts.
- Removed the unused direct pandas dependency; Streamlit may still resolve it as
  an indirect dependency.
- Made the Streamlit UI reject indexes built from a stale corpus and show
  actionable rebuild guidance.
- Made benchmark reports distinguish clean, dirty, unavailable, and unverified
  Git provenance; dirty runs no longer attribute their inputs to `HEAD`.
- Made report ordering independent of input-file order so `rag-eval report`
  reproduces the benchmark Markdown byte-for-byte.
- Made repository setup explicitly disable automatic merged-branch deletion;
  reviewed remote task branches remain a manual post-merge action.
- Made milestone discovery use JSON parsing instead of a shell-sensitive filter
  so project setup works reliably in Windows PowerShell.
- Made GitHub CLI output explicitly UTF-8 and constructed the milestone em dash
  by code point so BOM-less PowerShell scripts preserve its public title.

### Security

- Kept API keys environment-only, redacted sensitive log fields, excluded local
  secrets/indexes/generated runs from version control, and used only fictional
  public-safe sample data.
- Limited agent capabilities to two allowlisted retrieval tools with no shell,
  browser, arbitrary filesystem, or code-execution access.
- Added a suite-wide outbound-network guard so tests fail if a live client or
  DNS lookup bypasses the injected fakes.
- Pinned every third-party GitHub Actions reference to a reviewed immutable
  release commit while retaining readable version comments.
