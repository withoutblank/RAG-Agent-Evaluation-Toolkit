"""Typed domain models shared across the toolkit.

The models deliberately contain data rather than provider or persistence logic.
This keeps the deterministic pipeline easy to test and allows external providers
to be replaced with offline fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


def _validate_portable_relative_path(relative_path: str) -> None:
    """Reject absolute, parent-traversing, or non-normalised relative paths."""
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
    ):
        msg = "relative_path must be a portable POSIX-style relative path"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A normalized source document and its provenance metadata."""

    source_id: str
    source_name: str
    relative_path: str
    file_type: str
    text: str
    content_sha256: str
    page_count: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate path and basic content invariants."""
        _validate_portable_relative_path(self.relative_path)
        if not self.source_name:
            raise ValueError("source_name must not be empty")
        if not self.text:
            raise ValueError("text must not be empty")
        if self.page_count is not None and self.page_count < 1:
            raise ValueError("page_count must be positive when provided")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A deterministic character range linked to its source document."""

    chunk_id: str
    source_id: str
    source_name: str
    chunk_index: int
    text: str
    character_start: int
    character_end: int
    page_number: int | None
    token_estimate: int
    chunk_size: int
    overlap: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk offsets and configuration invariants."""
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if not self.text:
            raise ValueError("chunk text must not be empty")
        if self.character_start < 0 or self.character_end <= self.character_start:
            raise ValueError("chunk character offsets are invalid")
        if self.character_end - self.character_start != len(self.text):
            raise ValueError("chunk offsets must match the chunk text length")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive when provided")
        if self.token_estimate < 1:
            raise ValueError("token_estimate must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Compatibility and provenance metadata for a persisted vector index."""

    schema_version: int
    created_at_utc: str
    corpus_checksum: str
    document_count: int
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    overlap: int
    normalisation_method: str
    vector_file: str
    metadata_file: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked evidence result returned for a retrieval query."""

    query: str
    rank: int
    score: float
    chunk_id: str
    source_id: str
    source_name: str
    text: str
    page_number: int | None
    citation_label: str


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation that resolves a generated statement to one source chunk."""

    citation_label: str
    source_id: str
    source_name: str
    chunk_id: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """A direct-RAG response with timings and the evidence used."""

    question: str
    answer: str
    citations: list[Citation]
    retrieval_results: list[RetrievalResult]
    model_name: str
    prompt_version: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    usage: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A concise, non-confidential record of one agent tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    started_at: str
    duration_ms: float
    result_summary: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """A tool-using agent response and its observable call trace."""

    question: str
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCall]
    model_name: str
    total_latency_ms: float
    usage: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """One labelled question from the versioned evaluation dataset."""

    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]
    required_facts: list[str]
    answerable: bool
    tags: list[str]
    difficulty: str


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Deterministic retrieval metrics for one question."""

    hit_rate_at_k: float
    reciprocal_rank: float
    source_recall_at_k: float
    retrieval_latency_ms: float


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """Deterministic answer, citation, and abstention metrics."""

    fact_coverage: float
    citation_validity: float
    citation_precision: float | None
    unanswerable_accuracy: float | None
    response_latency_ms: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One reproducible retrieval and generation experiment configuration."""

    name: str
    chunk_size: int
    overlap: int
    top_k: int
    embedding_model: str
    generation_model: str
    temperature: float = 0.0
    prompt_version: str = "v1"
    seed: int = 0
    evaluation_mode: str = "direct_rag"
    llm_judge_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate benchmark configuration constraints."""
        if not self.name:
            raise ValueError("experiment name must not be empty")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Serializable output for one completed experiment configuration."""

    run_id: str
    started_at_utc: str
    completed_at_utc: str
    git_commit: str | None
    corpus_checksum: str
    evaluation_dataset_checksum: str
    config: ExperimentConfig
    aggregate_metrics: dict[str, float]
    question_results: list[dict[str, Any]]
    environment: dict[str, str]
    warnings: list[str] = field(default_factory=list)
