"""Injectable direct-RAG experiment execution with deterministic scoring."""

from __future__ import annotations

import math
import platform
import re
import sys
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from rag_agent_eval_toolkit import __version__
from rag_agent_eval_toolkit.evaluation.dataset import (
    EVALUATION_DATASET_SCHEMA_VERSION,
    EvaluationDataset,
)
from rag_agent_eval_toolkit.evaluation.metrics import (
    AbstentionAssessment,
    CitationScore,
    RetrievalScore,
    assess_abstention,
    citation_score,
    extract_citation_labels,
    latency_summary,
    required_fact_coverage,
    retrieval_score,
)
from rag_agent_eval_toolkit.exceptions import ExperimentError
from rag_agent_eval_toolkit.experiments.config import BenchmarkSuiteConfig
from rag_agent_eval_toolkit.experiments.provenance import (
    GIT_DIRTY_WARNING,
    GIT_UNAVAILABLE_WARNING,
    GIT_UNVERIFIED_WARNING,
    GIT_WORKTREE_STATE_FIELD,
    GitProvenance,
    detect_git_commit,
    detect_git_provenance,
)
from rag_agent_eval_toolkit.models import (
    EvaluationQuestion,
    ExperimentConfig,
    ExperimentResult,
    RagAnswer,
    RetrievalResult,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SENSITIVE_ENVIRONMENT_KEY_PATTERN = re.compile(
    r"(?:secret|token|password|api[_-]?key|credential)",
    flags=re.IGNORECASE,
)
_COMPLETED_FAILURE_CATEGORIES = ("ingestion", "retrieval", "generation")

TimestampFactory = Callable[[], str]
RunIdFactory = Callable[[], str]


class DirectRagExecutor(Protocol):
    """Callable adapter for one direct-RAG question/configuration execution."""

    def __call__(
        self,
        question: str,
        config: ExperimentConfig,
        /,
    ) -> RagAnswer:
        """Return one typed direct-RAG answer from question text and configuration only."""


class DirectRagExecutorFactory(Protocol):
    """Build a configuration-specific direct-RAG executor."""

    def __call__(self, config: ExperimentConfig, /) -> DirectRagExecutor:
        """Prepare retrieval and generation for one experiment configuration."""


@dataclass(frozen=True, slots=True)
class _CompletedQuestion:
    payload: dict[str, Any]
    answerable: bool
    retrieval: RetrievalScore | None
    fact_coverage: float | None
    citations: CitationScore | None
    abstention: AbstentionAssessment
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


def run_experiment(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    executor: DirectRagExecutor,
    *,
    corpus_checksum: str,
    available_sources: Collection[str] | None = None,
    git_commit: str | None = None,
    repo_dir: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timestamp_factory: TimestampFactory | None = None,
    run_id_factory: RunIdFactory | None = None,
    continue_on_error: bool = True,
) -> ExperimentResult:
    """Execute and score one configuration while preserving partial failures."""

    _validate_experiment_inputs(config, dataset, corpus_checksum)
    timestamp = timestamp_factory or _utc_timestamp
    make_run_id = run_id_factory or _new_run_id
    started_at = _validated_generated_text(timestamp(), name="started_at_utc")
    run_id = _validated_generated_text(make_run_id(), name="run_id")

    git_provenance = _resolve_git_provenance(
        git_commit,
        repo_dir=repo_dir or Path.cwd(),
    )
    warnings = _git_provenance_warnings(git_provenance)

    run_environment = _build_environment(
        config,
        environment,
        git_worktree_state=git_provenance.worktree_state,
    )
    known_sources = _validated_available_sources(available_sources)
    completed: list[_CompletedQuestion] = []
    question_results: list[dict[str, Any]] = []

    for question in dataset.questions:
        try:
            # Evaluation labels never cross the executor boundary.
            answer = executor(question.question, config)
            scored = _score_question(
                question,
                answer,
                config=config,
                available_sources=known_sources,
            )
        except Exception as exc:
            if not continue_on_error:
                raise ExperimentError(
                    f"Experiment {config.name!r} failed on question {question.id!r}."
                ) from exc
            question_results.append(_failed_question_payload(question, exc))
            continue
        completed.append(scored)
        question_results.append(scored.payload)

    aggregate_metrics, aggregate_warnings = _aggregate_metrics(
        completed,
        total_question_count=len(dataset.questions),
    )
    warnings.extend(aggregate_warnings)
    failed_count = len(dataset.questions) - len(completed)
    if failed_count:
        warnings.append(
            f"{failed_count} of {len(dataset.questions)} questions failed at runtime; "
            "aggregate quality metrics exclude failed questions."
        )

    completed_at = _validated_generated_text(timestamp(), name="completed_at_utc")
    return ExperimentResult(
        run_id=run_id,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        git_commit=git_provenance.commit,
        corpus_checksum=corpus_checksum,
        evaluation_dataset_checksum=dataset.checksum_sha256,
        config=config,
        aggregate_metrics=aggregate_metrics,
        question_results=question_results,
        environment=run_environment,
        warnings=_deduplicate_strings(warnings),
    )


def run_benchmark(
    suite: BenchmarkSuiteConfig,
    dataset: EvaluationDataset,
    executor_factory: DirectRagExecutorFactory,
    *,
    corpus_checksum: str,
    available_sources: Collection[str] | None = None,
    git_commit: str | None = None,
    repo_dir: Path | None = None,
    environment: Mapping[str, str] | None = None,
    timestamp_factory: TimestampFactory | None = None,
    run_id_factory: RunIdFactory | None = None,
    continue_on_error: bool = True,
) -> tuple[ExperimentResult, ...]:
    """Execute every validated configuration in deterministic YAML order."""

    common_environment = {"provider": suite.provider, **dict(environment or {})}
    results: list[ExperimentResult] = []
    for config in suite.experiments:
        try:
            executor = executor_factory(config)
        except Exception as exc:
            raise ExperimentError(
                f"Unable to prepare direct-RAG executor for experiment {config.name!r}."
            ) from exc
        results.append(
            run_experiment(
                config,
                dataset,
                executor,
                corpus_checksum=corpus_checksum,
                available_sources=available_sources,
                git_commit=git_commit,
                repo_dir=repo_dir,
                environment=common_environment,
                timestamp_factory=timestamp_factory,
                run_id_factory=run_id_factory,
                continue_on_error=continue_on_error,
            )
        )
    return tuple(results)


def _score_question(
    question: EvaluationQuestion,
    answer: RagAnswer,
    *,
    config: ExperimentConfig,
    available_sources: set[str] | None,
) -> _CompletedQuestion:
    _validate_rag_answer(question, answer, config=config)
    retrieval_results = tuple(answer.retrieval_results)
    abstention = assess_abstention(answer.answer, answerable=question.answerable)

    retrieval: RetrievalScore | None = None
    facts: float | None = None
    citations: CitationScore | None = None
    if question.answerable:
        retrieval = retrieval_score(
            retrieval_results,
            question.expected_sources,
            top_k=config.top_k,
        )
        facts = required_fact_coverage(answer.answer, question.required_facts)
        citations = citation_score(
            answer.answer,
            retrieval_results,
            question.expected_sources,
        )

    retrieved_sources = list(dict.fromkeys(result.source_name for result in retrieval_results))
    citation_labels = list(extract_citation_labels(answer.answer))
    structured_citations = [citation.citation_label for citation in answer.citations]
    warnings = list(answer.warnings)
    if structured_citations != citation_labels:
        warnings.append(
            "Structured citation labels differ from citation occurrences in the answer text."
        )
    if citations is not None and citations.invalid_labels:
        warnings.append(
            f"{len(citations.invalid_labels)} citation occurrence(s) did not resolve "
            "to supplied evidence."
        )

    failure_categories = _classify_failure(
        question=question,
        retrieval=retrieval,
        fact_coverage=facts,
        citations=citations,
        abstention=abstention,
        available_sources=available_sources,
    )
    payload: dict[str, Any] = {
        "question_id": question.id,
        "question": question.question,
        "expected_answer": question.expected_answer,
        "answerable": question.answerable,
        "difficulty": question.difficulty,
        "tags": list(question.tags),
        "expected_sources": list(question.expected_sources),
        "required_facts": list(question.required_facts),
        "status": "completed",
        "first_relevant_rank": (retrieval.first_relevant_rank if retrieval is not None else None),
        "retrieved_sources": retrieved_sources,
        "retrieved_chunks": [
            {
                "rank": result.rank,
                "score": result.score,
                "source_name": result.source_name,
                "source_id": result.source_id,
                "chunk_id": result.chunk_id,
                "citation_label": result.citation_label,
            }
            for result in retrieval_results
        ],
        "generated_answer": answer.answer,
        "citations": citation_labels,
        "hit_rate_at_k": retrieval.hit_rate_at_k if retrieval is not None else None,
        "reciprocal_rank": retrieval.reciprocal_rank if retrieval is not None else None,
        "source_recall_at_k": (retrieval.source_recall_at_k if retrieval is not None else None),
        "fact_coverage": facts,
        "citation_validity": citations.validity if citations is not None else None,
        "citation_precision": citations.precision if citations is not None else None,
        "abstention_result": {
            "expected_to_abstain": abstention.expected_to_abstain,
            "explicit_abstention": abstention.explicit_abstention,
            "contains_additional_claims": abstention.contains_additional_claims,
            "correct": abstention.correct,
        },
        "retrieval_latency_ms": answer.retrieval_latency_ms,
        "generation_latency_ms": answer.generation_latency_ms,
        "total_latency_ms": answer.total_latency_ms,
        "warnings": _deduplicate_strings(warnings),
        "failure_categories": failure_categories,
        "error": None,
    }
    return _CompletedQuestion(
        payload=payload,
        answerable=question.answerable,
        retrieval=retrieval,
        fact_coverage=facts,
        citations=citations,
        abstention=abstention,
        retrieval_latency_ms=answer.retrieval_latency_ms,
        generation_latency_ms=answer.generation_latency_ms,
        total_latency_ms=answer.total_latency_ms,
    )


def _classify_failure(
    *,
    question: EvaluationQuestion,
    retrieval: RetrievalScore | None,
    fact_coverage: float | None,
    citations: CitationScore | None,
    abstention: AbstentionAssessment,
    available_sources: set[str] | None,
) -> list[str]:
    categories: list[str] = []
    missing_from_corpus = (
        set(question.expected_sources) - available_sources
        if available_sources is not None
        else set()
    )
    if missing_from_corpus:
        categories.append("ingestion")

    if question.answerable:
        if not missing_from_corpus and retrieval is not None and retrieval.hit_rate_at_k == 0.0:
            categories.append("retrieval")
        evidence_retrieved = retrieval is not None and retrieval.hit_rate_at_k == 1.0
        generation_failed = (
            fact_coverage is None
            or fact_coverage < 1.0
            or citations is None
            or citations.validity < 1.0
            or citations.precision < 1.0
            or abstention.explicit_abstention
        )
        if evidence_retrieved and generation_failed:
            categories.append("generation")
    elif abstention.correct is not True:
        categories.append("generation")
    return categories


def _failed_question_payload(
    question: EvaluationQuestion,
    exception: Exception,
) -> dict[str, Any]:
    return {
        "question_id": question.id,
        "question": question.question,
        "expected_answer": question.expected_answer,
        "answerable": question.answerable,
        "difficulty": question.difficulty,
        "tags": list(question.tags),
        "expected_sources": list(question.expected_sources),
        "required_facts": list(question.required_facts),
        "status": "failed",
        "first_relevant_rank": None,
        "retrieved_sources": [],
        "retrieved_chunks": [],
        "generated_answer": "",
        "citations": [],
        "hit_rate_at_k": None,
        "reciprocal_rank": None,
        "source_recall_at_k": None,
        "fact_coverage": None,
        "citation_validity": None,
        "citation_precision": None,
        "abstention_result": None,
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": None,
        "warnings": ["Direct-RAG execution failed; inspect local logs for provider details."],
        "failure_categories": ["configuration/runtime"],
        "error": {
            "type": type(exception).__name__,
            "message": "Direct-RAG execution failed.",
        },
    }


def _aggregate_metrics(
    completed: Sequence[_CompletedQuestion],
    *,
    total_question_count: int,
) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {
        "question_count": float(total_question_count),
        "successful_question_count": float(len(completed)),
        "failed_question_count": float(total_question_count - len(completed)),
    }
    warnings: list[str] = []
    answerable = [item for item in completed if item.answerable]
    unanswerable = [item for item in completed if not item.answerable]
    metrics["successful_answerable_question_count"] = float(len(answerable))
    metrics["successful_unanswerable_question_count"] = float(len(unanswerable))

    retrieval_values = [item.retrieval for item in answerable if item.retrieval is not None]
    if retrieval_values:
        metrics["hit_rate_at_k"] = fmean(score.hit_rate_at_k for score in retrieval_values)
        metrics["mrr"] = fmean(score.reciprocal_rank for score in retrieval_values)
        metrics["source_recall_at_k"] = fmean(
            score.source_recall_at_k for score in retrieval_values
        )
    else:
        warnings.append("Retrieval metrics omitted: no successful answerable questions.")

    fact_values = [item.fact_coverage for item in answerable if item.fact_coverage is not None]
    if fact_values:
        metrics["required_fact_coverage"] = fmean(fact_values)
    else:
        warnings.append("Fact coverage omitted: no successful answerable questions.")

    citation_values = [item.citations for item in answerable if item.citations is not None]
    if citation_values:
        metrics["citation_validity"] = fmean(score.validity for score in citation_values)
        metrics["citation_precision"] = fmean(score.precision for score in citation_values)
    else:
        warnings.append("Citation metrics omitted: no successful answerable questions.")

    abstention_values = [
        item.abstention.correct for item in unanswerable if item.abstention.correct is not None
    ]
    if abstention_values:
        metrics["unanswerable_accuracy"] = fmean(
            1.0 if value else 0.0 for value in abstention_values
        )
    else:
        warnings.append("Unanswerable accuracy omitted: no successful unanswerable questions.")

    latency_inputs = {
        "retrieval_latency": [item.retrieval_latency_ms for item in completed],
        "generation_latency": [item.generation_latency_ms for item in completed],
        "total_latency": [item.total_latency_ms for item in completed],
    }
    for metric_name, values in latency_inputs.items():
        summary = latency_summary(values, metric_name=metric_name)
        warnings.extend(summary.warnings)
        if summary.mean_ms is not None:
            metrics[f"{metric_name}_mean_ms"] = summary.mean_ms
        if summary.median_ms is not None:
            metrics[f"{metric_name}_median_ms"] = summary.median_ms
        if summary.p95_ms is not None:
            metrics[f"{metric_name}_p95_ms"] = summary.p95_ms

    for category in _COMPLETED_FAILURE_CATEGORIES:
        count = sum(category in item.payload["failure_categories"] for item in completed)
        metrics[f"failure_{category}_count"] = float(count)
    runtime_failure_count = total_question_count - len(completed)
    metrics["failure_configuration_runtime_count"] = float(runtime_failure_count)
    return metrics, warnings


def _validate_experiment_inputs(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    corpus_checksum: str,
) -> None:
    if dataset.schema_version != EVALUATION_DATASET_SCHEMA_VERSION:
        raise ExperimentError(
            f"Unsupported evaluation dataset schema version: {dataset.schema_version}"
        )
    if _SHA256_PATTERN.fullmatch(corpus_checksum) is None:
        raise ExperimentError("corpus_checksum must be a 64-character lowercase SHA-256 hash")
    if _SHA256_PATTERN.fullmatch(dataset.checksum_sha256) is None:
        raise ExperimentError(
            "evaluation dataset checksum must be a 64-character lowercase SHA-256 hash"
        )
    if config.evaluation_mode != "direct_rag":
        raise ExperimentError("Only direct_rag evaluation is supported by the MVP runner")
    if config.llm_judge_enabled:
        raise ExperimentError("LLM-as-judge must be disabled for deterministic MVP runs")
    for field_name, value in (
        ("embedding_model", config.embedding_model),
        ("generation_model", config.generation_model),
        ("prompt_version", config.prompt_version),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ExperimentError(f"config.{field_name} must be a non-empty, trimmed string")
    if (
        isinstance(config.temperature, bool)
        or not isinstance(config.temperature, (int, float))
        or not math.isfinite(float(config.temperature))
        or not 0.0 <= float(config.temperature) <= 2.0
    ):
        raise ExperimentError("config.temperature must be a finite number between 0 and 2")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ExperimentError("config.seed must be an integer")


def _validate_rag_answer(
    question: EvaluationQuestion,
    answer: RagAnswer,
    *,
    config: ExperimentConfig,
) -> None:
    if not isinstance(answer, RagAnswer):
        raise TypeError("direct-RAG executor must return RagAnswer")
    if answer.question != question.question:
        raise ValueError("RagAnswer.question does not match the evaluation question")
    if answer.model_name != config.generation_model:
        raise ValueError("RagAnswer model_name does not match the experiment configuration")
    if answer.prompt_version != config.prompt_version:
        raise ValueError("RagAnswer prompt_version does not match the experiment configuration")
    if len(answer.retrieval_results) > config.top_k:
        raise ValueError("RagAnswer contains more retrieval results than configured top_k")

    expected_ranks = list(range(1, len(answer.retrieval_results) + 1))
    actual_ranks = [result.rank for result in answer.retrieval_results]
    if actual_ranks != expected_ranks:
        raise ValueError("RagAnswer retrieval ranks must be contiguous and start at one")
    for result in answer.retrieval_results:
        _validate_retrieval_result(result)

    _validate_latency(answer.retrieval_latency_ms, name="retrieval_latency_ms")
    _validate_latency(answer.generation_latency_ms, name="generation_latency_ms")
    _validate_latency(answer.total_latency_ms, name="total_latency_ms")


def _validate_retrieval_result(result: RetrievalResult) -> None:
    expected_label = f"[{result.source_name}#{result.chunk_id}]"
    if result.citation_label != expected_label:
        raise ValueError("RetrievalResult citation_label does not match its source and chunk")
    if not math.isfinite(result.score):
        raise ValueError("RetrievalResult score must be finite")


def _validate_latency(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _build_environment(
    config: ExperimentConfig,
    supplied: Mapping[str, str] | None,
    *,
    git_worktree_state: str,
) -> dict[str, str]:
    environment = {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "operating_system": f"{platform.system()} {platform.release()}".strip(),
        "package_version": __version__,
        "embedding_model": config.embedding_model,
        "generation_model": config.generation_model,
        GIT_WORKTREE_STATE_FIELD: git_worktree_state,
    }
    for key, value in (supplied or {}).items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ExperimentError("environment keys must be non-empty, trimmed strings")
        if _SENSITIVE_ENVIRONMENT_KEY_PATTERN.search(key):
            raise ExperimentError(f"Sensitive environment field is not allowed: {key}")
        if not isinstance(value, str):
            raise ExperimentError(f"Environment value for {key!r} must be a string")
        if key in environment and environment[key] != value:
            raise ExperimentError(
                f"Environment field {key!r} conflicts with recorded run provenance"
            )
        environment[key] = value
    return environment


def _resolve_git_provenance(
    git_commit: str | None,
    *,
    repo_dir: Path,
) -> GitProvenance:
    if git_commit is None:
        return detect_git_provenance(repo_dir)
    if _GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ExperimentError("git_commit must be a full 40- or 64-character lowercase hash")
    return GitProvenance(commit=git_commit, worktree_state="unverified")


def _git_provenance_warnings(provenance: GitProvenance) -> list[str]:
    if provenance.worktree_state == "dirty":
        return [GIT_DIRTY_WARNING]
    if provenance.worktree_state == "unavailable":
        return [GIT_UNAVAILABLE_WARNING]
    if provenance.worktree_state == "unverified":
        return [GIT_UNVERIFIED_WARNING]
    return []


def _validated_available_sources(
    values: Collection[str] | None,
) -> set[str] | None:
    if values is None:
        return None
    parsed: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ExperimentError(
                "available_sources must contain only non-empty, trimmed source names"
            )
        parsed.add(value)
    return parsed


def _validated_generated_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExperimentError(f"{name} factory must return a non-empty, trimmed string")
    return value


def _deduplicate_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    return uuid.uuid4().hex


__all__ = [
    "DirectRagExecutor",
    "DirectRagExecutorFactory",
    "detect_git_commit",
    "detect_git_provenance",
    "run_benchmark",
    "run_experiment",
]
