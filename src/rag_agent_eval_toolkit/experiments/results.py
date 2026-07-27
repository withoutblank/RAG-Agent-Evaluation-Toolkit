"""Experiment-result schema serialization and reproducibility projections."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_agent_eval_toolkit.exceptions import ExperimentError
from rag_agent_eval_toolkit.experiments.provenance import (
    GIT_WORKTREE_STATE_FIELD,
    GIT_WORKTREE_STATES,
    LEGACY_GIT_PROVENANCE_WARNING,
)
from rag_agent_eval_toolkit.models import ExperimentConfig, ExperimentResult

EXPERIMENT_RESULT_SCHEMA_VERSION = 1

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "started_at_utc",
        "completed_at_utc",
        "git_commit",
        "corpus_checksum",
        "evaluation_dataset_checksum",
        "config",
        "aggregate_metrics",
        "question_results",
        "environment",
        "warnings",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "name",
        "chunk_size",
        "overlap",
        "top_k",
        "embedding_model",
        "generation_model",
        "temperature",
        "prompt_version",
        "seed",
        "evaluation_mode",
        "llm_judge_enabled",
    }
)
_QUESTION_FIELDS = frozenset(
    {
        "question_id",
        "question",
        "expected_answer",
        "answerable",
        "difficulty",
        "tags",
        "expected_sources",
        "required_facts",
        "status",
        "first_relevant_rank",
        "retrieved_sources",
        "retrieved_chunks",
        "generated_answer",
        "citations",
        "hit_rate_at_k",
        "reciprocal_rank",
        "source_recall_at_k",
        "fact_coverage",
        "citation_validity",
        "citation_precision",
        "abstention_result",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "warnings",
        "failure_categories",
        "error",
    }
)
_RETRIEVED_CHUNK_FIELDS = frozenset(
    {
        "rank",
        "score",
        "source_name",
        "source_id",
        "chunk_id",
        "citation_label",
    }
)
_ABSTENTION_FIELDS = frozenset(
    {
        "expected_to_abstain",
        "explicit_abstention",
        "contains_additional_claims",
        "correct",
    }
)
_ERROR_FIELDS = frozenset({"type", "message"})
_REQUIRED_AGGREGATE_FIELDS = frozenset(
    {
        "question_count",
        "successful_question_count",
        "failed_question_count",
        "successful_answerable_question_count",
        "successful_unanswerable_question_count",
        "failure_ingestion_count",
        "failure_retrieval_count",
        "failure_generation_count",
        "failure_configuration_runtime_count",
    }
)
_ALLOWED_AGGREGATE_FIELDS = _REQUIRED_AGGREGATE_FIELDS | frozenset(
    {
        "hit_rate_at_k",
        "mrr",
        "source_recall_at_k",
        "required_fact_coverage",
        "citation_validity",
        "citation_precision",
        "unanswerable_accuracy",
        "retrieval_latency_mean_ms",
        "retrieval_latency_median_ms",
        "retrieval_latency_p95_ms",
        "generation_latency_mean_ms",
        "generation_latency_median_ms",
        "generation_latency_p95_ms",
        "total_latency_mean_ms",
        "total_latency_median_ms",
        "total_latency_p95_ms",
    }
)
_REQUIRED_ENVIRONMENT_FIELDS = frozenset(
    {
        "python_version",
        "operating_system",
        "package_version",
        "embedding_model",
        "generation_model",
    }
)
_FAILURE_CATEGORIES = frozenset({"ingestion", "retrieval", "generation", "configuration/runtime"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:secret|token|password|api[_-]?key|credential)",
    flags=re.IGNORECASE,
)
_VARIABLE_TOP_LEVEL_FIELDS = frozenset({"run_id", "started_at_utc", "completed_at_utc"})
_VARIABLE_QUESTION_FIELDS = frozenset(
    {"retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"}
)


def experiment_result_payload(result: ExperimentResult) -> dict[str, Any]:
    """Return the complete JSON-ready schema-version-1 result mapping."""

    environment = dict(result.environment)
    warnings = list(result.warnings)
    _normalise_git_provenance(
        git_commit=result.git_commit,
        environment=environment,
        warnings=warnings,
    )
    return {
        "schema_version": EXPERIMENT_RESULT_SCHEMA_VERSION,
        **asdict(result),
        "environment": environment,
        "warnings": warnings,
    }


def load_experiment_result(path: Path) -> ExperimentResult:
    """Load one strict schema-version-1 experiment-result JSON artifact."""

    result_path = Path(path)
    if result_path.suffix.lower() != ".json":
        raise ExperimentError(f"Experiment result must use the .json extension: {result_path}")
    try:
        content = result_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExperimentError(f"Unable to read experiment result {result_path}: {exc}") from exc
    if not content.strip():
        raise ExperimentError(f"Experiment result is empty: {result_path}")
    if content.startswith("\ufeff"):
        raise ExperimentError("Experiment result must not contain a UTF-8 BOM")
    try:
        loaded: object = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExperimentError(f"Invalid experiment result JSON in {result_path}: {exc}") from exc

    root = _require_mapping(loaded, context="experiment result")
    _require_exact_fields(root, expected=_RESULT_FIELDS, context="experiment result")
    schema_version = _require_int(root["schema_version"], name="schema_version")
    if schema_version != EXPERIMENT_RESULT_SCHEMA_VERSION:
        raise ExperimentError(
            f"Unsupported experiment result schema version {schema_version}; "
            f"supported version is {EXPERIMENT_RESULT_SCHEMA_VERSION}."
        )

    config = _load_config(root["config"])
    aggregate_metrics = _load_aggregate_metrics(root["aggregate_metrics"])
    question_results = _load_question_results(root["question_results"])
    environment = _load_environment(root["environment"], config=config)
    warnings = _require_string_list(root["warnings"], name="warnings")

    run_id = _require_text(root["run_id"], name="run_id")
    started_at = _require_utc_timestamp(root["started_at_utc"], name="started_at_utc")
    completed_at = _require_utc_timestamp(root["completed_at_utc"], name="completed_at_utc")
    git_commit_value = root["git_commit"]
    if git_commit_value is not None and (
        not isinstance(git_commit_value, str)
        or _GIT_COMMIT_PATTERN.fullmatch(git_commit_value) is None
    ):
        raise ExperimentError("git_commit must be null or a full lowercase Git hash")
    _normalise_git_provenance(
        git_commit=git_commit_value,
        environment=environment,
        warnings=warnings,
    )
    corpus_checksum = _require_sha256(root["corpus_checksum"], name="corpus_checksum")
    dataset_checksum = _require_sha256(
        root["evaluation_dataset_checksum"],
        name="evaluation_dataset_checksum",
    )

    _validate_aggregate_counts(aggregate_metrics, question_results)
    return ExperimentResult(
        run_id=run_id,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        git_commit=git_commit_value,
        corpus_checksum=corpus_checksum,
        evaluation_dataset_checksum=dataset_checksum,
        config=config,
        aggregate_metrics=aggregate_metrics,
        question_results=question_results,
        environment=environment,
        warnings=warnings,
    )


def reproducible_result_payload(result: ExperimentResult) -> dict[str, Any]:
    """Return deterministic content with run identity, timestamps, and latency removed."""

    payload = experiment_result_payload(result)
    for field_name in _VARIABLE_TOP_LEVEL_FIELDS:
        payload.pop(field_name, None)

    aggregate = payload.get("aggregate_metrics")
    if isinstance(aggregate, dict):
        payload["aggregate_metrics"] = {
            key: value for key, value in aggregate.items() if "latency" not in str(key)
        }

    question_results = payload.get("question_results")
    if isinstance(question_results, list):
        deterministic_questions: list[object] = []
        for item in question_results:
            if not isinstance(item, dict):
                deterministic_questions.append(item)
                continue
            deterministic_questions.append(
                {key: value for key, value in item.items() if key not in _VARIABLE_QUESTION_FIELDS}
            )
        payload["question_results"] = deterministic_questions
    return payload


def _load_config(value: object) -> ExperimentConfig:
    mapping = _require_mapping(value, context="experiment config")
    _require_exact_fields(mapping, expected=_CONFIG_FIELDS, context="experiment config")
    try:
        config = ExperimentConfig(
            name=_require_text(mapping["name"], name="config.name"),
            chunk_size=_require_positive_int(mapping["chunk_size"], name="config.chunk_size"),
            overlap=_require_non_negative_int(mapping["overlap"], name="config.overlap"),
            top_k=_require_positive_int(mapping["top_k"], name="config.top_k"),
            embedding_model=_require_text(
                mapping["embedding_model"], name="config.embedding_model"
            ),
            generation_model=_require_text(
                mapping["generation_model"], name="config.generation_model"
            ),
            temperature=_require_finite_number(mapping["temperature"], name="config.temperature"),
            prompt_version=_require_text(mapping["prompt_version"], name="config.prompt_version"),
            seed=_require_int(mapping["seed"], name="config.seed"),
            evaluation_mode=_require_text(
                mapping["evaluation_mode"], name="config.evaluation_mode"
            ),
            llm_judge_enabled=_require_bool(
                mapping["llm_judge_enabled"], name="config.llm_judge_enabled"
            ),
        )
    except ValueError as exc:
        raise ExperimentError(f"Invalid experiment config: {exc}") from exc
    if not 0.0 <= config.temperature <= 2.0:
        raise ExperimentError("config.temperature must be between 0 and 2")
    if config.evaluation_mode != "direct_rag":
        raise ExperimentError("config.evaluation_mode must be 'direct_rag'")
    if config.llm_judge_enabled:
        raise ExperimentError("config.llm_judge_enabled must be false for MVP results")
    return config


def _load_aggregate_metrics(value: object) -> dict[str, float]:
    mapping = _require_mapping(value, context="aggregate_metrics")
    supplied = frozenset(mapping)
    missing = sorted(_REQUIRED_AGGREGATE_FIELDS - supplied)
    unknown = sorted(supplied - _ALLOWED_AGGREGATE_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ExperimentError(f"aggregate_metrics has invalid fields ({'; '.join(details)})")

    metrics = {
        key: _require_finite_number(raw_value, name=f"aggregate_metrics.{key}")
        for key, raw_value in mapping.items()
    }
    for key in _REQUIRED_AGGREGATE_FIELDS:
        value_number = metrics[key]
        if value_number < 0 or not value_number.is_integer():
            raise ExperimentError(f"aggregate_metrics.{key} must be a non-negative count")
    for key in (
        "hit_rate_at_k",
        "mrr",
        "source_recall_at_k",
        "required_fact_coverage",
        "citation_validity",
        "citation_precision",
        "unanswerable_accuracy",
    ):
        metric = metrics.get(key)
        if metric is not None and not 0.0 <= metric <= 1.0:
            raise ExperimentError(f"aggregate_metrics.{key} must be between 0 and 1")
    for key, metric in metrics.items():
        if "latency" in key and metric < 0:
            raise ExperimentError(f"aggregate_metrics.{key} must not be negative")
    return metrics


def _load_question_results(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ExperimentError("question_results must be a list")
    parsed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        context = f"question_results[{index - 1}]"
        mapping = _require_mapping(item, context=context)
        _require_exact_fields(mapping, expected=_QUESTION_FIELDS, context=context)
        question_id = _require_text(mapping["question_id"], name=f"{context}.question_id")
        if question_id in seen_ids:
            raise ExperimentError(f"Duplicate question result id: {question_id}")
        seen_ids.add(question_id)
        _validate_question_result(mapping, context=context)
        parsed.append(dict(mapping))
    return parsed


def _validate_question_result(values: dict[str, object], *, context: str) -> None:
    _require_text(values["question"], name=f"{context}.question")
    _require_text(values["expected_answer"], name=f"{context}.expected_answer")
    _require_bool(values["answerable"], name=f"{context}.answerable")
    difficulty = _require_text(values["difficulty"], name=f"{context}.difficulty")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ExperimentError(f"{context}.difficulty is unsupported")
    for field_name in (
        "tags",
        "expected_sources",
        "required_facts",
        "retrieved_sources",
        "citations",
        "warnings",
    ):
        _require_string_list(values[field_name], name=f"{context}.{field_name}")

    status = _require_text(values["status"], name=f"{context}.status")
    if status not in {"completed", "failed"}:
        raise ExperimentError(f"{context}.status must be 'completed' or 'failed'")
    first_rank = values["first_relevant_rank"]
    if first_rank is not None:
        _require_positive_int(first_rank, name=f"{context}.first_relevant_rank")
    _require_text_allow_empty(values["generated_answer"], name=f"{context}.generated_answer")

    chunks = values["retrieved_chunks"]
    if not isinstance(chunks, list):
        raise ExperimentError(f"{context}.retrieved_chunks must be a list")
    for chunk_index, chunk in enumerate(chunks):
        chunk_context = f"{context}.retrieved_chunks[{chunk_index}]"
        chunk_mapping = _require_mapping(chunk, context=chunk_context)
        _require_exact_fields(
            chunk_mapping,
            expected=_RETRIEVED_CHUNK_FIELDS,
            context=chunk_context,
        )
        _require_positive_int(chunk_mapping["rank"], name=f"{chunk_context}.rank")
        _require_finite_number(chunk_mapping["score"], name=f"{chunk_context}.score")
        for field_name in ("source_name", "source_id", "chunk_id", "citation_label"):
            _require_text(chunk_mapping[field_name], name=f"{chunk_context}.{field_name}")

    for field_name in (
        "hit_rate_at_k",
        "reciprocal_rank",
        "source_recall_at_k",
        "fact_coverage",
        "citation_validity",
        "citation_precision",
    ):
        metric = _require_optional_finite_number(values[field_name], name=f"{context}.{field_name}")
        if metric is not None and not 0.0 <= metric <= 1.0:
            raise ExperimentError(f"{context}.{field_name} must be between 0 and 1")

    for field_name in (
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
    ):
        latency = _require_optional_finite_number(
            values[field_name], name=f"{context}.{field_name}"
        )
        if latency is not None and latency < 0:
            raise ExperimentError(f"{context}.{field_name} must not be negative")

    categories = _require_string_list(
        values["failure_categories"], name=f"{context}.failure_categories"
    )
    unsupported_categories = sorted(set(categories) - _FAILURE_CATEGORIES)
    if unsupported_categories:
        raise ExperimentError(
            f"{context}.failure_categories contains unsupported values: "
            + ", ".join(unsupported_categories)
        )

    abstention = values["abstention_result"]
    error = values["error"]
    if status == "completed":
        _load_abstention(abstention, context=f"{context}.abstention_result")
        if error is not None:
            raise ExperimentError(f"{context}.error must be null for a completed result")
        for field_name in (
            "retrieval_latency_ms",
            "generation_latency_ms",
            "total_latency_ms",
        ):
            if values[field_name] is None:
                raise ExperimentError(f"{context}.{field_name} is required for a completed result")
    else:
        if abstention is not None:
            raise ExperimentError(f"{context}.abstention_result must be null after failure")
        error_mapping = _require_mapping(error, context=f"{context}.error")
        _require_exact_fields(
            error_mapping,
            expected=_ERROR_FIELDS,
            context=f"{context}.error",
        )
        _require_text(error_mapping["type"], name=f"{context}.error.type")
        _require_text(error_mapping["message"], name=f"{context}.error.message")


def _load_abstention(value: object, *, context: str) -> None:
    mapping = _require_mapping(value, context=context)
    _require_exact_fields(mapping, expected=_ABSTENTION_FIELDS, context=context)
    _require_bool(mapping["expected_to_abstain"], name=f"{context}.expected_to_abstain")
    _require_bool(mapping["explicit_abstention"], name=f"{context}.explicit_abstention")
    _require_bool(
        mapping["contains_additional_claims"],
        name=f"{context}.contains_additional_claims",
    )
    correct = mapping["correct"]
    if correct is not None:
        _require_bool(correct, name=f"{context}.correct")


def _load_environment(
    value: object,
    *,
    config: ExperimentConfig,
) -> dict[str, str]:
    mapping = _require_mapping(value, context="environment")
    missing = sorted(_REQUIRED_ENVIRONMENT_FIELDS - frozenset(mapping))
    if missing:
        raise ExperimentError("environment is missing: " + ", ".join(missing))
    parsed: dict[str, str] = {}
    for key, raw_value in mapping.items():
        if _SENSITIVE_KEY_PATTERN.search(key):
            raise ExperimentError(f"Sensitive environment field is not allowed: {key}")
        parsed[key] = _require_text(raw_value, name=f"environment.{key}")
    if parsed["embedding_model"] != config.embedding_model:
        raise ExperimentError("environment.embedding_model conflicts with config")
    if parsed["generation_model"] != config.generation_model:
        raise ExperimentError("environment.generation_model conflicts with config")
    return parsed


def _normalise_git_provenance(
    *,
    git_commit: str | None,
    environment: dict[str, str],
    warnings: list[str],
) -> None:
    state = environment.get(GIT_WORKTREE_STATE_FIELD)
    if state is None:
        state = "unverified"
        environment[GIT_WORKTREE_STATE_FIELD] = state
        if LEGACY_GIT_PROVENANCE_WARNING not in warnings:
            warnings.append(LEGACY_GIT_PROVENANCE_WARNING)
    elif state not in GIT_WORKTREE_STATES:
        allowed = ", ".join(sorted(GIT_WORKTREE_STATES))
        raise ExperimentError(f"environment.{GIT_WORKTREE_STATE_FIELD} must be one of: {allowed}")

    if state == "clean" and git_commit is None:
        raise ExperimentError("A clean Git worktree requires git_commit")
    if state in {"dirty", "unavailable"} and git_commit is not None:
        raise ExperimentError(f"git_commit must be null when Git worktree state is {state}")


def _validate_aggregate_counts(
    aggregate_metrics: dict[str, float],
    question_results: list[dict[str, Any]],
) -> None:
    total = int(aggregate_metrics["question_count"])
    successful = int(aggregate_metrics["successful_question_count"])
    failed = int(aggregate_metrics["failed_question_count"])
    if total != len(question_results) or successful + failed != total:
        raise ExperimentError("Aggregate question counts do not match question_results")
    successful_by_answerability = int(
        aggregate_metrics["successful_answerable_question_count"]
        + aggregate_metrics["successful_unanswerable_question_count"]
    )
    if successful_by_answerability != successful:
        raise ExperimentError("Aggregate answerability counts do not match successful count")
    if int(aggregate_metrics["failure_configuration_runtime_count"]) != failed:
        raise ExperimentError("Aggregate runtime failure count does not match failed count")
    actual_successful = sum(item["status"] == "completed" for item in question_results)
    if successful != actual_successful:
        raise ExperimentError("Aggregate successful count does not match question_results")


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExperimentError(f"{context} must be an object with string keys")
    return {str(key): item for key, item in value.items()}


def _require_exact_fields(
    values: dict[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    supplied = frozenset(values)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unknown:
        details.append("unknown: " + ", ".join(unknown))
    raise ExperimentError(f"{context} has invalid fields ({'; '.join(details)})")


def _require_text(value: object, *, name: str) -> str:
    parsed = _require_text_allow_empty(value, name=name)
    if not parsed:
        raise ExperimentError(f"{name} must not be empty")
    return parsed


def _require_text_allow_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ExperimentError(f"{name} must be a trimmed string")
    return value


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentError(f"{name} must be an integer")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    parsed = _require_int(value, name=name)
    if parsed <= 0:
        raise ExperimentError(f"{name} must be positive")
    return parsed


def _require_non_negative_int(value: object, *, name: str) -> int:
    parsed = _require_int(value, name=name)
    if parsed < 0:
        raise ExperimentError(f"{name} must not be negative")
    return parsed


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExperimentError(f"{name} must be a finite number")
    return parsed


def _require_optional_finite_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_number(value, name=name)


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentError(f"{name} must be a boolean")
    return value


def _require_string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ExperimentError(f"{name} must be a list")
    parsed = [_require_text(item, name=name) for item in value]
    return parsed


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ExperimentError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _require_utc_timestamp(value: object, *, name: str) -> str:
    parsed = _require_text(value, name=name)
    if not parsed.endswith("Z"):
        raise ExperimentError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(parsed[:-1] + "+00:00")
    except ValueError as exc:
        raise ExperimentError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    return parsed


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


__all__ = [
    "EXPERIMENT_RESULT_SCHEMA_VERSION",
    "experiment_result_payload",
    "load_experiment_result",
    "reproducible_result_payload",
]
