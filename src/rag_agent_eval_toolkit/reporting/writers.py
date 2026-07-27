"""Durable, deterministic benchmark result writers."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from rag_agent_eval_toolkit.exceptions import ExperimentError
from rag_agent_eval_toolkit.experiments.provenance import GIT_WORKTREE_STATE_FIELD
from rag_agent_eval_toolkit.experiments.results import experiment_result_payload
from rag_agent_eval_toolkit.models import ExperimentResult

_SAFE_CONFIG_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

SUMMARY_COLUMNS = (
    "configuration",
    "chunk_size",
    "overlap",
    "top_k",
    "hit_rate_at_k",
    "mrr",
    "source_recall_at_k",
    "required_fact_coverage",
    "citation_validity",
    "citation_precision",
    "unanswerable_accuracy",
    "median_retrieval_latency_ms",
    "median_total_latency_ms",
    "successful_question_count",
    "failed_question_count",
)


@dataclass(frozen=True, slots=True)
class BenchmarkOutputPaths:
    """Paths written for one complete benchmark reporting operation."""

    experiment_json: tuple[Path, ...]
    summary_csv: Path
    markdown_report: Path


def write_experiment_json(result: ExperimentResult, path: Path) -> Path:
    """Write one complete schema-versioned result as strict UTF-8 JSON."""

    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ExperimentError(f"Experiment result path must use .json: {destination}")
    try:
        content = json.dumps(
            experiment_result_payload(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            f"Experiment result {result.config.name!r} is not valid JSON data."
        ) from exc
    _atomic_write_text(destination, content + "\n")
    return destination


def write_summary_csv(
    results: Sequence[ExperimentResult],
    path: Path,
) -> Path:
    """Write one stable summary row per configuration."""

    validated = _validated_results(results)
    destination = Path(path)
    if destination.suffix.lower() != ".csv":
        raise ExperimentError(f"Benchmark summary path must use .csv: {destination}")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for result in validated:
        writer.writerow(_summary_row(result))
    _atomic_write_text(destination, stream.getvalue())
    return destination


def render_markdown_report(
    results: Sequence[ExperimentResult],
    *,
    title: str = "RAG Agent Evaluation Benchmark",
) -> str:
    """Render an evidence-backed Markdown report without a composite score."""

    validated = _validated_results(results)
    if not title or title != title.strip():
        raise ExperimentError("Report title must be a non-empty, trimmed string")

    lines = [
        f"# {title}",
        "",
        "> This benchmark uses a small fictional corpus. Results demonstrate an "
        "evaluation method, not model superiority, and do not generalise to production data. "
        "API-backed results may vary.",
        "",
        "## Provenance",
        "",
        "| Configuration | Run ID | Git commit | Git worktree | Corpus SHA-256 | "
        "Dataset SHA-256 | Python | Models |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in validated:
        environment = result.environment
        model_text = (
            f"{environment.get('embedding_model', result.config.embedding_model)} / "
            f"{environment.get('generation_model', result.config.generation_model)}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(result.config.name),
                    _markdown_text(result.run_id),
                    _markdown_text(result.git_commit or "unavailable"),
                    _markdown_text(environment.get(GIT_WORKTREE_STATE_FIELD, "unverified")),
                    _markdown_text(result.corpus_checksum),
                    _markdown_text(result.evaluation_dataset_checksum),
                    _markdown_text(environment.get("python_version", "unavailable")),
                    _markdown_text(model_text),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Experiment result schema version: `1`. Metrics are macro-averaged over "
            "eligible successful questions: retrieval, fact, and citation metrics use "
            "answerable questions; unanswerable accuracy uses unanswerable questions. "
            "Failed runtime questions are excluded and counted explicitly.",
            "A Git commit identifies the run inputs only when the Git worktree state is "
            "`clean`; `dirty`, `unavailable`, and `unverified` states are not reproducible "
            "from that field alone.",
            "",
            "## Configuration summary",
            "",
            "| Configuration | Chunk size | Overlap | Top-k | Hit Rate@k | MRR | "
            "Source Recall@k | Fact coverage | Citation validity | Citation precision | "
            "Unanswerable accuracy | Median retrieval (ms) | Median total (ms) | "
            "Completed | Failed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in validated:
        row = _summary_row(result)
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(row["configuration"]),
                    str(row["chunk_size"]),
                    str(row["overlap"]),
                    str(row["top_k"]),
                    _markdown_metric(row["hit_rate_at_k"]),
                    _markdown_metric(row["mrr"]),
                    _markdown_metric(row["source_recall_at_k"]),
                    _markdown_metric(row["required_fact_coverage"]),
                    _markdown_metric(row["citation_validity"]),
                    _markdown_metric(row["citation_precision"]),
                    _markdown_metric(row["unanswerable_accuracy"]),
                    _markdown_number(row["median_retrieval_latency_ms"]),
                    _markdown_number(row["median_total_latency_ms"]),
                    _markdown_number(row["successful_question_count"], decimals=0),
                    _markdown_number(row["failed_question_count"], decimals=0),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Interpretation", ""])
    lines.extend(_interpretation_lines(validated))
    lines.extend(
        [
            "",
            "No composite winner is calculated. Comparisons above use each named metric "
            "directly. Chunk size and overlap change together, so differences cannot be "
            "attributed to chunk size alone.",
            "",
            "## Deterministic failure analysis",
            "",
            "Failure categories are assigned in order from available evidence: expected "
            "source absent from the loaded corpus (`ingestion`), no expected source in "
            "top-k (`retrieval`), retrieved evidence followed by fact/citation/abstention "
            "failure (`generation`), or an exception (`configuration/runtime`).",
            "",
            "| Category | Occurrences |",
            "|---|---:|",
        ]
    )
    failure_counts = _failure_counts(validated)
    for category in ("ingestion", "retrieval", "generation", "configuration/runtime"):
        lines.append(f"| `{category}` | {failure_counts[category]} |")

    consistent_failures = _consistently_failed_questions(validated)
    lines.extend(["", "### Questions failing in every configuration", ""])
    if consistent_failures:
        lines.extend(f"- `{question_id}`" for question_id in consistent_failures)
    else:
        lines.append("None.")

    lines.extend(["", "## Per-question results", ""])
    for result in validated:
        lines.extend(_per_question_section(result))

    lines.extend(
        [
            "## Latency caveat",
            "",
            "Latency is observational. Fake-provider timings are not evidence of live API "
            "performance. Mean and median use successful questions only. Nearest-rank p95 "
            "is emitted only with at least 20 successful observations; otherwise the run "
            "warning records that it was omitted. Wall-clock timings are excluded from "
            "exact reproducibility comparisons.",
            "",
            "## Run warnings",
            "",
        ]
    )
    any_warnings = False
    for result in validated:
        for warning in result.warnings:
            any_warnings = True
            lines.append(f"- `{_markdown_text(result.config.name)}`: {_markdown_text(warning)}")
    if not any_warnings:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## What to test next",
            "",
            "- Add questions targeting any consistently failing source terminology.",
            "- Inspect retrieval misses before changing prompts or generation models.",
            "- Repeat with a larger held-out corpus before interpreting tail latency.",
            "- Keep any future LLM-as-judge scores separate and labelled non-deterministic.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(
    results: Sequence[ExperimentResult],
    path: Path,
    *,
    title: str = "RAG Agent Evaluation Benchmark",
) -> Path:
    """Render and atomically write the benchmark Markdown report."""

    destination = Path(path)
    if destination.suffix.lower() not in {".md", ".markdown"}:
        raise ExperimentError(f"Benchmark report path must use .md: {destination}")
    _atomic_write_text(destination, render_markdown_report(results, title=title))
    return destination


def write_benchmark_outputs(
    results: Sequence[ExperimentResult],
    output_dir: Path,
    *,
    report_title: str = "RAG Agent Evaluation Benchmark",
) -> BenchmarkOutputPaths:
    """Write per-configuration JSON, summary CSV, and Markdown report."""

    validated = _validated_results(results)
    destination = Path(output_dir)
    experiment_paths: list[Path] = []
    for result in validated:
        if _SAFE_CONFIG_NAME_PATTERN.fullmatch(result.config.name) is None:
            raise ExperimentError(
                f"Unsafe experiment name cannot be used as a filename: {result.config.name!r}"
            )
        experiment_paths.append(
            write_experiment_json(result, destination / f"{result.config.name}.json")
        )

    summary_path = write_summary_csv(validated, destination / "summary.csv")
    report_path = write_markdown_report(
        validated,
        destination / "report.md",
        title=report_title,
    )
    return BenchmarkOutputPaths(
        experiment_json=tuple(experiment_paths),
        summary_csv=summary_path,
        markdown_report=report_path,
    )


def _summary_row(result: ExperimentResult) -> dict[str, str | int]:
    metrics = result.aggregate_metrics
    return {
        "configuration": result.config.name,
        "chunk_size": result.config.chunk_size,
        "overlap": result.config.overlap,
        "top_k": result.config.top_k,
        "hit_rate_at_k": _csv_number(metrics.get("hit_rate_at_k")),
        "mrr": _csv_number(metrics.get("mrr")),
        "source_recall_at_k": _csv_number(metrics.get("source_recall_at_k")),
        "required_fact_coverage": _csv_number(metrics.get("required_fact_coverage")),
        "citation_validity": _csv_number(metrics.get("citation_validity")),
        "citation_precision": _csv_number(metrics.get("citation_precision")),
        "unanswerable_accuracy": _csv_number(metrics.get("unanswerable_accuracy")),
        "median_retrieval_latency_ms": _csv_number(metrics.get("retrieval_latency_median_ms")),
        "median_total_latency_ms": _csv_number(metrics.get("total_latency_median_ms")),
        "successful_question_count": _csv_number(metrics.get("successful_question_count")),
        "failed_question_count": _csv_number(metrics.get("failed_question_count")),
    }


def _interpretation_lines(results: Sequence[ExperimentResult]) -> list[str]:
    lines: list[str] = []
    hit_values = [
        (result.config.name, result.aggregate_metrics.get("hit_rate_at_k")) for result in results
    ]
    defined_hits = [
        (name, value)
        for name, value in hit_values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if defined_hits:
        best_value = max(float(value) for _, value in defined_hits)
        names = [name for name, value in defined_hits if float(value) == best_value]
        lines.append(
            f"- Highest Hit Rate@k: **{', '.join(_markdown_text(name) for name in names)}** "
            f"at **{best_value:.3f}** (ties retained)."
        )
    else:
        lines.append("- Highest Hit Rate@k: unavailable because no eligible result exists.")

    chunk_groups = _group_metric(results, group_field="chunk_size", metric="required_fact_coverage")
    if len(chunk_groups) >= 2:
        smallest = min(chunk_groups)
        largest = max(chunk_groups)
        lines.append(
            f"- Required-fact coverage averaged **{chunk_groups[smallest]:.3f}** for "
            f"{smallest}-character chunks and **{chunk_groups[largest]:.3f}** for "
            f"{largest}-character chunks. This is an association between presets, not "
            "an isolated chunk-size effect."
        )
    else:
        lines.append("- Larger-chunk fact-coverage comparison: insufficient eligible results.")

    recall_by_k = _group_metric(results, group_field="top_k", metric="source_recall_at_k")
    precision_by_k = _group_metric(
        results,
        group_field="top_k",
        metric="citation_precision",
    )
    if 3 in recall_by_k and 5 in recall_by_k:
        precision_text = (
            f"; citation precision changed from {precision_by_k[3]:.3f} to {precision_by_k[5]:.3f}"
            if 3 in precision_by_k and 5 in precision_by_k
            else "; citation precision was unavailable"
        )
        lines.append(
            f"- Moving from top-k=3 to top-k=5 changed mean Source Recall@k from "
            f"**{recall_by_k[3]:.3f}** to **{recall_by_k[5]:.3f}**{precision_text}."
        )
    else:
        lines.append("- Top-k recall/precision comparison: insufficient eligible results.")
    return lines


def _group_metric(
    results: Sequence[ExperimentResult],
    *,
    group_field: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for result in results:
        raw_group = getattr(result.config, group_field)
        raw_value = result.aggregate_metrics.get(metric)
        if not isinstance(raw_group, int):
            continue
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            continue
        grouped.setdefault(raw_group, []).append(float(raw_value))
    return {group: fmean(values) for group, values in grouped.items()}


def _failure_counts(results: Sequence[ExperimentResult]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        for question in result.question_results:
            categories = question.get("failure_categories")
            if isinstance(categories, list):
                counts.update(str(category) for category in categories)
    return counts


def _consistently_failed_questions(
    results: Sequence[ExperimentResult],
) -> list[str]:
    failures_by_configuration: list[set[str]] = []
    for result in results:
        failures: set[str] = set()
        for question in result.question_results:
            question_id = question.get("question_id")
            categories = question.get("failure_categories")
            if isinstance(question_id, str) and isinstance(categories, list) and categories:
                failures.add(question_id)
        failures_by_configuration.append(failures)
    if not failures_by_configuration:
        return []
    return sorted(set.intersection(*failures_by_configuration))


def _per_question_section(result: ExperimentResult) -> list[str]:
    lines = [
        f"### {_markdown_text(result.config.name)}",
        "",
        "| Question | Status | Expected source(s) | First relevant rank | "
        "Retrieved source(s) | Generated answer | Citations | Fact coverage | "
        "Abstention | Total latency (ms) | Failure(s) | Warnings |",
        "|---|---|---|---:|---|---|---|---:|---|---:|---|---|",
    ]
    for question in result.question_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_text(
                        f"{question.get('question_id', '')}: {question.get('question', '')}"
                    ),
                    _markdown_text(question.get("status", "")),
                    _markdown_list(question.get("expected_sources")),
                    _markdown_number(question.get("first_relevant_rank"), decimals=0),
                    _markdown_list(question.get("retrieved_sources")),
                    _markdown_text(question.get("generated_answer", "")),
                    _markdown_list(question.get("citations")),
                    _markdown_metric(question.get("fact_coverage")),
                    _abstention_text(question.get("abstention_result")),
                    _markdown_number(question.get("total_latency_ms")),
                    _markdown_list(question.get("failure_categories")),
                    _markdown_list(question.get("warnings")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _abstention_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return "N/A"
    expected = value.get("expected_to_abstain")
    correct = value.get("correct")
    explicit = value.get("explicit_abstention")
    if expected is True:
        return "correct" if correct is True else "incorrect"
    return "false abstention" if explicit is True else "N/A"


def _markdown_list(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "N/A" if value is None else _markdown_text(value)
    if not value:
        return "—"
    return _markdown_text(", ".join(str(item) for item in value))


def _markdown_text(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _markdown_metric(value: object) -> str:
    return _markdown_number(value, decimals=3)


def _markdown_number(value: object, *, decimals: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, str) and value:
            try:
                parsed = float(value)
            except ValueError:
                return "N/A"
        else:
            return "N/A"
    else:
        parsed = float(value)
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:.{decimals}f}"


def _csv_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return format(value, ".12g")


def _validated_results(
    results: Sequence[ExperimentResult],
) -> tuple[ExperimentResult, ...]:
    if not results:
        raise ExperimentError("At least one experiment result is required")
    parsed = tuple(results)
    if not all(isinstance(result, ExperimentResult) for result in parsed):
        raise ExperimentError("All report inputs must be ExperimentResult instances")
    names = [result.config.name for result in parsed]
    if len(names) != len(set(names)):
        raise ExperimentError("Experiment result names must be unique")
    return tuple(sorted(parsed, key=lambda result: result.config.name))


def _atomic_write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        temporary = locals().get("temporary_path")
        if isinstance(temporary, Path):
            temporary.unlink(missing_ok=True)
        raise ExperimentError(f"Unable to write benchmark artifact {path}: {exc}") from exc


__all__ = [
    "SUMMARY_COLUMNS",
    "BenchmarkOutputPaths",
    "render_markdown_report",
    "write_benchmark_outputs",
    "write_experiment_json",
    "write_markdown_report",
    "write_summary_csv",
]
