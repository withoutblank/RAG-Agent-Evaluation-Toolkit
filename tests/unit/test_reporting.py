"""Experiment JSON, summary CSV, and Markdown report tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rag_agent_eval_toolkit.exceptions import ExperimentError
from rag_agent_eval_toolkit.models import ExperimentConfig, ExperimentResult
from rag_agent_eval_toolkit.reporting import (
    render_markdown_report,
    write_benchmark_outputs,
    write_experiment_json,
)


def _result(
    *,
    name: str = "small-k3",
    failed: bool = False,
    aggregate_overrides: dict[str, float] | None = None,
) -> ExperimentResult:
    aggregate = {
        "question_count": 2.0,
        "successful_question_count": 1.0 if failed else 2.0,
        "failed_question_count": 1.0 if failed else 0.0,
        "hit_rate_at_k": 1.0,
        "mrr": 0.75,
        "source_recall_at_k": 0.5,
        "required_fact_coverage": 0.8,
        "citation_validity": 1.0,
        "citation_precision": 0.5,
        "unanswerable_accuracy": 1.0,
        "retrieval_latency_median_ms": 2.5,
        "total_latency_median_ms": 7.5,
    }
    aggregate.update(aggregate_overrides or {})
    question_result = {
        "question_id": "ev-test",
        "question": "What is documented?",
        "expected_sources": ["guide.md"],
        "status": "failed" if failed else "completed",
        "first_relevant_rank": None if failed else 1,
        "retrieved_sources": [] if failed else ["guide.md"],
        "generated_answer": "" if failed else "Fact [guide.md#aaaaaaaaaaaaaaaaaaaa]",
        "citations": [] if failed else ["[guide.md#aaaaaaaaaaaaaaaaaaaa]"],
        "fact_coverage": None if failed else 1.0,
        "abstention_result": None,
        "total_latency_ms": None if failed else 7.5,
        "failure_categories": ["configuration/runtime"] if failed else [],
        "warnings": ["offline failure"] if failed else [],
    }
    return ExperimentResult(
        run_id=f"run-{name}",
        started_at_utc="2026-01-01T00:00:00Z",
        completed_at_utc="2026-01-01T00:00:01Z",
        git_commit="e" * 40,
        corpus_checksum="c" * 64,
        evaluation_dataset_checksum="d" * 64,
        config=ExperimentConfig(
            name=name,
            chunk_size=300,
            overlap=50,
            top_k=3,
            embedding_model="fake-hash-v1",
            generation_model="fake-cited-v1",
            seed=42,
        ),
        aggregate_metrics=aggregate,
        question_results=[question_result],
        environment={
            "python_version": "3.11.9",
            "operating_system": "Windows 11",
            "package_version": "0.1.0",
            "embedding_model": "fake-hash-v1",
            "generation_model": "fake-cited-v1",
            "git_worktree_state": "clean",
            "provider": "fake",
        },
        warnings=["retrieval_latency p95 omitted: 2 observations."],
    )


def test_write_all_benchmark_outputs_with_stable_schemas(tmp_path: Path) -> None:
    result = _result()

    paths = write_benchmark_outputs([result], tmp_path)

    assert paths.experiment_json == (tmp_path / "small-k3.json",)
    assert paths.summary_csv == tmp_path / "summary.csv"
    assert paths.markdown_report == tmp_path / "report.md"
    payload = json.loads(paths.experiment_json[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["config"]["name"] == "small-k3"
    assert payload["corpus_checksum"] == "c" * 64
    assert payload["evaluation_dataset_checksum"] == "d" * 64
    assert payload["environment"]["provider"] == "fake"

    with paths.summary_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["configuration"] == "small-k3"
    assert rows[0]["mrr"] == "0.75"
    assert rows[0]["median_total_latency_ms"] == "7.5"

    report = paths.markdown_report.read_text(encoding="utf-8")
    assert "# RAG Agent Evaluation Benchmark" in report
    assert "small fictional corpus" in report
    assert "No composite winner is calculated" in report
    assert "configuration/runtime" in report
    assert "Nearest-rank p95" in report
    assert "## Per-question results" in report
    assert "| Git commit | Git worktree |" in report
    assert f"| {'e' * 40} | clean |" in report


def test_partial_failure_is_clear_in_markdown_report() -> None:
    report = render_markdown_report([_result(failed=True)])

    assert "| ev-test: What is documented? | failed |" in report
    assert "`configuration/runtime` | 1" in report
    assert "`ev-test`" in report
    assert "offline failure" in report


def test_report_order_is_independent_of_input_order() -> None:
    """Saved JSON reload order cannot change a generated report."""

    first = _result(name="small-k3")
    second = _result(name="large-k3")

    assert render_markdown_report([first, second]) == render_markdown_report([second, first])


def test_json_writer_rejects_non_finite_metrics(tmp_path: Path) -> None:
    result = _result(aggregate_overrides={"mrr": float("nan")})

    with pytest.raises(ExperimentError, match="not valid JSON"):
        write_experiment_json(result, tmp_path / "result.json")


def test_report_rejects_duplicate_configuration_names() -> None:
    with pytest.raises(ExperimentError, match="names must be unique"):
        render_markdown_report([_result(), _result()])


def test_output_filename_rejects_unsafe_configuration_name(tmp_path: Path) -> None:
    result = _result(name="../unsafe")

    with pytest.raises(ExperimentError, match="Unsafe experiment name"):
        write_benchmark_outputs([result], tmp_path)
