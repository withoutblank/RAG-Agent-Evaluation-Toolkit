"""Injectable direct-RAG experiment runner tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rag_agent_eval_toolkit.evaluation.dataset import EvaluationDataset
from rag_agent_eval_toolkit.exceptions import ExperimentError
from rag_agent_eval_toolkit.experiments.config import load_benchmark_config
from rag_agent_eval_toolkit.experiments.results import (
    experiment_result_payload,
    load_experiment_result,
    reproducible_result_payload,
)
from rag_agent_eval_toolkit.experiments.runner import (
    DirectRagExecutor,
    detect_git_commit,
    detect_git_provenance,
    run_benchmark,
    run_experiment,
)
from rag_agent_eval_toolkit.models import (
    Citation,
    EvaluationQuestion,
    ExperimentConfig,
    ExperimentResult,
    RagAnswer,
    RetrievalResult,
)

_CHUNK_ID = "a" * 20
_CORPUS_CHECKSUM = "c" * 64
_DATASET_CHECKSUM = "d" * 64
_GIT_COMMIT = "e" * 40
_DIRTY_WARNING = (
    "Git worktree has tracked or untracked changes; git_commit was omitted because "
    "HEAD does not fully identify this run."
)
_UNAVAILABLE_WARNING = "Git commit and worktree state could not be determined for this run."
_UNVERIFIED_WARNING = "Git worktree state is unverified because git_commit was supplied explicitly."
_LEGACY_WARNING = "Git worktree state was not recorded; git_commit provenance is unverified."


def _questions() -> tuple[EvaluationQuestion, EvaluationQuestion]:
    return (
        EvaluationQuestion(
            id="answerable",
            question="What facts are documented?",
            expected_answer="Fact one and fact two.",
            expected_sources=["guide.md"],
            required_facts=["fact one", "fact two"],
            answerable=True,
            tags=["test"],
            difficulty="easy",
        ),
        EvaluationQuestion(
            id="unanswerable",
            question="What is the unknown price?",
            expected_answer="The documents do not contain the price.",
            expected_sources=[],
            required_facts=[],
            answerable=False,
            tags=["unanswerable"],
            difficulty="easy",
        ),
    )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        schema_version=1,
        path=Path("questions.jsonl"),
        checksum_sha256=_DATASET_CHECKSUM,
        questions=_questions(),
    )


def _config(name: str = "test-k3", *, top_k: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        chunk_size=300,
        overlap=50,
        top_k=top_k,
        embedding_model="fake-hash-v1",
        generation_model="fake-cited-v1",
        seed=42,
    )


def _answer(
    question: str,
    config: ExperimentConfig,
    *,
    latency_offset: float = 0.0,
) -> RagAnswer:
    if question == "What facts are documented?":
        label = f"[guide.md#{_CHUNK_ID}]"
        retrieval_result = RetrievalResult(
            query=question,
            rank=1,
            score=0.9,
            chunk_id=_CHUNK_ID,
            source_id="guide-source",
            source_name="guide.md",
            text="Fact one and fact two.",
            page_number=None,
            citation_label=label,
        )
        return RagAnswer(
            question=question,
            answer=f"Fact one and fact two. {label}",
            citations=[
                Citation(
                    citation_label=label,
                    source_id="guide-source",
                    source_name="guide.md",
                    chunk_id=_CHUNK_ID,
                )
            ],
            retrieval_results=[retrieval_result],
            model_name=config.generation_model,
            prompt_version=config.prompt_version,
            retrieval_latency_ms=1.0 + latency_offset,
            generation_latency_ms=2.0 + latency_offset,
            total_latency_ms=3.0 + latency_offset,
        )
    return RagAnswer(
        question=question,
        answer="The provided documents do not contain an unknown price.",
        citations=[],
        retrieval_results=[],
        model_name=config.generation_model,
        prompt_version=config.prompt_version,
        retrieval_latency_ms=4.0 + latency_offset,
        generation_latency_ms=5.0 + latency_offset,
        total_latency_ms=9.0 + latency_offset,
    )


def _run(
    executor: DirectRagExecutor = _answer,
    *,
    run_id: str = "run-one",
    timestamp: str = "2026-01-01T00:00:00Z",
) -> ExperimentResult:
    return run_experiment(
        _config(),
        _dataset(),
        executor,
        corpus_checksum=_CORPUS_CHECKSUM,
        available_sources={"guide.md"},
        git_commit=_GIT_COMMIT,
        environment={"provider": "fake"},
        timestamp_factory=lambda: timestamp,
        run_id_factory=lambda: run_id,
    )


def _committed_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / ".gitignore").write_text("ignored-output.txt\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=Benchmark Test",
        "-c",
        "user.email=benchmark@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "test fixture",
    )
    return repo, _git(repo, "rev-parse", "HEAD").strip().lower()


def _git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    return process.stdout


def test_experiment_runner_calculates_separated_aggregate_metrics() -> None:
    result = _run()

    assert result.aggregate_metrics["hit_rate_at_k"] == 1.0
    assert result.aggregate_metrics["mrr"] == 1.0
    assert result.aggregate_metrics["source_recall_at_k"] == 1.0
    assert result.aggregate_metrics["required_fact_coverage"] == 1.0
    assert result.aggregate_metrics["citation_validity"] == 1.0
    assert result.aggregate_metrics["citation_precision"] == 1.0
    assert result.aggregate_metrics["unanswerable_accuracy"] == 1.0
    assert result.aggregate_metrics["retrieval_latency_mean_ms"] == 2.5
    assert result.aggregate_metrics["retrieval_latency_median_ms"] == 2.5
    assert result.aggregate_metrics["total_latency_median_ms"] == 6.0
    assert "retrieval_latency_p95_ms" not in result.aggregate_metrics
    assert any("p95 omitted" in warning for warning in result.warnings)
    assert result.corpus_checksum == _CORPUS_CHECKSUM
    assert result.evaluation_dataset_checksum == _DATASET_CHECKSUM
    assert result.git_commit == _GIT_COMMIT
    assert result.environment["provider"] == "fake"
    assert result.environment["git_worktree_state"] == "unverified"
    assert _UNVERIFIED_WARNING in result.warnings
    assert all(item["failure_categories"] == [] for item in result.question_results)


def test_clean_git_worktree_records_verified_commit(tmp_path: Path) -> None:
    repo, expected_commit = _committed_git_repo(tmp_path)
    (repo / "ignored-output.txt").write_text("ignored\n", encoding="utf-8")

    provenance = detect_git_provenance(repo)
    result = run_experiment(
        _config(),
        _dataset(),
        _answer,
        corpus_checksum=_CORPUS_CHECKSUM,
        repo_dir=repo,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: "clean-run",
    )

    assert provenance.commit == expected_commit
    assert provenance.worktree_state == "clean"
    assert detect_git_commit(repo) == expected_commit
    assert result.git_commit == expected_commit
    assert result.environment["git_worktree_state"] == "clean"
    assert _DIRTY_WARNING not in result.warnings


@pytest.mark.parametrize("change_kind", ["tracked", "untracked"])
def test_dirty_git_worktree_omits_commit_and_records_state(
    tmp_path: Path,
    change_kind: str,
) -> None:
    repo, _expected_commit = _committed_git_repo(tmp_path)
    if change_kind == "tracked":
        (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    else:
        (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    provenance = detect_git_provenance(repo)
    result = run_experiment(
        _config(),
        _dataset(),
        _answer,
        corpus_checksum=_CORPUS_CHECKSUM,
        repo_dir=repo,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: f"dirty-{change_kind}",
    )

    assert provenance.commit is None
    assert provenance.worktree_state == "dirty"
    assert detect_git_commit(repo) is None
    assert result.git_commit is None
    assert result.environment["git_worktree_state"] == "dirty"
    assert _DIRTY_WARNING in result.warnings


def test_missing_git_repository_records_unavailable_state(tmp_path: Path) -> None:
    result = run_experiment(
        _config(),
        _dataset(),
        _answer,
        corpus_checksum=_CORPUS_CHECKSUM,
        repo_dir=tmp_path,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: "no-git-run",
    )

    assert result.git_commit is None
    assert result.environment["git_worktree_state"] == "unavailable"
    assert _UNAVAILABLE_WARNING in result.warnings


def test_executor_receives_question_text_without_evaluation_labels() -> None:
    received: list[object] = []

    def inspecting_executor(question: str, config: ExperimentConfig) -> RagAnswer:
        received.append(question)
        return _answer(question, config)

    _run(inspecting_executor)

    assert received == [
        "What facts are documented?",
        "What is the unknown price?",
    ]
    assert all(isinstance(value, str) for value in received)


def test_partial_failure_is_serialized_and_excluded_from_quality_aggregates() -> None:
    def failing_executor(
        question: str,
        config: ExperimentConfig,
    ) -> RagAnswer:
        if question == "What is the unknown price?":
            raise RuntimeError("provider token=must-not-be-serialized")
        return _answer(question, config)

    result = _run(failing_executor)

    assert result.aggregate_metrics["successful_question_count"] == 1.0
    assert result.aggregate_metrics["failed_question_count"] == 1.0
    assert "unanswerable_accuracy" not in result.aggregate_metrics
    failed = result.question_results[1]
    assert failed["status"] == "failed"
    assert failed["failure_categories"] == ["configuration/runtime"]
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "Direct-RAG execution failed.",
    }
    assert "must-not-be-serialized" not in str(failed)
    assert any("failed at runtime" in warning for warning in result.warnings)


def test_continue_on_error_false_raises_domain_error() -> None:
    def failing_executor(
        _question: str,
        _config_value: ExperimentConfig,
    ) -> RagAnswer:
        raise RuntimeError("offline failure")

    with pytest.raises(ExperimentError, match="failed on question"):
        run_experiment(
            _config(),
            _dataset(),
            failing_executor,
            corpus_checksum=_CORPUS_CHECKSUM,
            git_commit=_GIT_COMMIT,
            continue_on_error=False,
        )


def test_reproducibility_projection_excludes_identity_timestamps_and_latency() -> None:
    first = run_experiment(
        _config(),
        _dataset(),
        lambda question, config: _answer(question, config, latency_offset=0),
        corpus_checksum=_CORPUS_CHECKSUM,
        git_commit=_GIT_COMMIT,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: "first-run",
    )
    second = run_experiment(
        _config(),
        _dataset(),
        lambda question, config: _answer(question, config, latency_offset=100),
        corpus_checksum=_CORPUS_CHECKSUM,
        git_commit=_GIT_COMMIT,
        timestamp_factory=lambda: "2026-02-01T00:00:00Z",
        run_id_factory=lambda: "second-run",
    )

    assert first.run_id != second.run_id
    assert (
        first.aggregate_metrics["total_latency_mean_ms"]
        != (second.aggregate_metrics["total_latency_mean_ms"])
    )
    assert reproducible_result_payload(first) == reproducible_result_payload(second)


def test_experiment_result_json_round_trip(tmp_path: Path) -> None:
    result = _run()
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(experiment_result_payload(result), allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )

    loaded = load_experiment_result(path)

    assert loaded == result


def test_legacy_result_loads_with_unverified_git_provenance(tmp_path: Path) -> None:
    payload = experiment_result_payload(_run())
    payload["environment"].pop("git_worktree_state")
    payload["warnings"] = [
        warning for warning in payload["warnings"] if warning != _UNVERIFIED_WARNING
    ]
    path = tmp_path / "legacy-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    loaded = load_experiment_result(path)

    assert loaded.git_commit == _GIT_COMMIT
    assert loaded.environment["git_worktree_state"] == "unverified"
    assert _LEGACY_WARNING in loaded.warnings


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(schema_version=2),
            "Unsupported experiment result schema",
        ),
        (
            lambda payload: payload["config"].update(unexpected=True),
            "unknown: unexpected",
        ),
        (
            lambda payload: payload["aggregate_metrics"].update(mrr=float("nan")),
            "non-standard JSON constant",
        ),
        (
            lambda payload: payload["environment"].update(generation_model="different-model"),
            "conflicts with config",
        ),
        (
            lambda payload: payload["question_results"][0].update(status="unknown"),
            "status must be",
        ),
        (
            lambda payload: payload["environment"].update(git_worktree_state="invented"),
            "git_worktree_state must be one of",
        ),
        (
            lambda payload: (
                payload.update(git_commit=None),
                payload["environment"].update(git_worktree_state="clean"),
            ),
            "clean Git worktree requires git_commit",
        ),
        (
            lambda payload: payload["environment"].update(git_worktree_state="dirty"),
            "git_commit must be null",
        ),
    ],
)
def test_experiment_result_loader_rejects_malformed_artifacts(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = experiment_result_payload(_run())
    assert callable(mutation)
    mutation(payload)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(ExperimentError, match=message):
        load_experiment_result(path)


def test_all_four_configurations_execute_in_yaml_order() -> None:
    suite = load_benchmark_config(Path("configs/benchmark.yaml"))
    prepared: list[str] = []

    def factory(config: ExperimentConfig) -> DirectRagExecutor:
        prepared.append(config.name)
        return _answer

    results = run_benchmark(
        suite,
        _dataset(),
        factory,
        corpus_checksum=_CORPUS_CHECKSUM,
        available_sources={"guide.md"},
        git_commit=_GIT_COMMIT,
        timestamp_factory=lambda: "2026-01-01T00:00:00Z",
        run_id_factory=lambda: "fixed-id",
    )

    assert len(results) == 4
    assert prepared == [
        "small-k3",
        "large-k3",
        "small-k5",
        "large-k5",
    ]
    assert [result.config.name for result in results] == prepared
    assert all(result.aggregate_metrics["successful_question_count"] == 2.0 for result in results)


def test_sensitive_environment_fields_are_rejected() -> None:
    with pytest.raises(ExperimentError, match="Sensitive environment"):
        run_experiment(
            _config(),
            _dataset(),
            _answer,
            corpus_checksum=_CORPUS_CHECKSUM,
            git_commit=_GIT_COMMIT,
            environment={"api_key": "must-not-appear"},
        )


def test_conflicting_model_provenance_becomes_a_clear_partial_failure() -> None:
    def mismatched_executor(question: str, config: ExperimentConfig) -> RagAnswer:
        answer = _answer(question, config)
        return RagAnswer(
            question=answer.question,
            answer=answer.answer,
            citations=answer.citations,
            retrieval_results=answer.retrieval_results,
            model_name="different-model",
            prompt_version=answer.prompt_version,
            retrieval_latency_ms=answer.retrieval_latency_ms,
            generation_latency_ms=answer.generation_latency_ms,
            total_latency_ms=answer.total_latency_ms,
        )

    result = _run(mismatched_executor)

    assert result.aggregate_metrics["successful_question_count"] == 0.0
    assert result.aggregate_metrics["failed_question_count"] == 2.0
    assert all(
        question["failure_categories"] == ["configuration/runtime"]
        for question in result.question_results
    )
