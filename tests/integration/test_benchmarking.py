"""End-to-end coverage for concrete offline benchmark orchestration."""

from pathlib import Path

from rag_agent_eval_toolkit.benchmarking import (
    FAKE_EMBEDDING_DIMENSIONS,
    execute_benchmark,
    execute_evaluation,
)


def test_project_benchmark_runs_all_configurations_offline(tmp_path: Path) -> None:
    execution = execute_benchmark(
        Path("configs/benchmark.yaml"),
        provider="fake",
        output_dir=tmp_path,
    )

    assert len(execution.dataset.questions) == 20
    assert [result.config.name for result in execution.results] == [
        "small-k3",
        "large-k3",
        "small-k5",
        "large-k5",
    ]
    assert all(
        result.config.embedding_model == f"fake-hash-v3-d{FAKE_EMBEDDING_DIMENSIONS}-s42"
        for result in execution.results
    )
    assert len(execution.outputs.experiment_json) == 4
    assert execution.outputs.summary_csv.is_file()
    assert execution.outputs.markdown_report.is_file()


def test_project_evaluation_selects_one_configuration() -> None:
    result = execute_evaluation(
        Path("configs/benchmark.yaml"),
        experiment_name="large-k5",
        provider="fake",
    )

    assert result.config.name == "large-k5"
    assert result.aggregate_metrics["successful_question_count"] == 20.0
