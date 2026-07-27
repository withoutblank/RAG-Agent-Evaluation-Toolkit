"""Benchmark configuration schema and matrix tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_agent_eval_toolkit.exceptions import ConfigurationError
from rag_agent_eval_toolkit.experiments.config import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    load_benchmark_config,
)


def _config_mapping() -> dict[str, object]:
    loaded = yaml.safe_load(Path("configs/benchmark.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_config(path: Path, mapping: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(mapping, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def test_load_required_four_configuration_matrix() -> None:
    suite = load_benchmark_config(Path("configs/benchmark.yaml"))

    assert suite.schema_version == BENCHMARK_CONFIG_SCHEMA_VERSION
    assert suite.provider == "fake"
    assert suite.corpus_dir.as_posix() == "sample_data/fictional_ev_support"
    assert suite.evaluation_dataset.as_posix() == ("evals/fictional_ev_support_questions.jsonl")
    assert [experiment.name for experiment in suite.experiments] == [
        "small-k3",
        "large-k3",
        "small-k5",
        "large-k5",
    ]
    assert {
        (experiment.chunk_size, experiment.overlap, experiment.top_k)
        for experiment in suite.experiments
    } == {(300, 50, 3), (600, 100, 3), (300, 50, 5), (600, 100, 5)}
    assert all(experiment.temperature == 0 for experiment in suite.experiments)
    assert all(not experiment.llm_judge_enabled for experiment in suite.experiments)


def test_duplicate_experiment_name_is_rejected(tmp_path: Path) -> None:
    mapping = _config_mapping()
    experiments = mapping["experiments"]
    assert isinstance(experiments, list)
    assert isinstance(experiments[1], dict)
    experiments[1]["name"] = "small-k3"
    path = tmp_path / "benchmark.yaml"
    _write_config(path, mapping)

    with pytest.raises(ConfigurationError, match="Duplicate experiment name"):
        load_benchmark_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda mapping: mapping.update(schema_version=2), "Unsupported benchmark schema"),
        (lambda mapping: mapping.update(corpus_dir="../private"), "project-relative"),
        (lambda mapping: mapping.update(provider="unknown"), "provider"),
        (lambda mapping: mapping.update(llm_judge_enabled=True), "must be disabled"),
        (lambda mapping: mapping.update(unexpected=True), "unknown: unexpected"),
        (lambda mapping: mapping.update(temperature=float("nan")), "finite number"),
    ],
)
def test_invalid_benchmark_configuration_is_rejected(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    mapping = _config_mapping()
    assert callable(mutation)
    mutation(mapping)
    path = tmp_path / "benchmark.yaml"
    _write_config(path, mapping)

    with pytest.raises(ConfigurationError, match=message):
        load_benchmark_config(path)


def test_exact_four_experiments_are_required(tmp_path: Path) -> None:
    mapping = _config_mapping()
    experiments = mapping["experiments"]
    assert isinstance(experiments, list)
    mapping["experiments"] = experiments[:3]
    path = tmp_path / "benchmark.yaml"
    _write_config(path, mapping)

    with pytest.raises(ConfigurationError, match="exactly 4 experiments"):
        load_benchmark_config(path)


def test_invalid_experiment_values_are_rejected(tmp_path: Path) -> None:
    mapping = _config_mapping()
    experiments = mapping["experiments"]
    assert isinstance(experiments, list)
    assert isinstance(experiments[0], dict)
    experiments[0]["overlap"] = experiments[0]["chunk_size"]
    path = tmp_path / "benchmark.yaml"
    _write_config(path, mapping)

    with pytest.raises(ConfigurationError, match="overlap"):
        load_benchmark_config(path)
