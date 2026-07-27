"""Strict schema-versioned loading for the four-configuration benchmark."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from rag_agent_eval_toolkit.exceptions import ConfigurationError
from rag_agent_eval_toolkit.models import ExperimentConfig

BENCHMARK_CONFIG_SCHEMA_VERSION = 1
REQUIRED_EXPERIMENT_COUNT = 4

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "corpus_dir",
        "evaluation_dataset",
        "results_dir",
        "provider",
        "embedding_model",
        "generation_model",
        "temperature",
        "prompt_version",
        "seed",
        "evaluation_mode",
        "llm_judge_enabled",
        "experiments",
    }
)
_EXPERIMENT_FIELDS = frozenset({"name", "chunk_size", "overlap", "top_k"})
_EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteConfig:
    """One controlled benchmark suite and its four experiment configurations."""

    schema_version: int
    corpus_dir: Path
    evaluation_dataset: Path
    results_dir: Path
    provider: str
    experiments: tuple[ExperimentConfig, ...]


def load_benchmark_config(
    path: Path,
    *,
    expected_experiment_count: int = REQUIRED_EXPERIMENT_COUNT,
) -> BenchmarkSuiteConfig:
    """Load and strictly validate the benchmark YAML configuration."""

    if (
        isinstance(expected_experiment_count, bool)
        or not isinstance(expected_experiment_count, int)
        or expected_experiment_count < 1
    ):
        raise ValueError("expected_experiment_count must be a positive integer")

    config_path = Path(path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(
            f"Benchmark configuration must use a .yaml or .yml extension: {config_path}"
        )
    try:
        content = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(
            f"Unable to read benchmark configuration {config_path}: {exc}"
        ) from exc
    if not content.strip():
        raise ConfigurationError(f"Benchmark configuration is empty: {config_path}")

    try:
        loaded: object = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in benchmark configuration {config_path}: {exc}"
        ) from exc
    root = _require_mapping(loaded, context="benchmark configuration")
    _require_exact_fields(root, expected=_ROOT_FIELDS, context="benchmark configuration")

    schema_version = _require_int(root["schema_version"], name="schema_version")
    if schema_version != BENCHMARK_CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported benchmark schema version {schema_version}; "
            f"supported version is {BENCHMARK_CONFIG_SCHEMA_VERSION}."
        )

    corpus_dir = _require_portable_relative_path(root["corpus_dir"], name="corpus_dir")
    evaluation_dataset = _require_portable_relative_path(
        root["evaluation_dataset"], name="evaluation_dataset"
    )
    if evaluation_dataset.suffix.lower() != ".jsonl":
        raise ConfigurationError("evaluation_dataset must reference a .jsonl file")
    results_dir = _require_portable_relative_path(root["results_dir"], name="results_dir")

    provider = _require_text(root["provider"], name="provider")
    if provider not in {"fake", "openai"}:
        raise ConfigurationError("provider must be 'fake' or 'openai'")
    embedding_model = _require_text(root["embedding_model"], name="embedding_model")
    generation_model = _require_text(root["generation_model"], name="generation_model")
    prompt_version = _require_text(root["prompt_version"], name="prompt_version")

    temperature = _require_number(root["temperature"], name="temperature")
    if not 0.0 <= temperature <= 2.0:
        raise ConfigurationError("temperature must be between 0 and 2")
    seed = _require_int(root["seed"], name="seed")

    evaluation_mode = _require_text(root["evaluation_mode"], name="evaluation_mode")
    if evaluation_mode != "direct_rag":
        raise ConfigurationError("Phase 5 benchmark evaluation_mode must be 'direct_rag'")
    llm_judge_enabled = _require_bool(root["llm_judge_enabled"], name="llm_judge_enabled")
    if llm_judge_enabled:
        raise ConfigurationError(
            "LLM-as-judge is outside the deterministic MVP and must be disabled."
        )

    experiment_values = root["experiments"]
    if not isinstance(experiment_values, list):
        raise ConfigurationError("experiments must be a list")
    if len(experiment_values) != expected_experiment_count:
        raise ConfigurationError(
            f"Benchmark configuration must define exactly {expected_experiment_count} "
            f"experiments; found {len(experiment_values)}."
        )

    experiments: list[ExperimentConfig] = []
    names: set[str] = set()
    for index, raw_experiment in enumerate(experiment_values, start=1):
        context = f"experiment {index}"
        experiment_mapping = _require_mapping(raw_experiment, context=context)
        _require_exact_fields(
            experiment_mapping,
            expected=_EXPERIMENT_FIELDS,
            context=context,
        )
        name = _require_text(experiment_mapping["name"], name=f"{context}.name")
        if _EXPERIMENT_NAME_PATTERN.fullmatch(name) is None:
            raise ConfigurationError(
                f"{context}.name must use lowercase letters, numbers, hyphens, or underscores"
            )
        if name in names:
            raise ConfigurationError(f"Duplicate experiment name: {name}")
        names.add(name)

        chunk_size = _require_positive_int(
            experiment_mapping["chunk_size"], name=f"{context}.chunk_size"
        )
        overlap = _require_int(experiment_mapping["overlap"], name=f"{context}.overlap")
        if overlap < 0 or overlap >= chunk_size:
            raise ConfigurationError(f"{context}.overlap must satisfy 0 <= overlap < chunk_size")
        top_k = _require_positive_int(experiment_mapping["top_k"], name=f"{context}.top_k")

        experiments.append(
            ExperimentConfig(
                name=name,
                chunk_size=chunk_size,
                overlap=overlap,
                top_k=top_k,
                embedding_model=embedding_model,
                generation_model=generation_model,
                temperature=temperature,
                prompt_version=prompt_version,
                seed=seed,
                evaluation_mode=evaluation_mode,
                llm_judge_enabled=llm_judge_enabled,
            )
        )

    return BenchmarkSuiteConfig(
        schema_version=schema_version,
        corpus_dir=corpus_dir,
        evaluation_dataset=evaluation_dataset,
        results_dir=results_dir,
        provider=provider,
        experiments=tuple(experiments),
    )


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context} must be a mapping with string keys")
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
    raise ConfigurationError(f"{context} has invalid fields ({'; '.join(details)})")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigurationError(f"{name} must be a non-empty, trimmed string")
    return value


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    parsed = _require_int(value, name=name)
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _require_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{name} must be a finite number")
    return parsed


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _require_portable_relative_path(value: object, *, name: str) -> Path:
    text = _require_text(value, name=name)
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    if (
        "\\" in text
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ConfigurationError(f"{name} must be a portable project-relative path")
    return Path(*posix_path.parts)


__all__ = [
    "BENCHMARK_CONFIG_SCHEMA_VERSION",
    "REQUIRED_EXPERIMENT_COUNT",
    "BenchmarkSuiteConfig",
    "load_benchmark_config",
]
