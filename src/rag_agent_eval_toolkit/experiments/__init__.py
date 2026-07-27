"""Validated benchmark configuration and injectable experiment execution."""

from rag_agent_eval_toolkit.experiments.config import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BenchmarkSuiteConfig,
    load_benchmark_config,
)
from rag_agent_eval_toolkit.experiments.provenance import (
    GitProvenance,
    GitWorktreeState,
    detect_git_commit,
    detect_git_provenance,
)
from rag_agent_eval_toolkit.experiments.results import (
    EXPERIMENT_RESULT_SCHEMA_VERSION,
    experiment_result_payload,
    load_experiment_result,
    reproducible_result_payload,
)
from rag_agent_eval_toolkit.experiments.runner import (
    DirectRagExecutor,
    DirectRagExecutorFactory,
    run_benchmark,
    run_experiment,
)

__all__ = [
    "BENCHMARK_CONFIG_SCHEMA_VERSION",
    "EXPERIMENT_RESULT_SCHEMA_VERSION",
    "BenchmarkSuiteConfig",
    "DirectRagExecutor",
    "DirectRagExecutorFactory",
    "GitProvenance",
    "GitWorktreeState",
    "detect_git_commit",
    "detect_git_provenance",
    "experiment_result_payload",
    "load_benchmark_config",
    "load_experiment_result",
    "reproducible_result_payload",
    "run_benchmark",
    "run_experiment",
]
