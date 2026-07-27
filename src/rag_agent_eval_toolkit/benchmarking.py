"""Concrete local orchestration for deterministic and API-backed benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from rag_agent_eval_toolkit.application import calculate_corpus_checksum
from rag_agent_eval_toolkit.chunking import chunk_documents
from rag_agent_eval_toolkit.config import load_settings
from rag_agent_eval_toolkit.embeddings import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from rag_agent_eval_toolkit.evaluation import (
    EvaluationDataset,
    load_evaluation_dataset,
    validate_mvp_dataset,
)
from rag_agent_eval_toolkit.exceptions import ConfigurationError
from rag_agent_eval_toolkit.experiments import (
    BenchmarkSuiteConfig,
    DirectRagExecutor,
    load_benchmark_config,
    run_benchmark,
)
from rag_agent_eval_toolkit.generation import (
    DeterministicFakeGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.ingestion import NORMALISATION_METHOD, load_documents
from rag_agent_eval_toolkit.models import (
    ExperimentConfig,
    ExperimentResult,
    RagAnswer,
    SourceDocument,
)
from rag_agent_eval_toolkit.rag import DirectRagPipeline
from rag_agent_eval_toolkit.reporting import BenchmarkOutputPaths, write_benchmark_outputs
from rag_agent_eval_toolkit.retrieval import Retriever

ProviderName = Literal["fake", "openai"]

FAKE_EMBEDDING_DIMENSIONS = 4096
FAKE_GENERATION_MODEL = "fake-generation-v1"


@dataclass(frozen=True, slots=True)
class BenchmarkExecution:
    """A completed benchmark and the durable files produced from it."""

    suite: BenchmarkSuiteConfig
    dataset: EvaluationDataset
    results: tuple[ExperimentResult, ...]
    outputs: BenchmarkOutputPaths
    corpus_checksum: str


@dataclass(frozen=True, slots=True)
class _PreparedBenchmark:
    suite: BenchmarkSuiteConfig
    dataset: EvaluationDataset
    documents: tuple[SourceDocument, ...]
    corpus_checksum: str
    project_root: Path
    api_key: str | None

    @property
    def available_sources(self) -> set[str]:
        return {document.source_name for document in self.documents}

    def executor_factory(self, config: ExperimentConfig) -> DirectRagExecutor:
        chunks = chunk_documents(
            self.documents,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )
        embedding_provider: EmbeddingProvider
        generation_provider: GenerationProvider
        if self.suite.provider == "fake":
            embedding_provider = DeterministicFakeEmbeddingProvider(
                dimensions=FAKE_EMBEDDING_DIMENSIONS,
                seed=config.seed,
            )
            if embedding_provider.model_name != config.embedding_model:
                raise ConfigurationError(
                    "Fake benchmark embedding model metadata does not match the "
                    "deterministic provider implementation."
                )
            generation_provider = DeterministicFakeGenerationProvider(
                model_name=config.generation_model,
            )
        else:
            if self.api_key is None:
                raise ConfigurationError(
                    "OPENAI_API_KEY is required for an OpenAI benchmark; "
                    "use --provider fake for an offline run."
                )
            embedding_provider = OpenAIEmbeddingProvider(
                config.embedding_model,
                api_key=self.api_key,
            )
            generation_provider = OpenAIGenerationProvider(
                config.generation_model,
                api_key=self.api_key,
            )

        store = NumpyVectorStore.build(
            chunks,
            embedding_provider,
            corpus_checksum=self.corpus_checksum,
            normalisation_method=NORMALISATION_METHOD,
        )
        retriever = Retriever(
            store,
            embedding_provider,
            default_top_k=config.top_k,
        )
        pipeline = DirectRagPipeline(
            retriever,
            generation_provider,
            default_top_k=config.top_k,
            temperature=config.temperature,
            prompt_version=config.prompt_version,
        )

        def execute(
            question: str,
            experiment: ExperimentConfig,
        ) -> RagAnswer:
            return pipeline.ask(question, top_k=experiment.top_k)

        return execute


def execute_benchmark(
    config_path: Path,
    *,
    provider: ProviderName | None = None,
    output_dir: Path | None = None,
) -> BenchmarkExecution:
    """Run the validated suite and write JSON, CSV, and Markdown outputs."""

    prepared = _prepare_benchmark(config_path, provider=provider)
    results = run_benchmark(
        prepared.suite,
        prepared.dataset,
        prepared.executor_factory,
        corpus_checksum=prepared.corpus_checksum,
        available_sources=prepared.available_sources,
        repo_dir=prepared.project_root,
        environment={"provider": prepared.suite.provider},
    )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else prepared.project_root / prepared.suite.results_dir
    )
    outputs = write_benchmark_outputs(results, destination)
    return BenchmarkExecution(
        suite=prepared.suite,
        dataset=prepared.dataset,
        results=results,
        outputs=outputs,
        corpus_checksum=prepared.corpus_checksum,
    )


def execute_evaluation(
    config_path: Path,
    *,
    experiment_name: str | None = None,
    provider: ProviderName | None = None,
) -> ExperimentResult:
    """Evaluate one named configuration without writing benchmark artifacts."""

    prepared = _prepare_benchmark(config_path, provider=provider)
    selected = _select_experiment(prepared.suite, experiment_name)
    single_suite = replace(prepared.suite, experiments=(selected,))
    (result,) = run_benchmark(
        single_suite,
        prepared.dataset,
        prepared.executor_factory,
        corpus_checksum=prepared.corpus_checksum,
        available_sources=prepared.available_sources,
        repo_dir=prepared.project_root,
        environment={"provider": prepared.suite.provider},
    )
    return result


def _prepare_benchmark(
    config_path: Path,
    *,
    provider: ProviderName | None,
) -> _PreparedBenchmark:
    resolved_config = Path(config_path).resolve()
    suite = _suite_for_provider(load_benchmark_config(resolved_config), provider)
    project_root = _find_project_root(resolved_config)
    documents = tuple(load_documents(project_root / suite.corpus_dir))
    dataset = load_evaluation_dataset(project_root / suite.evaluation_dataset)
    validate_mvp_dataset(dataset)

    api_key: str | None = None
    if suite.provider == "openai":
        settings = load_settings(overrides={"provider": "openai"})
        if settings.openai_api_key is not None:
            api_key = settings.openai_api_key.get_secret_value()

    return _PreparedBenchmark(
        suite=suite,
        dataset=dataset,
        documents=documents,
        corpus_checksum=calculate_corpus_checksum(documents),
        project_root=project_root,
        api_key=api_key,
    )


def _suite_for_provider(
    suite: BenchmarkSuiteConfig,
    provider: ProviderName | None,
) -> BenchmarkSuiteConfig:
    selected_provider = suite.provider if provider is None else provider
    if selected_provider == "fake":
        experiments = tuple(
            replace(
                config,
                embedding_model=(f"fake-hash-v3-d{FAKE_EMBEDDING_DIMENSIONS}-s{config.seed}"),
                generation_model=FAKE_GENERATION_MODEL,
            )
            for config in suite.experiments
        )
    elif selected_provider == "openai":
        if suite.provider == "openai":
            experiments = suite.experiments
        else:
            settings = load_settings(overrides={"provider": "openai"})
            experiments = tuple(
                replace(
                    config,
                    embedding_model=settings.embedding_model,
                    generation_model=settings.generation_model,
                )
                for config in suite.experiments
            )
    else:  # Defensive for direct Python callers outside the typed CLI.
        raise ConfigurationError("Benchmark provider must be 'fake' or 'openai'.")
    return replace(
        suite,
        provider=selected_provider,
        experiments=experiments,
    )


def _select_experiment(
    suite: BenchmarkSuiteConfig,
    experiment_name: str | None,
) -> ExperimentConfig:
    if experiment_name is None:
        return suite.experiments[0]
    for experiment in suite.experiments:
        if experiment.name == experiment_name:
            return experiment
    available = ", ".join(experiment.name for experiment in suite.experiments)
    raise ConfigurationError(
        f"Unknown experiment {experiment_name!r}; available configurations: {available}."
    )


def _find_project_root(config_path: Path) -> Path:
    for directory in (config_path.parent, *config_path.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise ConfigurationError(
        f"Could not locate pyproject.toml above benchmark config: {config_path}"
    )


__all__ = [
    "FAKE_EMBEDDING_DIMENSIONS",
    "FAKE_GENERATION_MODEL",
    "BenchmarkExecution",
    "ProviderName",
    "execute_benchmark",
    "execute_evaluation",
]
