"""Compact Streamlit presentation layer for local RAG demonstrations.

The module keeps Streamlit and pipeline construction behind small injected
boundaries. Importing it does not load an index, create an API client, or make a
network request, which keeps the UI smoke-testable with deterministic fakes.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from rag_agent_eval_toolkit.config import Settings
    from rag_agent_eval_toolkit.models import (
        AgentRunResult,
        RagAnswer,
        RetrievalResult,
        SourceDocument,
    )

logger = logging.getLogger(__name__)

ProviderName = Literal["fake", "openai"]
AnswerMode = Literal["direct", "agent"]
BenchmarkValue = str | int | float | None

_PROVIDER_OPTIONS: tuple[ProviderName, ...] = ("fake", "openai")
_MODE_LABELS: Mapping[str, AnswerMode] = {
    "Direct RAG": "direct",
    "Tool-using agent": "agent",
}
_BENCHMARK_COLUMNS = (
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
_INTEGER_COLUMNS = frozenset(
    {
        "chunk_size",
        "overlap",
        "top_k",
        "successful_question_count",
        "failed_question_count",
    }
)
_FLOAT_COLUMNS = frozenset(set(_BENCHMARK_COLUMNS) - _INTEGER_COLUMNS - {"configuration"})


class UiRuntimeError(RuntimeError):
    """An actionable, public-safe error that can be displayed in the UI."""


@dataclass(frozen=True, slots=True)
class ProjectStatus:
    """Safe corpus and index facts rendered without filesystem locations."""

    corpus_available: bool
    corpus_document_count: int
    duplicate_group_count: int
    index_available: bool
    index_document_count: int
    index_chunk_count: int
    index_provider: str | None = None
    index_model: str | None = None
    chunk_size: int | None = None
    overlap: int | None = None


@dataclass(frozen=True, slots=True)
class UiEvidence:
    """One display-safe ranked evidence record."""

    rank: int
    score: float
    source_name: str
    chunk_id: str
    page_number: int | None
    citation_label: str
    text: str


@dataclass(frozen=True, slots=True)
class UiAnswer:
    """Answer data shared by direct-RAG and agent presentation paths."""

    answer: str
    citations: tuple[str, ...]
    evidence: tuple[UiEvidence, ...]
    model_name: str
    total_latency_ms: float
    warnings: tuple[str, ...] = ()
    tool_trace: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Validated benchmark rows suitable for a compact table and chart."""

    rows: tuple[Mapping[str, BenchmarkValue], ...]


class AnswerBackend(Protocol):
    """Minimal answer service consumed by the Streamlit page."""

    def answer(
        self,
        question: str,
        *,
        mode: AnswerMode,
        provider: ProviderName,
        top_k: int,
    ) -> UiAnswer:
        """Return a cited answer and the evidence shown to the reviewer."""

        ...


@dataclass(frozen=True, slots=True)
class UiDependencies:
    """Injected page dependencies used by production and smoke tests."""

    settings_loader: Callable[[], Settings]
    status_loader: Callable[[Settings], ProjectStatus]
    backend_factory: Callable[[Settings], AnswerBackend]
    benchmark_loader: Callable[[Path], BenchmarkSummary]


class _RetrieverLike(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class _AgentRunnerLike(Protocol):
    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> AgentRunResult: ...


class _RecordingRetriever:
    """Record the agent's actual retrieval call for the evidence panel."""

    def __init__(self, delegate: _RetrieverLike) -> None:
        self._delegate = delegate
        self.last_results: list[RetrievalResult] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Delegate retrieval and retain only the latest result list."""

        results = self._delegate.retrieve(query, top_k=top_k)
        self.last_results = list(results)
        return results


class LocalAnswerBackend:
    """Build local providers lazily and answer against a persisted index."""

    def __init__(self, settings: Settings) -> None:
        """Retain validated settings without loading providers or index data."""

        self._settings = settings

    def answer(
        self,
        question: str,
        *,
        mode: AnswerMode,
        provider: ProviderName,
        top_k: int,
    ) -> UiAnswer:
        """Run the selected direct or agent path with no implicit ingestion."""

        if provider not in _PROVIDER_OPTIONS:
            raise UiRuntimeError("Select either the fake or OpenAI provider.")
        if mode not in {"direct", "agent"}:
            raise UiRuntimeError("Select either Direct RAG or tool-using agent mode.")

        index = self._load_index()
        if index.manifest.embedding_provider != provider:
            recorded_provider = (
                index.manifest.embedding_provider
                if index.manifest.embedding_provider in _PROVIDER_OPTIONS
                else "another provider"
            )
            raise UiRuntimeError(
                f"The current index was built with {recorded_provider!r}, not "
                f"{provider!r}. Rebuild the index with the selected provider."
            )

        embedding_provider = self._embedding_provider(
            provider,
            model=index.manifest.embedding_model,
            dimensions=index.manifest.embedding_dimensions,
        )

        from rag_agent_eval_toolkit.retrieval import Retriever

        retriever = Retriever(
            index,
            embedding_provider,
            default_top_k=top_k,
        )
        if mode == "direct":
            return self._run_direct(question, provider, retriever, top_k=top_k)
        return self._run_agent(question, provider, retriever, top_k=top_k)

    def _load_index(self) -> Any:
        from rag_agent_eval_toolkit.exceptions import RagEvalError
        from rag_agent_eval_toolkit.ingestion import ingest_directory

        try:
            documents = ingest_directory(self._settings.data_dir).documents
            return _load_index_for_current_corpus(self._settings, documents)
        except RagEvalError as exc:
            raise UiRuntimeError(
                "The local index is missing, incompatible, or does not match the "
                "current corpus. Run `rag-eval ingest` to rebuild it."
            ) from exc

    def _embedding_provider(
        self,
        provider: ProviderName,
        *,
        model: str,
        dimensions: int,
    ) -> Any:
        if provider == "fake":
            from rag_agent_eval_toolkit.embeddings import (
                DeterministicFakeEmbeddingProvider,
            )

            return DeterministicFakeEmbeddingProvider(
                dimensions=dimensions,
                seed=self._settings.seed,
            )

        api_key = _openai_api_key(self._settings)
        if api_key is None:
            raise UiRuntimeError(
                "OpenAI mode needs OPENAI_API_KEY in the local environment. "
                "The key is never shown by this app."
            )
        from rag_agent_eval_toolkit.embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
        )

    def _run_direct(
        self,
        question: str,
        provider: ProviderName,
        retriever: _RetrieverLike,
        *,
        top_k: int,
    ) -> UiAnswer:
        from rag_agent_eval_toolkit.generation import (
            DeterministicFakeGenerationProvider,
            GenerationProvider,
            OpenAIGenerationProvider,
        )
        from rag_agent_eval_toolkit.rag import DirectRagPipeline

        generation_provider: GenerationProvider
        if provider == "fake":
            generation_provider = DeterministicFakeGenerationProvider(
                model_name=self._settings.generation_model,
            )
        else:
            api_key = _openai_api_key(self._settings)
            if api_key is None:  # Defensive; checked while creating embeddings.
                raise UiRuntimeError("OpenAI mode needs OPENAI_API_KEY.")
            generation_provider = OpenAIGenerationProvider(
                self._settings.generation_model,
                api_key=api_key,
            )

        answer = DirectRagPipeline(
            cast(Any, retriever),
            generation_provider,
            default_top_k=top_k,
            temperature=self._settings.temperature,
            prompt_version=self._settings.prompt_version,
        ).ask(question, top_k=top_k)
        return _ui_answer_from_rag(answer)

    def _run_agent(
        self,
        question: str,
        provider: ProviderName,
        retriever: _RetrieverLike,
        *,
        top_k: int,
    ) -> UiAnswer:
        from rag_agent_eval_toolkit.agent import (
            AgentTools,
            DeterministicAgentRunner,
            OpenAIToolCallingAgentRunner,
        )
        from rag_agent_eval_toolkit.ingestion import load_documents

        try:
            documents = load_documents(self._settings.data_dir)
        except Exception as exc:
            _log_failure("agent_source_metadata", exc)
            raise UiRuntimeError(
                "Source metadata could not be loaded for agent mode. "
                "Check the configured corpus and rebuild the index."
            ) from exc

        recording_retriever = _RecordingRetriever(retriever)
        tools = AgentTools(
            recording_retriever,
            {document.source_id: document for document in documents},
        )
        runner: _AgentRunnerLike
        if provider == "fake":
            runner = DeterministicAgentRunner(tools, default_top_k=top_k)
        else:
            api_key = _openai_api_key(self._settings)
            if api_key is None:  # Defensive; checked while creating embeddings.
                raise UiRuntimeError("OpenAI mode needs OPENAI_API_KEY.")
            runner = OpenAIToolCallingAgentRunner(
                tools,
                self._settings.generation_model,
                api_key=api_key,
                default_top_k=top_k,
            )
        result = runner.run(question, top_k=top_k)
        return _ui_answer_from_agent(result, recording_retriever.last_results)


def default_dependencies() -> UiDependencies:
    """Return lazily imported production dependencies for the page."""

    from rag_agent_eval_toolkit.config import load_settings

    return UiDependencies(
        settings_loader=load_settings,
        status_loader=inspect_project_status,
        backend_factory=LocalAnswerBackend,
        benchmark_loader=load_benchmark_summary,
    )


def inspect_project_status(settings: Settings) -> ProjectStatus:
    """Inspect corpus and persisted-index state without exposing local paths."""

    from rag_agent_eval_toolkit.ingestion import ingest_directory

    corpus_available = False
    corpus_document_count = 0
    duplicate_group_count = 0
    corpus_documents: tuple[SourceDocument, ...] = ()
    try:
        corpus = ingest_directory(settings.data_dir)
        corpus_documents = corpus.documents
        corpus_document_count = corpus.document_count
        duplicate_group_count = len(corpus.duplicate_source_ids_by_checksum)
        corpus_available = corpus_document_count > 0
    except Exception as exc:
        _log_failure("corpus_status", exc)

    index_available = False
    index_document_count = 0
    index_chunk_count = 0
    index_provider: str | None = None
    index_model: str | None = None
    chunk_size: int | None = None
    overlap: int | None = None
    if corpus_documents:
        try:
            index = _load_index_for_current_corpus(settings, corpus_documents)
            manifest = index.manifest
            index_document_count = manifest.document_count
            index_chunk_count = manifest.chunk_count
            index_provider = _safe_identifier(manifest.embedding_provider)
            index_model = _safe_identifier(manifest.embedding_model)
            chunk_size = manifest.chunk_size
            overlap = manifest.overlap
            index_available = True
        except Exception as exc:
            _log_failure("index_status", exc)

    return ProjectStatus(
        corpus_available=corpus_available,
        corpus_document_count=corpus_document_count,
        duplicate_group_count=duplicate_group_count,
        index_available=index_available,
        index_document_count=index_document_count,
        index_chunk_count=index_chunk_count,
        index_provider=index_provider,
        index_model=index_model,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def load_benchmark_summary(results_dir: Path) -> BenchmarkSummary:
    """Load and strictly validate the generated benchmark ``summary.csv``."""

    summary_path = Path(results_dir) / "summary.csv"
    if not summary_path.is_file():
        return BenchmarkSummary(rows=())

    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise UiRuntimeError("Benchmark summary has no header.")
            missing_columns = sorted(set(_BENCHMARK_COLUMNS).difference(reader.fieldnames))
            if missing_columns:
                raise UiRuntimeError("Benchmark summary uses an incompatible schema.")
            rows = tuple(_parse_benchmark_row(row) for row in reader)
    except UiRuntimeError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise UiRuntimeError("Benchmark summary could not be read.") from exc
    return BenchmarkSummary(rows=rows)


def main(
    *,
    streamlit_module: Any | None = None,
    dependencies: UiDependencies | None = None,
) -> None:
    """Render the one-page Streamlit reviewer demonstration."""

    st = streamlit_module or _import_streamlit()
    deps = dependencies or default_dependencies()

    st.set_page_config(
        page_title="RAG Agent Evaluation Toolkit",
        page_icon="🔎",
        layout="wide",
    )
    st.title("RAG Agent Evaluation Toolkit")
    st.caption("Inspect a cited answer and compare reproducible retrieval configurations.")
    st.warning(
        "This demo uses a fictional EV support corpus. The fake provider is fully "
        "offline; the OpenAI provider can make billable API requests when selected."
    )

    try:
        settings = deps.settings_loader()
    except Exception as exc:
        _log_failure("settings", exc)
        st.error(
            "Configuration could not be loaded. Check `.env` and the documented "
            "settings, then restart the app."
        )
        return

    try:
        status = deps.status_loader(settings)
    except Exception as exc:
        _log_failure("status", exc)
        status = ProjectStatus(
            corpus_available=False,
            corpus_document_count=0,
            duplicate_group_count=0,
            index_available=False,
            index_document_count=0,
            index_chunk_count=0,
        )

    _render_status(st, status)

    st.subheader("Ask the knowledge base")
    configured_provider = settings.provider if settings.provider in _PROVIDER_OPTIONS else "fake"
    provider = cast(
        ProviderName,
        st.selectbox(
            "Provider",
            options=_PROVIDER_OPTIONS,
            index=_PROVIDER_OPTIONS.index(configured_provider),
            format_func=lambda value: (
                "Fake (offline)" if value == "fake" else "OpenAI (uses your API key)"
            ),
        ),
    )
    selected_mode_label = st.radio(
        "Answer mode",
        options=tuple(_MODE_LABELS),
        horizontal=True,
    )
    mode = _MODE_LABELS.get(selected_mode_label, "direct")
    top_k = int(
        st.slider(
            "Retrieved chunks (top-k)",
            min_value=1,
            max_value=10,
            value=min(max(settings.top_k, 1), 10),
            step=1,
        )
    )

    api_key_available = _openai_api_key(settings) is not None
    if not api_key_available:
        if provider == "openai":
            st.warning(
                "OpenAI mode is unavailable because `OPENAI_API_KEY` is not set. "
                "Set it in your local environment and restart the app; the key will "
                "never be displayed."
            )
        else:
            st.info(
                "No OpenAI API key is configured. The fake provider remains available "
                "for an offline, deterministic plumbing demonstration."
            )

    question = st.text_input(
        "Question",
        placeholder="What does the fictional battery warranty cover?",
    )
    ask_disabled = (
        not question.strip()
        or not status.index_available
        or (provider == "openai" and not api_key_available)
    )
    if not status.index_available:
        if status.corpus_available:
            st.info(
                "No valid local index matches the current corpus. Run "
                "`rag-eval ingest` to rebuild it before asking."
            )
        else:
            st.info("No valid local index is available. Run `rag-eval ingest` before asking.")

    if st.button("Ask", type="primary", disabled=ask_disabled):
        try:
            backend = deps.backend_factory(settings)
            answer = backend.answer(
                question.strip(),
                mode=mode,
                provider=provider,
                top_k=top_k,
            )
        except UiRuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            _log_failure("answer", exc)
            st.error(
                "The answer could not be generated. Check provider and index status, "
                "then try again."
            )
        else:
            _render_answer(st, answer)

    st.subheader("Benchmark comparison")
    try:
        summary = deps.benchmark_loader(settings.results_dir)
    except Exception as exc:
        _log_failure("benchmark_summary", exc)
        st.warning(
            "The benchmark summary is unavailable or incompatible. Re-run "
            "`rag-eval benchmark --provider fake` to regenerate it."
        )
    else:
        _render_benchmark(st, summary)


def _render_status(st: Any, status: ProjectStatus) -> None:
    st.subheader("Corpus and index status")
    columns = st.columns(3)
    columns[0].metric(
        "Corpus documents",
        status.corpus_document_count if status.corpus_available else "Unavailable",
    )
    columns[1].metric(
        "Indexed chunks",
        status.index_chunk_count if status.index_available else "Unavailable",
    )
    columns[2].metric(
        "Index provider",
        status.index_provider if status.index_available else "Unavailable",
    )
    if status.corpus_available and status.duplicate_group_count:
        st.warning(f"Corpus checksum scan found {status.duplicate_group_count} duplicate group(s).")
    if status.index_available:
        st.caption(
            f"Validated index: {status.index_document_count} documents; "
            f"chunk size {status.chunk_size}; overlap {status.overlap}; "
            f"model {status.index_model or 'unknown'}."
        )


def _render_answer(st: Any, answer: UiAnswer) -> None:
    st.subheader("Cited answer")
    st.markdown(answer.answer)
    citation_text = ", ".join(answer.citations) if answer.citations else "None (abstention)"
    st.caption(
        f"Citations: {citation_text} · Model: {answer.model_name} · "
        f"Total latency: {answer.total_latency_ms:.1f} ms"
    )
    for warning in answer.warnings:
        st.warning(warning)

    st.subheader("Retrieved evidence")
    if not answer.evidence:
        st.info("No evidence was retrieved.")
    for evidence in answer.evidence:
        page = f" · page {evidence.page_number}" if evidence.page_number is not None else ""
        st.markdown(f"**{evidence.rank}. {evidence.source_name}{page}**")
        st.caption(
            f"Score {evidence.score:.4f} · {evidence.citation_label} · chunk {evidence.chunk_id}"
        )
        st.write(evidence.text)

    if answer.tool_trace:
        st.subheader("Agent tool trace")
        for trace in answer.tool_trace:
            st.write(trace)


def _render_benchmark(st: Any, summary: BenchmarkSummary) -> None:
    if not summary.rows:
        st.info(
            "No benchmark summary is available yet. Run "
            "`rag-eval benchmark --config configs/benchmark.yaml --provider fake`."
        )
        return

    display_rows = [_display_benchmark_row(row) for row in summary.rows]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    chart_rows = [
        {
            "Configuration": row["configuration"],
            "Hit Rate@k": row["hit_rate_at_k"],
            "MRR": row["mrr"],
            "Citation validity": row["citation_validity"],
        }
        for row in summary.rows
    ]
    st.bar_chart(
        chart_rows,
        x="Configuration",
        y=["Hit Rate@k", "MRR", "Citation validity"],
        use_container_width=True,
    )
    st.caption(
        "Fake-provider results demonstrate deterministic pipeline behavior, not "
        "general semantic model quality."
    )


def _parse_benchmark_row(row: Mapping[str, str | None]) -> Mapping[str, BenchmarkValue]:
    parsed: dict[str, BenchmarkValue] = {}
    configuration = row.get("configuration")
    if (
        configuration is None
        or not configuration
        or len(configuration) > 80
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in configuration
        )
    ):
        raise UiRuntimeError("Benchmark summary contains an invalid configuration name.")
    parsed["configuration"] = configuration

    for column in _INTEGER_COLUMNS:
        raw_value = row.get(column)
        try:
            int_value = int(raw_value) if raw_value is not None and raw_value != "" else None
        except ValueError as exc:
            raise UiRuntimeError("Benchmark summary contains an invalid integer.") from exc
        if int_value is None or int_value < 0:
            raise UiRuntimeError("Benchmark summary contains an invalid integer.")
        parsed[column] = int_value

    for column in _FLOAT_COLUMNS:
        raw_value = row.get(column)
        if raw_value in {None, ""}:
            parsed[column] = None
            continue
        try:
            float_value = float(cast(str, raw_value))
        except ValueError as exc:
            raise UiRuntimeError("Benchmark summary contains an invalid number.") from exc
        if not (-1_000_000_000.0 < float_value < 1_000_000_000.0):
            raise UiRuntimeError("Benchmark summary contains an out-of-range number.")
        parsed[column] = float_value

    return {column: parsed[column] for column in _BENCHMARK_COLUMNS}


def _display_benchmark_row(
    row: Mapping[str, BenchmarkValue],
) -> Mapping[str, BenchmarkValue]:
    return {
        "Configuration": row["configuration"],
        "Chunk size": row["chunk_size"],
        "Overlap": row["overlap"],
        "Top-k": row["top_k"],
        "Hit Rate@k": row["hit_rate_at_k"],
        "MRR": row["mrr"],
        "Source Recall@k": row["source_recall_at_k"],
        "Fact coverage": row["required_fact_coverage"],
        "Citation validity": row["citation_validity"],
        "Citation precision": row["citation_precision"],
        "Unanswerable accuracy": row["unanswerable_accuracy"],
        "Median retrieval (ms)": row["median_retrieval_latency_ms"],
        "Median total (ms)": row["median_total_latency_ms"],
    }


def _ui_answer_from_rag(answer: RagAnswer) -> UiAnswer:
    return UiAnswer(
        answer=answer.answer,
        citations=tuple(citation.citation_label for citation in answer.citations),
        evidence=_ui_evidence(answer.retrieval_results),
        model_name=_safe_identifier(answer.model_name),
        total_latency_ms=answer.total_latency_ms,
        warnings=tuple(answer.warnings),
    )


def _ui_answer_from_agent(
    answer: AgentRunResult,
    evidence: Sequence[RetrievalResult],
) -> UiAnswer:
    return UiAnswer(
        answer=answer.answer,
        citations=tuple(citation.citation_label for citation in answer.citations),
        evidence=_ui_evidence(evidence),
        model_name=_safe_identifier(answer.model_name),
        total_latency_ms=answer.total_latency_ms,
        warnings=tuple(answer.warnings),
        tool_trace=tuple(f"{call.tool_name}: {call.result_summary}" for call in answer.tool_calls),
    )


def _ui_evidence(results: Sequence[RetrievalResult]) -> tuple[UiEvidence, ...]:
    return tuple(
        UiEvidence(
            rank=result.rank,
            score=result.score,
            source_name=_safe_source_name(result.source_name),
            chunk_id=_safe_identifier(result.chunk_id),
            page_number=result.page_number,
            citation_label=result.citation_label,
            text=result.text,
        )
        for result in results
    )


def _safe_source_name(value: str) -> str:
    portable = value.replace("\\", "/")
    return PurePosixPath(portable).name or "unknown source"


def _safe_identifier(value: str) -> str:
    if len(value) <= 120:
        return value
    return f"{value[:117]}..."


def _load_index_for_current_corpus(
    settings: Settings,
    documents: tuple[SourceDocument, ...],
) -> Any:
    from rag_agent_eval_toolkit.application import calculate_corpus_checksum
    from rag_agent_eval_toolkit.index import load_index

    return load_index(
        settings.index_dir,
        expected_corpus_checksum=calculate_corpus_checksum(documents),
        strict=True,
    )


def _openai_api_key(settings: Settings) -> str | None:
    secret = settings.openai_api_key
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def _import_streamlit() -> Any:
    try:
        import streamlit
    except ImportError as exc:  # pragma: no cover - installation error
        raise UiRuntimeError(
            "Streamlit is not installed. Install the project dependencies first."
        ) from exc
    return streamlit


def _log_failure(stage: str, exc: Exception) -> None:
    logger.error(
        "ui_operation_failed",
        extra={
            "stage": stage,
            "error_type": exc.__class__.__name__,
        },
    )


__all__ = [
    "AnswerBackend",
    "AnswerMode",
    "BenchmarkSummary",
    "LocalAnswerBackend",
    "ProjectStatus",
    "ProviderName",
    "UiAnswer",
    "UiDependencies",
    "UiEvidence",
    "default_dependencies",
    "inspect_project_status",
    "load_benchmark_summary",
    "main",
]
