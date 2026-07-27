"""Offline Streamlit UI smoke tests with injected page dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from rag_agent_eval_toolkit.application import build_and_save_index
from rag_agent_eval_toolkit.config import Settings
from rag_agent_eval_toolkit.ui import (
    BenchmarkSummary,
    LocalAnswerBackend,
    ProjectStatus,
    UiAnswer,
    UiDependencies,
    UiEvidence,
    UiRuntimeError,
    inspect_project_status,
    load_benchmark_summary,
    main,
)


class _Column:
    def __init__(self, streamlit: _FakeStreamlit) -> None:
        self._streamlit = streamlit

    def metric(self, label: str, value: object) -> None:
        self._streamlit._record("metric", label, value)


class _FakeStreamlit:
    def __init__(
        self,
        *,
        selections: Mapping[str, object] | None = None,
        question: str = "",
        click_ask: bool = False,
    ) -> None:
        self.selections = dict(selections or {})
        self.question = question
        self.click_ask = click_ask
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def rendered_text(self) -> str:
        return "\n".join(repr(call) for call in self.calls)

    def set_page_config(self, **kwargs: object) -> None:
        self._record("set_page_config", **kwargs)

    def title(self, value: object) -> None:
        self._record("title", value)

    def caption(self, value: object) -> None:
        self._record("caption", value)

    def warning(self, value: object) -> None:
        self._record("warning", value)

    def info(self, value: object) -> None:
        self._record("info", value)

    def error(self, value: object) -> None:
        self._record("error", value)

    def subheader(self, value: object) -> None:
        self._record("subheader", value)

    def markdown(self, value: object) -> None:
        self._record("markdown", value)

    def write(self, value: object) -> None:
        self._record("write", value)

    def columns(self, count: int) -> list[_Column]:
        self._record("columns", count)
        return [_Column(self) for _ in range(count)]

    def selectbox(
        self,
        label: str,
        *,
        options: tuple[object, ...],
        index: int,
        format_func: object,
    ) -> object:
        self._record(
            "selectbox",
            label,
            options,
            index,
            format_func,
        )
        return self.selections.get(label, options[index])

    def radio(
        self,
        label: str,
        *,
        options: tuple[object, ...],
        horizontal: bool,
    ) -> object:
        self._record("radio", label, options, horizontal)
        return self.selections.get(label, options[0])

    def slider(
        self,
        label: str,
        *,
        min_value: int,
        max_value: int,
        value: int,
        step: int,
    ) -> int:
        self._record("slider", label, min_value, max_value, value, step)
        return int(self.selections.get(label, value))

    def text_input(self, label: str, *, placeholder: str) -> str:
        self._record("text_input", label, placeholder)
        return self.question

    def button(self, label: str, *, type: str, disabled: bool) -> bool:
        self._record("button", label, type, disabled)
        return self.click_ask and not disabled

    def dataframe(self, data: object, **kwargs: object) -> None:
        self._record("dataframe", data, **kwargs)

    def bar_chart(self, data: object, **kwargs: object) -> None:
        self._record("bar_chart", data, **kwargs)


class _StubBackend:
    def __init__(self, answer: UiAnswer) -> None:
        self.answer_value = answer
        self.calls: list[tuple[str, str, str, int]] = []

    def answer(
        self,
        question: str,
        *,
        mode: str,
        provider: str,
        top_k: int,
    ) -> UiAnswer:
        self.calls.append((question, mode, provider, top_k))
        return self.answer_value


def _settings(tmp_path: Path, *, provider: str = "fake", api_key: str | None = None) -> Settings:
    return Settings.model_validate(
        {
            "provider": provider,
            "data_dir": tmp_path / "private" / "corpus",
            "index_dir": tmp_path / "private" / "index",
            "results_dir": tmp_path / "private" / "results",
            "openai_api_key": api_key,
            "top_k": 3,
        }
    )


def _ready_status() -> ProjectStatus:
    return ProjectStatus(
        corpus_available=True,
        corpus_document_count=8,
        duplicate_group_count=0,
        index_available=True,
        index_document_count=8,
        index_chunk_count=24,
        index_provider="fake",
        index_model="fake-hash-v1-d256-s42",
        chunk_size=600,
        overlap=100,
    )


def _sample_answer() -> UiAnswer:
    return UiAnswer(
        answer="The fictional warranty covers the battery pack [warranty.md#chunk-001].",
        citations=("[warranty.md#chunk-001]",),
        evidence=(
            UiEvidence(
                rank=1,
                score=0.8123,
                source_name="warranty.md",
                chunk_id="chunk-001",
                page_number=None,
                citation_label="[warranty.md#chunk-001]",
                text="The fictional warranty covers the battery pack.",
            ),
        ),
        model_name="fake-generation-v1",
        total_latency_ms=2.5,
    )


def test_root_app_imports_without_running_streamlit() -> None:
    """The root entry point imports without constructing runtime dependencies."""

    module = importlib.import_module("app")

    assert callable(module.main)


def test_main_page_runs_with_injected_offline_dependencies(tmp_path: Path) -> None:
    """A mocked reviewer interaction renders cited output and benchmark content."""

    settings = _settings(tmp_path)
    backend = _StubBackend(_sample_answer())
    summary = BenchmarkSummary(
        rows=(
            {
                "configuration": "A",
                "chunk_size": 300,
                "overlap": 50,
                "top_k": 3,
                "hit_rate_at_k": 1.0,
                "mrr": 0.9,
                "source_recall_at_k": 0.8,
                "required_fact_coverage": 0.75,
                "citation_validity": 1.0,
                "citation_precision": 1.0,
                "unanswerable_accuracy": 1.0,
                "median_retrieval_latency_ms": 1.2,
                "median_total_latency_ms": 2.5,
                "successful_question_count": 20,
                "failed_question_count": 0,
            },
        )
    )
    dependencies = UiDependencies(
        settings_loader=lambda: settings,
        status_loader=lambda _: _ready_status(),
        backend_factory=lambda _: backend,
        benchmark_loader=lambda _: summary,
    )
    fake_streamlit = _FakeStreamlit(
        question="What is covered?",
        click_ask=True,
    )

    main(streamlit_module=fake_streamlit, dependencies=dependencies)

    assert backend.calls == [("What is covered?", "direct", "fake", 3)]
    rendered = fake_streamlit.rendered_text()
    assert "fictional EV support corpus" in rendered
    assert "warranty.md#chunk-001" in rendered
    assert "Retrieved evidence" in rendered
    assert "bar_chart" in rendered


def test_missing_openai_key_is_helpful_and_prevents_request(tmp_path: Path) -> None:
    """Selecting OpenAI without a key explains recovery and never calls a backend."""

    settings = _settings(tmp_path, provider="openai")
    backend = _StubBackend(_sample_answer())
    dependencies = UiDependencies(
        settings_loader=lambda: settings,
        status_loader=lambda _: _ready_status(),
        backend_factory=lambda _: backend,
        benchmark_loader=lambda _: BenchmarkSummary(rows=()),
    )
    fake_streamlit = _FakeStreamlit(
        selections={"Provider": "openai"},
        question="What is covered?",
        click_ask=True,
    )

    main(streamlit_module=fake_streamlit, dependencies=dependencies)

    assert backend.calls == []
    rendered = fake_streamlit.rendered_text()
    assert "OPENAI_API_KEY" in rendered
    assert "never be displayed" in rendered


def test_page_does_not_render_absolute_configured_paths(tmp_path: Path) -> None:
    """Private local locations remain behind status counts and safe guidance."""

    settings = _settings(tmp_path)
    dependencies = UiDependencies(
        settings_loader=lambda: settings,
        status_loader=lambda _: ProjectStatus(
            corpus_available=False,
            corpus_document_count=0,
            duplicate_group_count=0,
            index_available=False,
            index_document_count=0,
            index_chunk_count=0,
        ),
        backend_factory=lambda _: pytest.fail("disabled Ask button must not build backend"),
        benchmark_loader=lambda _: BenchmarkSummary(rows=()),
    )
    fake_streamlit = _FakeStreamlit()

    main(streamlit_module=fake_streamlit, dependencies=dependencies)

    rendered = fake_streamlit.rendered_text()
    assert str(tmp_path) not in rendered
    assert str(settings.data_dir) not in rendered
    assert str(settings.index_dir) not in rendered
    assert str(settings.results_dir) not in rendered


def test_local_backend_rejects_index_for_changed_corpus(tmp_path: Path) -> None:
    """Answering cannot use an index built from an earlier corpus checksum."""

    settings = _settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    source_path = settings.data_dir / "charging.md"
    source_path.write_text(
        "Inspect the charging connector before use.",
        encoding="utf-8",
    )
    build_and_save_index(settings)
    source_path.write_text(
        "Inspect the charging connector and cable before use.",
        encoding="utf-8",
    )

    with pytest.raises(
        UiRuntimeError,
        match=r"does not match the current corpus.*rag-eval ingest",
    ):
        LocalAnswerBackend(settings).answer(
            "What should I inspect?",
            mode="direct",
            provider="fake",
            top_k=1,
        )


def test_stale_index_is_unavailable_with_actionable_page_guidance(tmp_path: Path) -> None:
    """The status probe disables Ask and tells the reviewer how to rebuild."""

    settings = _settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    source_path = settings.data_dir / "charging.md"
    source_path.write_text(
        "Inspect the charging connector before use.",
        encoding="utf-8",
    )
    build_and_save_index(settings)
    source_path.write_text(
        "Inspect the charging connector and cable before use.",
        encoding="utf-8",
    )

    status = inspect_project_status(settings)
    assert status.corpus_available is True
    assert status.index_available is False

    dependencies = UiDependencies(
        settings_loader=lambda: settings,
        status_loader=lambda _: status,
        backend_factory=lambda _: pytest.fail("stale index must keep Ask disabled"),
        benchmark_loader=lambda _: BenchmarkSummary(rows=()),
    )
    fake_streamlit = _FakeStreamlit(
        question="What should I inspect?",
        click_ask=True,
    )

    main(streamlit_module=fake_streamlit, dependencies=dependencies)

    rendered = fake_streamlit.rendered_text()
    assert "No valid local index matches the current corpus" in rendered
    assert "rag-eval ingest" in rendered


def test_load_benchmark_summary_parses_generated_schema(tmp_path: Path) -> None:
    """The UI reads the Phase 5 summary schema without requiring pandas."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    summary_path = results_dir / "summary.csv"
    summary_path.write_text(
        ",".join(
            (
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
        )
        + "\nA,300,50,3,1,0.75,0.8,0.7,1,1,1,2.5,5.5,20,0\n",
        encoding="utf-8",
    )

    summary = load_benchmark_summary(results_dir)

    assert summary.rows[0]["configuration"] == "A"
    assert summary.rows[0]["chunk_size"] == 300
    assert summary.rows[0]["mrr"] == 0.75


def test_load_benchmark_summary_rejects_path_like_configuration(tmp_path: Path) -> None:
    """Generated result labels cannot smuggle a local path into the page."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "summary.csv").write_text(
        ",".join(
            (
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
        )
        + "\nC:\\private\\run,300,50,3,1,1,1,1,1,1,1,1,1,20,0\n",
        encoding="utf-8",
    )

    with pytest.raises(UiRuntimeError, match="configuration name"):
        load_benchmark_summary(results_dir)
