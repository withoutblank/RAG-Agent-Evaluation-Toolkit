"""Offline integration coverage for application-level component composition."""

from pathlib import Path

import pytest

from rag_agent_eval_toolkit.application import (
    build_and_save_index,
    calculate_corpus_checksum,
    create_agent_runner,
    create_generation_provider,
    load_runtime,
)
from rag_agent_eval_toolkit.config import Settings
from rag_agent_eval_toolkit.exceptions import ConfigurationError
from rag_agent_eval_toolkit.ingestion import ingest_directory


def _settings(tmp_path: Path, corpus_dir: Path) -> Settings:
    return Settings(
        provider="fake",
        data_dir=corpus_dir,
        index_dir=tmp_path / "index",
        results_dir=tmp_path / "results",
        chunk_size=120,
        overlap=20,
        top_k=1,
        seed=7,
    )


def test_build_load_and_ask_offline(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "charging.md").write_text(
        "Before charging, inspect the connector for damage. "
        "Follow the charger's displayed safety instructions.",
        encoding="utf-8",
    )
    settings = _settings(tmp_path, corpus_dir)

    built = build_and_save_index(settings)
    runtime = load_runtime(settings)
    answer = runtime.rag_pipeline.ask(
        "What should I inspect before charging?",
        top_k=1,
    )

    assert built.document_count == 1
    assert built.chunk_count == 1
    assert built.manifest_path.is_file()
    assert len(runtime.index) == 1
    assert answer.citations
    assert answer.citations[0].source_name == "charging.md"
    assert answer.retrieval_results[0].rank == 1


def test_corpus_checksum_is_independent_of_input_order(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "b.md").write_text("Brake inspection guidance.", encoding="utf-8")
    (corpus_dir / "a.md").write_text("Charging inspection guidance.", encoding="utf-8")
    documents = ingest_directory(corpus_dir).documents

    assert calculate_corpus_checksum(documents) == calculate_corpus_checksum(
        tuple(reversed(documents))
    )


def test_build_load_and_run_agent_offline(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "charging.md").write_text(
        "Before charging, inspect the connector for damage.",
        encoding="utf-8",
    )
    settings = _settings(tmp_path, corpus_dir)
    build_and_save_index(settings)
    runtime = load_runtime(settings)

    result = create_agent_runner(settings, runtime).run(
        "What should I inspect before charging?",
        top_k=1,
    )

    assert result.citations
    assert result.citations[0].source_name == "charging.md"
    assert [call.tool_name for call in result.tool_calls] == [
        "search_corpus",
        "get_source_metadata",
    ]


def test_openai_generation_requires_environment_key(tmp_path: Path) -> None:
    settings = Settings(
        provider="openai",
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        results_dir=tmp_path / "results",
        openai_api_key=None,
    )

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create_generation_provider(settings)
