"""Unit tests for the constrained, typed agent tools."""

from __future__ import annotations

import json

import pytest

from rag_agent_eval_toolkit.agent import (
    GET_SOURCE_METADATA_TOOL_NAME,
    SEARCH_CORPUS_TOOL_NAME,
    AgentTools,
    SearchCorpusArguments,
    SearchCorpusResult,
    SourceMetadataArguments,
    SourceMetadataResult,
    get_source_metadata,
    openai_tool_definitions,
    search_corpus,
)
from rag_agent_eval_toolkit.exceptions import (
    CitationValidationError,
    RetrievalError,
)
from rag_agent_eval_toolkit.models import RetrievalResult, SourceDocument


class _StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results


def _source(
    *,
    source_id: str = "charging-source",
    source_name: str = "charging.md",
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_name=source_name,
        relative_path=f"fictional_ev_support/{source_name}",
        file_type=".md",
        text="Full source text must never be returned by the metadata tool.",
        content_sha256="a" * 64,
        page_count=None,
        metadata={
            "absolute_path": "C:/private/customer/file.md",
            "private_note": "must not leave the process",
        },
    )


def _retrieval_result(
    *,
    source_id: str = "charging-source",
    source_name: str = "charging.md",
    chunk_id: str = "charging-001",
    text: str = (
        "Inspect the connector before charging. This deliberately long remainder must be truncated."
    ),
    citation_label: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        query="What should I inspect?",
        rank=1,
        score=0.912345678,
        chunk_id=chunk_id,
        source_id=source_id,
        source_name=source_name,
        text=text,
        page_number=2,
        citation_label=citation_label or f"[{source_name}#{chunk_id}]",
    )


def test_tool_definitions_expose_exactly_two_strict_functions() -> None:
    definitions = openai_tool_definitions()

    assert [definition["name"] for definition in definitions] == [
        SEARCH_CORPUS_TOOL_NAME,
        GET_SOURCE_METADATA_TOOL_NAME,
    ]
    assert all(definition["type"] == "function" for definition in definitions)
    assert all(definition["strict"] is True for definition in definitions)
    assert all(
        definition["parameters"]["additionalProperties"] is False  # type: ignore[index]
        for definition in definitions
    )
    assert set(
        definitions[0]["parameters"]["properties"]  # type: ignore[index]
    ) == {"query", "top_k"}
    assert set(
        definitions[1]["parameters"]["properties"]  # type: ignore[index]
    ) == {"source_id"}


def test_typed_argument_schemas_strip_text_and_reject_invalid_values() -> None:
    search_arguments = SearchCorpusArguments.from_mapping(
        {"query": "  charging safety?  ", "top_k": 2}
    )
    metadata_arguments = SourceMetadataArguments.from_mapping({"source_id": " charging-source "})

    assert search_arguments == SearchCorpusArguments(
        query="charging safety?",
        top_k=2,
    )
    assert metadata_arguments == SourceMetadataArguments(source_id="charging-source")

    with pytest.raises(RetrievalError, match="non-whitespace"):
        SearchCorpusArguments(query=" ", top_k=1)
    with pytest.raises(RetrievalError, match="positive integer"):
        SearchCorpusArguments(query="question", top_k=True)
    with pytest.raises(RetrievalError, match="unexpected"):
        SearchCorpusArguments.from_mapping({"query": "question", "top_k": 1, "command": "whoami"})
    with pytest.raises(RetrievalError, match="missing"):
        SourceMetadataArguments.from_mapping({})


def test_search_tool_returns_typed_concise_citation_ready_output() -> None:
    retriever = _StubRetriever([_retrieval_result()])

    result = search_corpus(
        "  What should I inspect? ",
        1,
        retriever=retriever,
        max_excerpt_chars=40,
    )
    payload = json.loads(result.to_json())

    assert isinstance(result, SearchCorpusResult)
    assert retriever.calls == [("What should I inspect?", 1)]
    assert payload["result_count"] == 1
    assert payload["results"][0]["score"] == 0.912346
    assert payload["results"][0]["citation_label"] == "[charging.md#charging-001]"
    assert len(payload["results"][0]["excerpt"]) <= 40
    assert payload["results"][0]["excerpt"].endswith("...")
    assert result.as_retrieval_results()[0].citation_label == ("[charging.md#charging-001]")


def test_search_tool_caps_results_and_rejects_malformed_retrieval_output() -> None:
    first = _retrieval_result()
    second = _retrieval_result(
        source_id="warranty-source",
        source_name="warranty.md",
        chunk_id="warranty-001",
    )
    retriever = _StubRetriever([first, second])

    result = search_corpus("question", 1, retriever=retriever)

    assert len(result.matches) == 1
    with pytest.raises(CitationValidationError, match="non-canonical"):
        search_corpus(
            "question",
            1,
            retriever=_StubRetriever([_retrieval_result(citation_label="[invented.md#invented]")]),
        )


def test_metadata_tool_returns_allowlisted_fields_without_text_or_private_metadata() -> None:
    source = _source()

    result = get_source_metadata(source.source_id, sources={source.source_id: source})
    serialized = result.to_json()
    payload = json.loads(serialized)

    assert isinstance(result, SourceMetadataResult)
    assert payload == {
        "source_id": "charging-source",
        "source_name": "charging.md",
        "relative_path": "fictional_ev_support/charging.md",
        "file_type": ".md",
        "content_sha256": "a" * 64,
        "page_count": None,
    }
    assert source.text not in serialized
    assert "private_note" not in serialized
    assert "C:/private" not in serialized


def test_agent_tools_dispatches_only_allowlisted_tools() -> None:
    source = _source()
    retriever = _StubRetriever([_retrieval_result()])
    tools = AgentTools(retriever, {source.source_id: source})

    search_result = tools.invoke(
        SEARCH_CORPUS_TOOL_NAME,
        {"query": "question", "top_k": 1},
    )
    metadata_result = tools.invoke(
        GET_SOURCE_METADATA_TOOL_NAME,
        {"source_id": source.source_id},
    )

    assert isinstance(search_result, SearchCorpusResult)
    assert isinstance(metadata_result, SourceMetadataResult)
    assert tools.tool_names == (
        SEARCH_CORPUS_TOOL_NAME,
        GET_SOURCE_METADATA_TOOL_NAME,
    )
    with pytest.raises(RetrievalError, match="Unsupported agent tool"):
        tools.invoke("run_shell", {"command": "whoami"})


def test_tool_dependency_and_lookup_errors_are_actionable_and_sanitised() -> None:
    source = _source()

    with pytest.raises(RetrievalError, match="No source metadata"):
        get_source_metadata("unknown-source", sources={source.source_id: source})
    with pytest.raises(RetrievalError, match="mapping keys"):
        AgentTools(_StubRetriever([]), {"wrong-key": source})

    class _BrokenRetriever:
        def retrieve(
            self,
            query: str,
            *,
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            raise RuntimeError("private backend detail")

    with pytest.raises(RetrievalError, match="unexpectedly") as exc_info:
        search_corpus("question", 1, retriever=_BrokenRetriever())

    assert "private backend detail" not in str(exc_info.value)


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_excerpt_limit_validation(invalid_limit: object) -> None:
    with pytest.raises(RetrievalError, match="max_excerpt_chars"):
        AgentTools(
            _StubRetriever([]),
            {},
            max_excerpt_chars=invalid_limit,  # type: ignore[arg-type]
        )
