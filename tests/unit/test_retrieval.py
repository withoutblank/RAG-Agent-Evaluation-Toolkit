"""Unit tests for typed retrieval results and citation labels."""

from __future__ import annotations

import pytest

from rag_agent_eval_toolkit.embeddings import FakeEmbeddingProvider
from rag_agent_eval_toolkit.exceptions import (
    IndexCompatibilityError,
    RetrievalError,
)
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.models import DocumentChunk
from rag_agent_eval_toolkit.retrieval import Retriever


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str,
    source_name: str,
    page_number: int | None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_name=source_name,
        chunk_index=0,
        text=text,
        character_start=0,
        character_end=len(text),
        page_number=page_number,
        token_estimate=max(1, len(text) // 4),
        chunk_size=120,
        overlap=20,
        metadata={"fixture": True},
    )


def _retriever() -> Retriever:
    provider = FakeEmbeddingProvider(dimensions=256, seed=5)
    chunks = [
        _chunk(
            "aaa111",
            "Inspect the charging connector before fast charging.",
            source_id="charging-source",
            source_name="charging.md",
            page_number=2,
        ),
        _chunk(
            "bbb222",
            "Roadside support includes towing after a breakdown.",
            source_id="roadside-source",
            source_name="roadside.md",
            page_number=None,
        ),
    ]
    index = NumpyVectorStore.build(
        chunks,
        provider,
        corpus_checksum="fixture-corpus",
        normalisation_method="test-v1",
    )
    return Retriever(index, provider, default_top_k=2)


def test_retrieval_returns_ranked_source_metadata_and_citations() -> None:
    retriever = _retriever()

    results = retriever.retrieve("What should I inspect before charging?", top_k=2)

    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score
    assert results[0].chunk_id == "aaa111"
    assert results[0].source_id == "charging-source"
    assert results[0].source_name == "charging.md"
    assert results[0].page_number == 2
    assert results[0].citation_label == "[charging.md#aaa111]"
    assert all(result.query == "What should I inspect before charging?" for result in results)


@pytest.mark.parametrize("top_k", [0, -2, True])
def test_retrieval_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(RetrievalError, match="top_k"):
        _retriever().retrieve("charging", top_k=top_k)


def test_retrieval_rejects_blank_query() -> None:
    with pytest.raises(RetrievalError, match="non-whitespace"):
        _retriever().retrieve("  ")


def test_retriever_rejects_query_model_mismatch() -> None:
    index_provider = FakeEmbeddingProvider(dimensions=32, seed=1)
    query_provider = FakeEmbeddingProvider(dimensions=32, seed=2)
    chunk = _chunk(
        "aaa111",
        "Charging guidance.",
        source_id="charging-source",
        source_name="charging.md",
        page_number=None,
    )
    index = NumpyVectorStore.build(
        [chunk],
        index_provider,
        corpus_checksum="fixture-corpus",
        normalisation_method="test-v1",
    )

    with pytest.raises(IndexCompatibilityError, match="model"):
        Retriever(index, query_provider)
