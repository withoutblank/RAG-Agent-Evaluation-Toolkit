"""Offline Phase 2 integration test from chunks through persisted retrieval."""

from __future__ import annotations

from pathlib import Path

from rag_agent_eval_toolkit.embeddings import FakeEmbeddingProvider
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.models import DocumentChunk
from rag_agent_eval_toolkit.retrieval import Retriever


def _chunk(
    chunk_id: str,
    source_id: str,
    source_name: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_name=source_name,
        chunk_index=0,
        text=text,
        character_start=0,
        character_end=len(text),
        page_number=None,
        token_estimate=max(1, len(text) // 4),
        chunk_size=200,
        overlap=25,
        metadata={},
    )


def test_fake_embedding_index_persistence_and_retrieval(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=512, seed=23)
    chunks = [
        _chunk(
            "charging-001",
            "charging-source",
            "charging_and_battery.md",
            "Inspect the connector and confirm compatibility before fast charging.",
        ),
        _chunk(
            "warranty-001",
            "warranty-source",
            "warranty_policy.md",
            "Warranty coverage excludes damage caused by unauthorised modifications.",
        ),
        _chunk(
            "roadside-001",
            "roadside-source",
            "roadside_assistance.md",
            "Roadside assistance can arrange towing after an eligible breakdown.",
        ),
    ]
    built = NumpyVectorStore.build(
        chunks,
        provider,
        corpus_checksum="tiny-fixture-checksum",
        normalisation_method="unicode-newlines-v1",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    built.save(tmp_path)

    loaded = NumpyVectorStore.load(
        tmp_path,
        expected_corpus_checksum="tiny-fixture-checksum",
    )
    results = Retriever(loaded, provider).retrieve(
        "What should I inspect before fast charging?",
        top_k=2,
    )

    assert results[0].source_name == "charging_and_battery.md"
    assert results[0].citation_label == ("[charging_and_battery.md#charging-001]")
    assert results[0].rank == 1
