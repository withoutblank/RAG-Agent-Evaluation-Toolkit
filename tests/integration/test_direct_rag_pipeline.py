"""Offline integration test from a local vector index through cited direct RAG."""

from __future__ import annotations

from rag_agent_eval_toolkit.embeddings import FakeEmbeddingProvider
from rag_agent_eval_toolkit.generation import FakeGenerationProvider
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.models import DocumentChunk
from rag_agent_eval_toolkit.rag import DirectRagPipeline
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


def test_fake_retrieval_and_generation_return_a_resolved_cited_answer() -> None:
    embedding_provider = FakeEmbeddingProvider(dimensions=512, seed=31)
    chunks = [
        _chunk(
            "charging-001",
            "charging-source",
            "charging_and_battery.md",
            "Inspect the charging connector and confirm compatibility before fast charging.",
        ),
        _chunk(
            "warranty-001",
            "warranty-source",
            "warranty_policy.md",
            "Warranty coverage excludes damage from unauthorised modifications.",
        ),
    ]
    index = NumpyVectorStore.build(
        chunks,
        embedding_provider,
        corpus_checksum="phase-3-fixture",
        normalisation_method="unicode-newlines-v1",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    pipeline = DirectRagPipeline(
        Retriever(index, embedding_provider),
        FakeGenerationProvider(),
    )

    answer = pipeline.ask(
        "What should I inspect before fast charging?",
        top_k=1,
    )

    assert answer.retrieval_results[0].chunk_id == "charging-001"
    assert answer.answer.endswith("[charging_and_battery.md#charging-001]")
    assert [citation.chunk_id for citation in answer.citations] == ["charging-001"]
    assert answer.citations[0].source_id == "charging-source"
    assert answer.model_name == "fake-generation-v1"
    assert answer.retrieval_latency_ms >= 0.0
    assert answer.generation_latency_ms >= 0.0
    assert answer.total_latency_ms >= (answer.retrieval_latency_ms + answer.generation_latency_ms)
