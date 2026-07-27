"""Unit tests for cosine search and index compatibility."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rag_agent_eval_toolkit.embeddings import FakeEmbeddingProvider
from rag_agent_eval_toolkit.exceptions import (
    IndexCompatibilityError,
    RetrievalError,
)
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.models import DocumentChunk, IndexManifest


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "source-1",
    source_name: str = "charging.md",
    page_number: int | None = None,
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
        chunk_size=100,
        overlap=10,
        metadata={"category": "test"},
    )


def _manifest(
    *,
    chunk_count: int,
    document_count: int,
    dimensions: int = 2,
    model: str = "manual-test-model",
) -> IndexManifest:
    return IndexManifest(
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00Z",
        corpus_checksum="abc123",
        document_count=document_count,
        chunk_count=chunk_count,
        embedding_provider="test",
        embedding_model=model,
        embedding_dimensions=dimensions,
        chunk_size=100,
        overlap=10,
        normalisation_method="test-v1",
        vector_file="vectors.npy",
        metadata_file="chunks.json",
    )


def test_cosine_ranking_and_top_k_larger_than_corpus() -> None:
    chunks = [
        _chunk("chunk-a", "battery"),
        _chunk(
            "chunk-b",
            "warranty",
            source_id="source-2",
            source_name="warranty.md",
        ),
    ]
    store = NumpyVectorStore(
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunks=chunks,
        manifest=_manifest(chunk_count=2, document_count=2),
    )

    hits = store.search(np.asarray([0.9, 0.1]), top_k=10)

    assert [hit.chunk.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
    assert hits[0].score > hits[1].score


def test_equal_scores_are_tied_by_chunk_id() -> None:
    chunks = [
        _chunk("chunk-b", "second"),
        _chunk("chunk-a", "first"),
    ]
    store = NumpyVectorStore(
        vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        chunks=chunks,
        manifest=_manifest(chunk_count=2, document_count=1),
    )

    hits = store.search(np.asarray([1.0, 0.0]), top_k=2)

    assert [hit.chunk.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
    assert hits[0].score == hits[1].score == 1.0


def test_empty_index_returns_no_hits() -> None:
    store = NumpyVectorStore(
        vectors=np.empty((0, 2), dtype=np.float32),
        chunks=[],
        manifest=_manifest(chunk_count=0, document_count=0),
    )

    assert store.search(np.asarray([1.0, 0.0]), top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_search_rejects_invalid_top_k(top_k: int) -> None:
    store = NumpyVectorStore(
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[_chunk("chunk-a", "battery")],
        manifest=_manifest(chunk_count=1, document_count=1),
    )

    with pytest.raises(RetrievalError, match="top_k"):
        store.search(np.asarray([1.0, 0.0]), top_k=top_k)


def test_search_rejects_dimension_mismatch_clearly() -> None:
    store = NumpyVectorStore(
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[_chunk("chunk-a", "battery")],
        manifest=_manifest(chunk_count=1, document_count=1),
    )

    with pytest.raises(IndexCompatibilityError, match="dimension 3.*dimension 2"):
        store.search(np.asarray([1.0, 0.0, 0.0]), top_k=1)


def test_manifest_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(IndexCompatibilityError, match="dimensions"):
        NumpyVectorStore(
            vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            chunks=[_chunk("chunk-a", "battery")],
            manifest=_manifest(
                chunk_count=1,
                document_count=1,
                dimensions=3,
            ),
        )


def test_fake_provider_build_ranks_shared_terms_first() -> None:
    provider = FakeEmbeddingProvider(dimensions=512, seed=4)
    chunks = [
        _chunk("chunk-a", "battery charging connector safety"),
        _chunk(
            "chunk-b",
            "warranty repair coverage period",
            source_id="source-2",
            source_name="warranty.md",
        ),
        _chunk(
            "chunk-c",
            "software update installation",
            source_id="source-3",
            source_name="software.md",
        ),
    ]
    store = NumpyVectorStore.build(
        chunks,
        provider,
        corpus_checksum="corpus-checksum",
        normalisation_method="unicode-newlines-v1",
        created_at_utc="2026-01-01T00:00:00Z",
    )

    hits = store.search(provider.embed_query("battery charging"), top_k=2)

    assert hits[0].chunk.chunk_id == "chunk-a"
    assert store.manifest.embedding_dimensions == 512
    assert store.manifest.embedding_model == provider.model_name


def test_persisted_index_reloads_with_identical_results(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=64, seed=9)
    chunks = [
        _chunk("chunk-a", "battery charging connector"),
        _chunk(
            "chunk-b",
            "roadside towing assistance",
            source_id="source-2",
            source_name="roadside.md",
            page_number=2,
        ),
    ]
    original = NumpyVectorStore.build(
        chunks,
        provider,
        corpus_checksum="corpus-checksum",
        normalisation_method="unicode-newlines-v1",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    original.save(tmp_path)

    reloaded = NumpyVectorStore.load(
        tmp_path,
        expected_corpus_checksum="corpus-checksum",
    )
    query = provider.embed_query("roadside towing")
    original_hits = original.search(query, top_k=2)
    reloaded_hits = reloaded.search(query, top_k=2)

    assert reloaded.manifest == original.manifest
    assert reloaded.chunks == original.chunks
    np.testing.assert_array_equal(reloaded.vectors, original.vectors)
    assert [(hit.chunk.chunk_id, hit.score) for hit in reloaded_hits] == [
        (hit.chunk.chunk_id, hit.score) for hit in original_hits
    ]


def test_strict_load_rejects_different_corpus_checksum(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=16)
    store = NumpyVectorStore.build(
        [_chunk("chunk-a", "battery")],
        provider,
        corpus_checksum="expected",
        normalisation_method="test-v1",
    )
    store.save(tmp_path)

    with pytest.raises(IndexCompatibilityError, match="corpus checksum"):
        NumpyVectorStore.load(
            tmp_path,
            expected_corpus_checksum="different",
            strict=True,
        )

    loaded = NumpyVectorStore.load(
        tmp_path,
        expected_corpus_checksum="different",
        strict=False,
    )
    assert loaded.manifest.corpus_checksum == "expected"


def test_load_rejects_unsupported_schema_before_use(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=16)
    store = NumpyVectorStore.build(
        [_chunk("chunk-a", "battery")],
        provider,
        corpus_checksum="expected",
        normalisation_method="test-v1",
    )
    manifest_path = store.save(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="schema version"):
        NumpyVectorStore.load(tmp_path)


def test_load_rejects_vector_dimension_mismatch(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=16)
    store = NumpyVectorStore.build(
        [_chunk("chunk-a", "battery")],
        provider,
        corpus_checksum="expected",
        normalisation_method="test-v1",
    )
    store.save(tmp_path)
    np.save(
        tmp_path / "vectors.npy",
        np.zeros((1, 17), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(IndexCompatibilityError, match="dimensions"):
        NumpyVectorStore.load(tmp_path)


def test_load_rejects_missing_chunk_metadata_record(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dimensions=16)
    store = NumpyVectorStore.build(
        [_chunk("chunk-a", "battery")],
        provider,
        corpus_checksum="expected",
        normalisation_method="test-v1",
    )
    store.save(tmp_path)
    (tmp_path / "chunks.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="metadata records"):
        NumpyVectorStore.load(tmp_path)


def test_missing_metadata_records_are_rejected() -> None:
    chunks = [_chunk("chunk-a", "battery")]
    manifest = replace(
        _manifest(chunk_count=1, document_count=1),
        chunk_count=2,
    )

    with pytest.raises(IndexCompatibilityError, match="Manifest chunk count"):
        NumpyVectorStore(
            vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            chunks=chunks,
            manifest=manifest,
        )
