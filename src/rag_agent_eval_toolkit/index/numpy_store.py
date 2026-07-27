"""Transparent cosine-similarity search over a local NumPy matrix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rag_agent_eval_toolkit.embeddings.base import EmbeddingProvider
from rag_agent_eval_toolkit.exceptions import (
    IndexCompatibilityError,
    RetrievalError,
)
from rag_agent_eval_toolkit.models import DocumentChunk, IndexManifest

INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One scored chunk returned by the low-level vector store."""

    chunk: DocumentChunk
    score: float


class NumpyVectorStore:
    """In-memory NumPy matrix with deterministic cosine-similarity ranking."""

    def __init__(
        self,
        vectors: ArrayLike,
        chunks: Sequence[DocumentChunk],
        manifest: IndexManifest,
    ) -> None:
        """Validate and retain an index matrix, chunk records, and manifest."""

        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise IndexCompatibilityError(
                "Index vectors must be a numeric two-dimensional matrix."
            ) from exc

        chunk_records = tuple(chunks)
        _validate_index_state(matrix, chunk_records, manifest)

        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        matrix.setflags(write=False)
        vector_norms = np.linalg.norm(matrix, axis=1).astype(np.float32, copy=False)
        vector_norms.setflags(write=False)

        self._vectors = matrix
        self._chunks = chunk_records
        self._manifest = manifest
        self._vector_norms = vector_norms

    @classmethod
    def build(
        cls,
        chunks: Sequence[DocumentChunk],
        provider: EmbeddingProvider,
        *,
        corpus_checksum: str,
        normalisation_method: str,
        created_at_utc: str | None = None,
        vector_file: str = "vectors.npy",
        metadata_file: str = "chunks.json",
    ) -> NumpyVectorStore:
        """Embed chunks and build a schema-versioned local vector store."""

        chunk_records = tuple(chunks)
        if not chunk_records:
            raise IndexCompatibilityError(
                "Cannot build an embedding index without document chunks."
            )
        if not isinstance(corpus_checksum, str) or not corpus_checksum.strip():
            raise IndexCompatibilityError("Corpus checksum must not be empty.")
        if not isinstance(normalisation_method, str) or not normalisation_method.strip():
            raise IndexCompatibilityError("Normalisation method must not be empty.")

        chunk_sizes = {chunk.chunk_size for chunk in chunk_records}
        overlaps = {chunk.overlap for chunk in chunk_records}
        if len(chunk_sizes) != 1 or len(overlaps) != 1:
            raise IndexCompatibilityError(
                "All chunks in one index must use the same chunk size and overlap."
            )

        vectors = provider.embed_documents([chunk.text for chunk in chunk_records])
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise IndexCompatibilityError(
                "Embedding provider must return a two-dimensional NumPy matrix."
            )

        provider_name_value = getattr(
            provider,
            "provider_name",
            provider.__class__.__name__,
        )
        provider_name = str(provider_name_value)
        embedding_dimensions = int(vectors.shape[1])
        manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            created_at_utc=created_at_utc or _utc_now(),
            corpus_checksum=corpus_checksum,
            document_count=len({chunk.source_id for chunk in chunk_records}),
            chunk_count=len(chunk_records),
            embedding_provider=provider_name,
            embedding_model=provider.model_name,
            embedding_dimensions=embedding_dimensions,
            chunk_size=next(iter(chunk_sizes)),
            overlap=next(iter(overlaps)),
            normalisation_method=normalisation_method,
            vector_file=vector_file,
            metadata_file=metadata_file,
        )
        return cls(vectors=vectors, chunks=chunk_records, manifest=manifest)

    @property
    def manifest(self) -> IndexManifest:
        """Return the immutable index compatibility manifest."""

        return self._manifest

    @property
    def vectors(self) -> NDArray[np.float32]:
        """Return a defensive copy of the vector matrix."""

        return self._vectors.copy()

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """Return chunk metadata in vector-row order."""

        return self._chunks

    @property
    def dimension(self) -> int:
        """Return the required query-vector dimension."""

        return int(self._vectors.shape[1])

    def __len__(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    def search(
        self,
        query_vector: ArrayLike,
        *,
        top_k: int,
    ) -> list[VectorSearchHit]:
        """Rank chunks by cosine similarity with stable chunk-ID tie breaking."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer.")
        if not self._chunks:
            return []

        try:
            query = np.asarray(query_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RetrievalError("Query embedding must contain numeric values.") from exc
        if query.ndim != 1:
            raise RetrievalError("Query embedding must be one-dimensional.")
        if int(query.shape[0]) != self.dimension:
            raise IndexCompatibilityError(
                "Query embedding dimension "
                f"{query.shape[0]} does not match index dimension {self.dimension}."
            )
        if not np.all(np.isfinite(query)):
            raise RetrievalError("Query embedding contains non-finite values.")

        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            raise RetrievalError("Query embedding must have a non-zero norm.")

        dot_products = self._vectors @ query
        denominators = self._vector_norms * query_norm
        scores = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denominators > 0,
        )
        scores = np.clip(scores, -1.0, 1.0)

        ordered_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (
                -float(scores[index]),
                self._chunks[index].chunk_id,
                index,
            ),
        )
        result_count = min(top_k, len(ordered_indices))
        return [
            VectorSearchHit(
                chunk=self._chunks[index],
                score=float(scores[index]),
            )
            for index in ordered_indices[:result_count]
        ]

    def save(
        self,
        directory: str | Path,
        *,
        manifest_file: str = "manifest.json",
    ) -> Path:
        """Persist vectors, chunk metadata, and manifest to a directory."""

        from rag_agent_eval_toolkit.index.persistence import save_index

        return save_index(self, directory, manifest_file=manifest_file)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        manifest_file: str = "manifest.json",
        expected_corpus_checksum: str | None = None,
        strict: bool = True,
    ) -> NumpyVectorStore:
        """Load and validate a persisted index."""

        from rag_agent_eval_toolkit.index.persistence import load_index

        return load_index(
            directory,
            manifest_file=manifest_file,
            expected_corpus_checksum=expected_corpus_checksum,
            strict=strict,
        )


NumpyVectorIndex = NumpyVectorStore


def _validate_index_state(
    matrix: NDArray[np.float32],
    chunks: tuple[DocumentChunk, ...],
    manifest: IndexManifest,
) -> None:
    if manifest.schema_version != INDEX_SCHEMA_VERSION:
        raise IndexCompatibilityError(
            "Unsupported index schema version "
            f"{manifest.schema_version}; expected {INDEX_SCHEMA_VERSION}."
        )
    if matrix.ndim != 2:
        raise IndexCompatibilityError("Index vectors must be a two-dimensional matrix.")
    if matrix.shape[1] <= 0:
        raise IndexCompatibilityError("Index vectors must have at least one embedding dimension.")
    if not np.all(np.isfinite(matrix)):
        raise IndexCompatibilityError("Index vectors contain non-finite values.")
    if matrix.shape[0] != len(chunks):
        raise IndexCompatibilityError(
            "Vector row count does not match the number of chunk metadata records."
        )
    if manifest.chunk_count != len(chunks):
        raise IndexCompatibilityError("Manifest chunk count does not match chunk metadata records.")
    if manifest.embedding_dimensions != int(matrix.shape[1]):
        raise IndexCompatibilityError(
            "Manifest embedding dimensions do not match the vector matrix."
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise IndexCompatibilityError("Chunk metadata contains duplicate chunk IDs.")
    source_count = len({chunk.source_id for chunk in chunks})
    if manifest.document_count != source_count:
        raise IndexCompatibilityError(
            "Manifest document count does not match chunk source metadata."
        )

    for chunk in chunks:
        if chunk.chunk_size != manifest.chunk_size or chunk.overlap != manifest.overlap:
            raise IndexCompatibilityError("Chunk configuration does not match the index manifest.")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
