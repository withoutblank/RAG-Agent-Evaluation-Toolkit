"""Query embedding and typed ranked retrieval results."""

from __future__ import annotations

import logging
from time import perf_counter

from rag_agent_eval_toolkit.embeddings.base import EmbeddingProvider
from rag_agent_eval_toolkit.exceptions import (
    EmbeddingProviderError,
    IndexCompatibilityError,
    RetrievalError,
)
from rag_agent_eval_toolkit.index.numpy_store import NumpyVectorStore
from rag_agent_eval_toolkit.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve cited chunks using the provider that built a vector index."""

    def __init__(
        self,
        index: NumpyVectorStore,
        embedding_provider: EmbeddingProvider,
        *,
        default_top_k: int = 3,
    ) -> None:
        """Validate provider/index compatibility and configure default top-k."""

        _validate_top_k(default_top_k)
        if embedding_provider.model_name != index.manifest.embedding_model:
            raise IndexCompatibilityError(
                "Query embedding model does not match the model recorded in the index manifest."
            )

        provider_dimensions = getattr(
            embedding_provider,
            "embedding_dimensions",
            None,
        )
        if (
            isinstance(provider_dimensions, int)
            and not isinstance(provider_dimensions, bool)
            and provider_dimensions != index.dimension
        ):
            raise IndexCompatibilityError(
                "Query embedding provider dimensions do not match the index."
            )

        self._index = index
        self._embedding_provider = embedding_provider
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Embed a non-empty query and return ranked, citation-ready evidence."""

        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Retrieval query must contain non-whitespace text.")
        requested_top_k = self._default_top_k if top_k is None else top_k
        _validate_top_k(requested_top_k)

        started = perf_counter()
        try:
            query_vector = self._embedding_provider.embed_query(query)
            hits = self._index.search(query_vector, top_k=requested_top_k)
        except (EmbeddingProviderError, IndexCompatibilityError, RetrievalError):
            raise
        except Exception as exc:
            raise RetrievalError("Retrieval failed unexpectedly.") from exc

        results = [
            RetrievalResult(
                query=query,
                rank=rank,
                score=hit.score,
                chunk_id=hit.chunk.chunk_id,
                source_id=hit.chunk.source_id,
                source_name=hit.chunk.source_name,
                text=hit.chunk.text,
                page_number=hit.chunk.page_number,
                citation_label=citation_label_for(hit.chunk),
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "retrieval_completed",
            extra={
                "retrieval_latency_ms": latency_ms,
                "result_count": len(results),
                "top_k": requested_top_k,
            },
        )
        return results


def citation_label_for(chunk: DocumentChunk) -> str:
    """Return the canonical citation label for a source chunk."""

    return f"[{chunk.source_name}#{chunk.chunk_id}]"


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer.")
