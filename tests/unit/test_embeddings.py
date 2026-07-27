"""Unit tests for deterministic and injected embedding providers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rag_agent_eval_toolkit.embeddings import (
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from rag_agent_eval_toolkit.exceptions import EmbeddingProviderError


class _RecordingEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        data = [
            SimpleNamespace(index=index, embedding=self.vectors[index])
            for index in range(len(inputs))
        ]
        return SimpleNamespace(data=data)


class _Client:
    def __init__(self, embeddings: object) -> None:
        self.embeddings = embeddings


def test_fake_embeddings_are_deterministic_and_normalised() -> None:
    provider_one = FakeEmbeddingProvider(dimensions=64, seed=7)
    provider_two = FakeEmbeddingProvider(dimensions=64, seed=7)
    texts = ["battery charging safety", "warranty coverage"]

    first = provider_one.embed_documents(texts)
    second = provider_two.embed_documents(texts)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], provider_one.embed_query(texts[0]))
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2))
    assert first.shape == (2, 64)
    assert first.dtype == np.float32


def test_fake_provider_returns_typed_empty_document_matrix() -> None:
    provider = FakeEmbeddingProvider(dimensions=12)

    result = provider.embed_documents([])

    assert result.shape == (0, 12)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: FakeEmbeddingProvider(dimensions=0), "greater than 0"),
        (
            lambda: FakeEmbeddingProvider().embed_query("  "),
            "non-whitespace",
        ),
        (
            lambda: FakeEmbeddingProvider().embed_documents("one string"),
            "sequence of strings",
        ),
    ],
)
def test_fake_provider_rejects_invalid_input(operation: object, message: str) -> None:
    assert callable(operation)
    with pytest.raises(EmbeddingProviderError, match=message):
        operation()


def test_openai_provider_uses_injected_client_and_preserves_order() -> None:
    endpoint = _RecordingEmbeddings([[1.0, 0.0], [0.0, 1.0]])
    client = _Client(endpoint)
    provider = OpenAIEmbeddingProvider(
        "text-embedding-test",
        client=client,  # type: ignore[arg-type]
        dimensions=2,
        batch_size=2,
    )

    result = provider.embed_documents(["first", "second"])

    np.testing.assert_array_equal(
        result,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert endpoint.calls == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-test",
            "dimensions": 2,
        }
    ]
    assert provider.embedding_dimensions == 2


def test_openai_provider_reorders_indexed_response_items() -> None:
    class _ReversedEmbeddings:
        def create(self, **kwargs: object) -> object:
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                    SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                ]
            )

    provider = OpenAIEmbeddingProvider(
        "text-embedding-test",
        client=_Client(_ReversedEmbeddings()),  # type: ignore[arg-type]
        dimensions=2,
    )

    result = provider.embed_documents(["first", "second"])

    np.testing.assert_array_equal(
        result,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_openai_provider_retries_only_transient_failures() -> None:
    class APITimeoutError(Exception):
        pass

    class _FlakyEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise APITimeoutError("request body must not leak")
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])

    endpoint = _FlakyEmbeddings()
    sleeps: list[float] = []
    provider = OpenAIEmbeddingProvider(
        "text-embedding-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
        dimensions=2,
        max_retries=2,
        retry_base_seconds=0.5,
        sleeper=sleeps.append,
    )

    result = provider.embed_query("safe query")

    np.testing.assert_array_equal(result, np.asarray([1.0, 0.0], dtype=np.float32))
    assert endpoint.calls == 2
    assert sleeps == [0.5]


def test_openai_provider_does_not_retry_invalid_response() -> None:
    endpoint = _RecordingEmbeddings([[1.0, 0.0, 0.0]])
    provider = OpenAIEmbeddingProvider(
        "text-embedding-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
        dimensions=2,
        max_retries=2,
        sleeper=lambda _: pytest.fail("invalid responses must not be retried"),
    )

    with pytest.raises(EmbeddingProviderError, match="dimensions"):
        provider.embed_query("query")

    assert len(endpoint.calls) == 1
