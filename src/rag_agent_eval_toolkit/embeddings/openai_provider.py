"""Official OpenAI embeddings adapter with injectable client support."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from rag_agent_eval_toolkit.embeddings.base import _validated_texts
from rag_agent_eval_toolkit.exceptions import EmbeddingProviderError


class _EmbeddingsResource(Protocol):
    def create(self, **kwargs: object) -> object: ...


class OpenAIEmbeddingClient(Protocol):
    """Small client surface required from the official OpenAI SDK."""

    embeddings: _EmbeddingsResource


class OpenAIEmbeddingProvider:
    """Embed text through the official OpenAI client.

    A compatible client can be injected for tests. When no client is supplied,
    the official SDK is imported lazily and configured to leave retry policy to
    this adapter.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: OpenAIEmbeddingClient | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 100,
        max_retries: int = 2,
        retry_base_seconds: float = 0.25,
        timeout_seconds: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise an injectable, bounded-retry OpenAI adapter."""

        if not isinstance(model, str) or not model.strip():
            raise EmbeddingProviderError("OpenAI embedding model must not be empty.")
        if dimensions is not None and (
            isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0
        ):
            raise EmbeddingProviderError("Requested embedding dimensions must be greater than 0.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise EmbeddingProviderError("Embedding batch size must be greater than 0.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise EmbeddingProviderError("max_retries must be a non-negative integer.")
        if retry_base_seconds < 0:
            raise EmbeddingProviderError("retry_base_seconds must be non-negative.")
        if timeout_seconds <= 0:
            raise EmbeddingProviderError("timeout_seconds must be greater than 0.")

        self._model = model.strip()
        self._requested_dimensions = dimensions
        self._observed_dimensions: int | None = None
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._sleeper = sleeper
        self._client = client or self._create_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _create_client(
        *,
        api_key: str | None,
        timeout_seconds: float,
    ) -> OpenAIEmbeddingClient:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is runtime setup
            raise EmbeddingProviderError(
                "The 'openai' package is required for OpenAI embeddings."
            ) from exc

        return cast(
            OpenAIEmbeddingClient,
            OpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=timeout_seconds,
            ),
        )

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model name."""

        return self._model

    @property
    def embedding_dimensions(self) -> int | None:
        """Return configured or observed vector dimensions when known."""

        return self._requested_dimensions or self._observed_dimensions

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed document texts in bounded API batches."""

        values = _validated_texts(
            texts,
            allow_empty_sequence=True,
            parameter_name="texts",
        )
        if not values:
            dimensions = self.embedding_dimensions or 0
            return np.empty((0, dimensions), dtype=np.float32)

        batches = [
            self._request_batch(values[start : start + self._batch_size])
            for start in range(0, len(values), self._batch_size)
        ]
        return np.vstack(batches).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query and return a one-dimensional vector."""

        (value,) = _validated_texts(
            (text,),
            allow_empty_sequence=False,
            parameter_name="query",
        )
        return cast(NDArray[np.float32], self._request_batch((value,))[0])

    def _request_batch(self, texts: Sequence[str]) -> NDArray[np.float32]:
        request: dict[str, object] = {
            "input": list(texts),
            "model": self._model,
        }
        if self._requested_dimensions is not None:
            request["dimensions"] = self._requested_dimensions

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.embeddings.create(**request)
                return self._matrix_from_response(response, expected_count=len(texts))
            except EmbeddingProviderError:
                raise
            except Exception as exc:
                should_retry = attempt < self._max_retries and _is_transient_error(exc)
                if not should_retry:
                    raise _sanitised_provider_error(
                        exc,
                        attempts=attempt + 1,
                    ) from exc
                self._sleeper(self._retry_base_seconds * (2**attempt))

        raise EmbeddingProviderError("OpenAI embedding retry loop ended unexpectedly.")

    def _matrix_from_response(
        self,
        response: object,
        *,
        expected_count: int,
    ) -> NDArray[np.float32]:
        data = _read_field(response, "data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise EmbeddingProviderError(
                "OpenAI embedding response did not contain a data sequence."
            )
        if len(data) != expected_count:
            raise EmbeddingProviderError(
                "OpenAI embedding response count did not match the request."
            )

        ordered: list[tuple[int, NDArray[np.float32]]] = []
        for position, item in enumerate(data):
            raw_index = _read_field(item, "index", default=position)
            raw_embedding = _read_field(item, "embedding")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise EmbeddingProviderError(
                    "OpenAI embedding response contained an invalid item index."
                )
            try:
                vector = np.asarray(raw_embedding, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    "OpenAI embedding response contained a non-numeric vector."
                ) from exc
            if vector.ndim != 1 or vector.size == 0:
                raise EmbeddingProviderError(
                    "OpenAI embedding response vectors must be non-empty and 1D."
                )
            if not np.all(np.isfinite(vector)):
                raise EmbeddingProviderError(
                    "OpenAI embedding response contained non-finite values."
                )
            ordered.append((raw_index, vector))

        ordered.sort(key=lambda pair: pair[0])
        if [index for index, _ in ordered] != list(range(expected_count)):
            raise EmbeddingProviderError(
                "OpenAI embedding response indices were incomplete or duplicated."
            )

        dimensions = {int(vector.shape[0]) for _, vector in ordered}
        if len(dimensions) != 1:
            raise EmbeddingProviderError(
                "OpenAI embedding response contained inconsistent dimensions."
            )
        observed_dimensions = dimensions.pop()
        if (
            self._requested_dimensions is not None
            and observed_dimensions != self._requested_dimensions
        ):
            raise EmbeddingProviderError(
                "OpenAI embedding response dimensions did not match the request."
            )
        if (
            self._observed_dimensions is not None
            and observed_dimensions != self._observed_dimensions
        ):
            raise EmbeddingProviderError("OpenAI embedding dimensions changed between requests.")
        self._observed_dimensions = observed_dimensions
        return np.vstack([vector for _, vector in ordered]).astype(
            np.float32,
            copy=False,
        )


_MISSING = object()


def _read_field(
    value: object,
    name: str,
    *,
    default: object = _MISSING,
) -> object:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)

    if default is not _MISSING:
        return default
    raise EmbeddingProviderError(f"OpenAI embedding response was missing the '{name}' field.")


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500):
        return True

    transient_class_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
    return isinstance(exc, (ConnectionError, TimeoutError)) or (
        exc.__class__.__name__ in transient_class_names
    )


def _sanitised_provider_error(
    exc: Exception,
    *,
    attempts: int,
) -> EmbeddingProviderError:
    status_code = getattr(exc, "status_code", None)
    status_suffix = f", HTTP {status_code}" if isinstance(status_code, int) else ""
    return EmbeddingProviderError(
        "OpenAI embedding request failed "
        f"after {attempts} attempt(s) ({exc.__class__.__name__}{status_suffix})."
    )
