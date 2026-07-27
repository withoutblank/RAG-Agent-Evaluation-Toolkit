"""Shared embedding provider interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rag_agent_eval_toolkit.exceptions import EmbeddingProviderError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol implemented by document and query embedding providers."""

    @property
    def model_name(self) -> str:
        """Return the stable model identifier recorded in index manifests."""

        ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed document texts into a two-dimensional matrix."""

        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query into a one-dimensional vector."""

        ...


def _validated_texts(
    texts: Sequence[str],
    *,
    allow_empty_sequence: bool,
    parameter_name: str,
) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise EmbeddingProviderError(
            f"{parameter_name} must be a sequence of strings, not one string."
        )

    values = tuple(texts)
    if not values and not allow_empty_sequence:
        raise EmbeddingProviderError(f"{parameter_name} must not be empty.")

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise EmbeddingProviderError(f"{parameter_name}[{index}] must be a string.")
        if not value.strip():
            raise EmbeddingProviderError(
                f"{parameter_name}[{index}] must contain non-whitespace text."
            )
    return values
