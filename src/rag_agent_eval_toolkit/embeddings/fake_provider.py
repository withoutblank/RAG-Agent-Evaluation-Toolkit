"""Deterministic, local embeddings for tests and offline demonstrations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rag_agent_eval_toolkit.embeddings.base import _validated_texts
from rag_agent_eval_toolkit.exceptions import EmbeddingProviderError

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "do",
        "does",
        "driver",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "vehicle",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)
_TOKEN_ALIASES = {
    "charge": "charge",
    "charged": "charge",
    "charger": "charge",
    "chargers": "charge",
    "charging": "charge",
    "check": "inspect",
    "checked": "inspect",
    "checking": "inspect",
    "checks": "inspect",
    "connect": "connect",
    "connected": "connect",
    "connecting": "connect",
    "connection": "connect",
    "connector": "connect",
    "connectors": "connect",
    "inspect": "inspect",
    "inspected": "inspect",
    "inspecting": "inspect",
    "inspection": "inspect",
    "inspections": "inspect",
}


class DeterministicFakeEmbeddingProvider:
    """Create reproducible hashed bag-of-words embeddings without network access.

    The provider is intended for testing plumbing and reproducibility. It is not
    a substitute for a semantic embedding model and benchmark reports must not
    present its scores as model-quality evidence.
    """

    provider_name = "fake"

    def __init__(self, dimensions: int = 256, *, seed: int = 0) -> None:
        """Initialise the provider with a fixed vector size and hashing seed."""

        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise EmbeddingProviderError("Embedding dimensions must be an integer.")
        if dimensions <= 0:
            raise EmbeddingProviderError("Embedding dimensions must be greater than 0.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EmbeddingProviderError("Embedding seed must be an integer.")

        self._dimensions = dimensions
        self._seed = seed

    @property
    def model_name(self) -> str:
        """Return a stable identifier containing all output-affecting settings."""

        return f"fake-hash-v3-d{self._dimensions}-s{self._seed}"

    @property
    def embedding_dimensions(self) -> int:
        """Return the number of values in each generated embedding."""

        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed documents deterministically as a two-dimensional matrix."""

        values = _validated_texts(
            texts,
            allow_empty_sequence=True,
            parameter_name="texts",
        )
        if not values:
            return np.empty((0, self._dimensions), dtype=np.float32)
        return np.vstack([self._embed_text(text) for text in values]).astype(
            np.float32,
            copy=False,
        )

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query using the same deterministic feature mapping."""

        (value,) = _validated_texts(
            (text,),
            allow_empty_sequence=False,
            parameter_name="query",
        )
        return self._embed_text(value)

    def _embed_text(self, text: str) -> NDArray[np.float32]:
        tokens = sorted(
            {
                _TOKEN_ALIASES.get(token, token)
                for token in _TOKEN_PATTERN.findall(text.casefold())
                if token not in _STOP_WORDS
            }
        )
        if not tokens:
            # Non-whitespace punctuation-only inputs still receive a stable
            # feature rather than becoming an invalid zero vector.
            tokens = [text.casefold().strip()]

        vector = np.zeros(self._dimensions, dtype=np.float32)
        seed_prefix = str(self._seed).encode("ascii") + b":"
        for token in tokens:
            digest = hashlib.sha256(seed_prefix + token.encode("utf-8")).digest()
            feature_index = int.from_bytes(digest[:8], "big") % self._dimensions
            vector[feature_index] += 1.0

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # Defensive: validated text always contributes a feature.
            raise EmbeddingProviderError("Fake embedding unexpectedly had zero norm.")
        vector /= norm
        return vector


FakeEmbeddingProvider = DeterministicFakeEmbeddingProvider
