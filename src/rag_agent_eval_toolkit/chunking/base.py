"""Protocol for interchangeable deterministic document chunkers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_agent_eval_toolkit.models import DocumentChunk, SourceDocument


class Chunker(Protocol):
    """Interface implemented by document chunking strategies."""

    @property
    def chunk_size(self) -> int:
        """Return the maximum number of normalized characters per chunk."""
        ...

    @property
    def overlap(self) -> int:
        """Return the repeated normalized-character count between chunks."""
        ...

    def chunk(self, document: SourceDocument) -> list[DocumentChunk]:
        """Split one normalized document into stable chunks."""
        ...

    def chunk_many(self, documents: Sequence[SourceDocument]) -> list[DocumentChunk]:
        """Split multiple normalized documents while retaining input order."""
        ...
