"""Fixed-window chunking over normalized document characters.

The module name is retained from the architecture specification, but the MVP
implementation intentionally uses transparent fixed windows rather than a
framework-specific recursive text splitter.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_agent_eval_toolkit.exceptions import InvalidChunkConfigError
from rag_agent_eval_toolkit.models import DocumentChunk, SourceDocument


def _validate_chunk_config(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise InvalidChunkConfigError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise InvalidChunkConfigError("overlap must not be negative.")
    if overlap >= chunk_size:
        raise InvalidChunkConfigError("overlap must be smaller than chunk_size.")


def _stable_chunk_id(
    *,
    source_id: str,
    chunk_index: int,
    chunk_size: int,
    overlap: int,
    chunk_text: str,
) -> str:
    raw_value = (f"{source_id}:{chunk_index}:{chunk_size}:{overlap}:{chunk_text}").encode()
    return hashlib.sha256(raw_value).hexdigest()[:20]


def _page_context(
    page_spans: object, character_start: int, character_end: int
) -> tuple[int | None, list[int]]:
    if not isinstance(page_spans, list):
        return None, []

    page_numbers: list[int] = []
    for span in page_spans:
        if not isinstance(span, Mapping):
            continue
        page_number = span.get("page_number")
        page_start = span.get("character_start")
        page_end = span.get("character_end")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or not isinstance(page_start, int)
            or isinstance(page_start, bool)
            or not isinstance(page_end, int)
            or isinstance(page_end, bool)
        ):
            continue
        if page_start < character_end and page_end > character_start:
            page_numbers.append(page_number)
    return (page_numbers[0] if page_numbers else None), page_numbers


def chunk_document(
    document: SourceDocument, *, chunk_size: int, overlap: int
) -> list[DocumentChunk]:
    """Split a source into deterministic fixed normalized-character windows.

    Character offsets are half-open ranges into ``document.text``. A chunk may
    span PDF pages; in that case ``page_number`` is the first overlapping page
    and ``metadata["page_numbers"]`` retains every overlapping page.
    """
    _validate_chunk_config(chunk_size, overlap)
    if not document.text:
        return []

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for chunk_index, character_start in enumerate(range(0, len(document.text), step)):
        character_end = min(character_start + chunk_size, len(document.text))
        chunk_text = document.text[character_start:character_end]
        if not chunk_text:
            continue

        page_number, page_numbers = _page_context(
            document.metadata.get("page_spans"),
            character_start,
            character_end,
        )
        metadata: dict[str, Any] = copy.deepcopy(document.metadata)
        metadata["relative_path"] = document.relative_path
        if page_numbers:
            metadata["page_numbers"] = page_numbers

        chunks.append(
            DocumentChunk(
                chunk_id=_stable_chunk_id(
                    source_id=document.source_id,
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunk_text=chunk_text,
                ),
                source_id=document.source_id,
                source_name=document.source_name,
                chunk_index=chunk_index,
                text=chunk_text,
                character_start=character_start,
                character_end=character_end,
                page_number=page_number,
                token_estimate=max(1, math.ceil(len(chunk_text) / 4)),
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=metadata,
            )
        )
        if character_end == len(document.text):
            break
    return chunks


def chunk_documents(
    documents: Sequence[SourceDocument], *, chunk_size: int, overlap: int
) -> list[DocumentChunk]:
    """Chunk several documents deterministically while retaining input order."""
    _validate_chunk_config(chunk_size, overlap)
    return [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    ]


@dataclass(frozen=True, slots=True)
class CharacterChunker:
    """Configured fixed-window chunker measured in normalized characters."""

    chunk_size: int
    overlap: int

    def __post_init__(self) -> None:
        """Reject invalid window and overlap values at construction time."""
        _validate_chunk_config(self.chunk_size, self.overlap)

    def chunk(self, document: SourceDocument) -> list[DocumentChunk]:
        """Split one source document using this chunking configuration."""
        return chunk_document(
            document,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

    def chunk_many(self, documents: Sequence[SourceDocument]) -> list[DocumentChunk]:
        """Split multiple documents using this chunking configuration."""
        return chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
