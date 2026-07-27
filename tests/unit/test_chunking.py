"""Tests for deterministic normalized-character chunking."""

from __future__ import annotations

import hashlib

import pytest

from rag_agent_eval_toolkit.chunking import (
    CharacterChunker,
    chunk_document,
    chunk_documents,
)
from rag_agent_eval_toolkit.exceptions import InvalidChunkConfigError
from rag_agent_eval_toolkit.models import SourceDocument


def _document(
    text: str,
    *,
    source_id: str = "source-1",
    metadata: dict[str, object] | None = None,
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_name="guide.md",
        relative_path="guide.md",
        file_type="markdown",
        text=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        page_count=None,
        metadata={} if metadata is None else metadata,
    )


def test_short_document_produces_one_complete_chunk() -> None:
    chunks = chunk_document(_document("short text"), chunk_size=20, overlap=5)

    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].character_start == 0
    assert chunks[0].character_end == 10
    assert chunks[0].token_estimate == 3


def test_exact_boundary_does_not_create_an_empty_chunk() -> None:
    chunks = chunk_document(_document("abcdefgh"), chunk_size=4, overlap=0)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh"]
    assert all(chunk.text for chunk in chunks)


def test_overlap_is_measured_in_normalized_characters() -> None:
    chunks = chunk_document(_document("abcdefghij"), chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert [(chunk.character_start, chunk.character_end) for chunk in chunks] == [(0, 6), (4, 10)]


def test_chunking_is_deterministic_with_repeated_headings() -> None:
    document = _document("# Safety\nDetails\n# Safety\nOther details")
    chunker = CharacterChunker(chunk_size=18, overlap=4)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_id_uses_the_specified_hash_input() -> None:
    document = _document("abcdefghij")
    chunk = chunk_document(document, chunk_size=6, overlap=2)[0]
    expected_input = f"{document.source_id}:0:6:2:abcdef".encode()

    assert chunk.chunk_id == hashlib.sha256(expected_input).hexdigest()[:20]


def test_configuration_changes_stable_chunk_ids() -> None:
    document = _document("abcdefghijklmnopqrstuvwxyz")

    small = chunk_document(document, chunk_size=10, overlap=2)
    large = chunk_document(document, chunk_size=12, overlap=2)

    assert small[0].chunk_id != large[0].chunk_id


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(InvalidChunkConfigError):
        CharacterChunker(chunk_size=chunk_size, overlap=overlap)


def test_page_metadata_is_propagated_for_cross_page_chunk() -> None:
    text = "page one\n\npage two"
    document = _document(
        text,
        metadata={
            "page_spans": [
                {"page_number": 1, "character_start": 0, "character_end": 8},
                {"page_number": 2, "character_start": 10, "character_end": 18},
            ]
        },
    )

    chunk = chunk_document(document, chunk_size=30, overlap=0)[0]

    assert chunk.page_number == 1
    assert chunk.metadata["page_numbers"] == [1, 2]
    assert chunk.metadata["relative_path"] == "guide.md"


def test_chunk_metadata_is_copied_not_shared() -> None:
    document = _document(
        "abcdefgh",
        metadata={"nested": {"value": "original"}},
    )

    chunks = chunk_document(document, chunk_size=4, overlap=0)
    nested = chunks[0].metadata["nested"]
    assert isinstance(nested, dict)
    nested["value"] = "changed"

    assert document.metadata["nested"] == {"value": "original"}
    assert chunks[1].metadata["nested"] == {"value": "original"}


def test_chunk_many_retains_document_and_chunk_order() -> None:
    first = _document("abcdefgh", source_id="source-a")
    second = _document("ijklmnop", source_id="source-b")

    chunks = chunk_documents([first, second], chunk_size=4, overlap=0)

    assert [chunk.source_id for chunk in chunks] == [
        "source-a",
        "source-a",
        "source-b",
        "source-b",
    ]
