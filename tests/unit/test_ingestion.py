"""Tests for supported document loading and deterministic corpus ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_agent_eval_toolkit.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from rag_agent_eval_toolkit.ingestion import (
    find_duplicate_documents,
    ingest_directory,
    load_document,
    load_documents,
    normalise_text,
)


def _write_text_pdf(path: Path, page_texts: list[str]) -> None:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)  # noqa: SLF001
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        stream = DecodedStreamObject()
        escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
    writer.write(path)


def test_normalise_text_handles_unicode_newlines_and_trailing_space() -> None:
    assert normalise_text("\ufeff Cafe\u0301  \r\nSecond\t\r\n\r\n") == "Café\nSecond"


@pytest.mark.parametrize(
    ("filename", "file_type"),
    [("guide.md", "markdown"), ("guide.txt", "text")],
)
def test_loads_utf8_text_formats(tmp_path: Path, filename: str, file_type: str) -> None:
    path = tmp_path / filename
    path.write_bytes(b"\xef\xbb\xbfFirst line  \r\nSecond line\r\n")

    document = load_document(path, root_dir=tmp_path)

    assert document.source_name == filename
    assert document.relative_path == filename
    assert document.file_type == file_type
    assert document.text == "First line\nSecond line"
    assert document.page_count is None
    assert document.metadata["encoding"] == "utf-8"
    assert len(document.content_sha256) == 64
    assert len(document.source_id) == 16


def test_loads_pdf_text_and_page_spans(tmp_path: Path) -> None:
    path = tmp_path / "guide.pdf"
    _write_text_pdf(path, ["First page text.", "Second page text."])

    document = load_document(path, root_dir=tmp_path)

    assert document.file_type == "pdf"
    assert document.page_count == 2
    assert document.text == "First page text.\n\nSecond page text."
    assert document.metadata["page_spans"] == [
        {
            "page_number": 1,
            "character_start": 0,
            "character_end": len("First page text."),
        },
        {
            "page_number": 2,
            "character_start": len("First page text.\n\n"),
            "character_end": len(document.text),
        },
    ]


def test_unsupported_extension_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "guide.csv"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match=r"\.md, \.pdf, \.txt"):
        load_document(path, root_dir=tmp_path)


def test_empty_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text(" \r\n\t", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="no text"):
        load_document(path, root_dir=tmp_path)


def test_invalid_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"valid prefix\xffinvalid")

    with pytest.raises(DocumentLoadError, match="not valid UTF-8"):
        load_document(path, root_dir=tmp_path)


def test_checksum_and_source_id_are_stable_after_normalisation(tmp_path: Path) -> None:
    path = tmp_path / "stable.md"
    path.write_bytes("Café  \r\nLine\r\n".encode())
    first = load_document(path, root_dir=tmp_path)

    path.write_bytes("Cafe\u0301\nLine\n".encode())
    second = load_document(path, root_dir=tmp_path)

    assert first.text == second.text
    assert first.content_sha256 == second.content_sha256
    assert first.source_id == second.source_id


def test_relative_path_is_portable_and_never_absolute(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    path = nested / "guide.md"
    path.write_text("# Guide", encoding="utf-8")

    document = load_document(path, root_dir=tmp_path)

    assert document.relative_path == "nested/guide.md"
    assert str(tmp_path) not in document.relative_path


def test_document_outside_corpus_root_is_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    path = tmp_path / "outside.md"
    path.write_text("# Outside", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="outside the corpus directory"):
        load_document(path, root_dir=corpus)


def test_directory_loading_is_recursive_sorted_and_ignores_other_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "zeta.txt").write_text("Zeta", encoding="utf-8")
    (tmp_path / "nested" / "alpha.md").write_text("Alpha", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.relative_path for document in documents] == [
        "nested/alpha.md",
        "zeta.txt",
    ]


def test_duplicate_documents_are_grouped_by_normalised_checksum(
    tmp_path: Path,
) -> None:
    (tmp_path / "one.md").write_bytes(b"Same text\r\n")
    (tmp_path / "two.txt").write_bytes(b"Same text\n")

    result = ingest_directory(tmp_path)
    duplicates = find_duplicate_documents(result.documents)

    assert result.document_count == 2
    assert result.duplicate_source_ids_by_checksum == duplicates
    assert list(duplicates) == [result.documents[0].content_sha256]
    assert next(iter(duplicates.values())) == tuple(
        document.source_id for document in result.documents
    )


def test_all_eight_fictional_corpus_documents_load() -> None:
    repository_root = Path(__file__).parents[2]
    corpus_dir = repository_root / "sample_data" / "fictional_ev_support"

    documents = load_documents(corpus_dir)

    assert {document.source_name for document in documents} == {
        "charging_and_battery.md",
        "delivery_process.md",
        "fleet_and_business_support.md",
        "privacy_and_customer_data.md",
        "roadside_assistance.md",
        "scheduled_maintenance.md",
        "software_updates.md",
        "warranty_policy.md",
    }
    assert len(documents) == 8
    assert all("Fictional sample data" in document.text for document in documents)
    assert all(len(document.text) > 1_000 for document in documents)
