"""Load supported local documents into normalized domain objects."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Any

from rag_agent_eval_toolkit.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from rag_agent_eval_toolkit.models import SourceDocument

SUPPORTED_EXTENSIONS = frozenset({".md", ".pdf", ".txt"})
"""File extensions accepted by the ingestion pipeline."""

NORMALISATION_METHOD = "unicode-nfc-newlines-trailing-whitespace-v1"
"""Versioned normalization method recorded with documents and indexes."""

_FILE_TYPES = {
    ".md": "markdown",
    ".pdf": "pdf",
    ".txt": "text",
}


def normalise_text(text: str) -> str:
    """Return deterministic Unicode text with portable newlines.

    Normalization removes an initial UTF-8 byte-order mark, converts Unicode to
    NFC, converts all newline forms to ``\n``, removes horizontal whitespace at
    line ends, and removes blank space around the complete document.
    """
    without_bom = text.removeprefix("\ufeff")
    nfc_text = unicodedata.normalize("NFC", without_bom)
    portable_newlines = nfc_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (line.rstrip(" \t") for line in portable_newlines.split("\n"))
    return "\n".join(lines).strip()


def _content_checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _source_id(relative_path: str, content_sha256: str) -> str:
    value = f"{relative_path}:{content_sha256}".encode()
    return hashlib.sha256(value).hexdigest()[:16]


def _portable_relative_path(path: Path, root_dir: Path | None) -> str:
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise DocumentLoadError(f"Cannot resolve document path '{path}': {exc}") from exc

    if root_dir is None:
        return resolved_path.name

    try:
        resolved_root = root_dir.resolve(strict=True)
    except OSError as exc:
        raise DocumentLoadError(f"Cannot resolve corpus directory '{root_dir}': {exc}") from exc

    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise DocumentLoadError(
            f"Document '{path}' is outside the corpus directory '{root_dir}'."
        ) from exc
    return relative_path.as_posix()


def _read_utf8_text(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        raw_text = path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Document '{path.name}' is not valid UTF-8 at byte {exc.start}."
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(f"Cannot read document '{path.name}': {exc}") from exc
    return normalise_text(raw_text), {"encoding": "utf-8"}


def _read_pdf(path: Path) -> tuple[str, int, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - installation error
        raise DocumentLoadError("PDF support requires the project dependency 'pypdf'.") from exc

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentLoadError(f"Cannot load encrypted PDF document '{path.name}'.")

        page_texts: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted_text = page.extract_text() or ""
            except Exception as exc:
                raise DocumentLoadError(
                    f"Cannot extract page {page_number} from PDF '{path.name}': {exc}"
                ) from exc
            normalized_page = normalise_text(extracted_text)
            if normalized_page:
                page_texts.append((page_number, normalized_page))
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Cannot parse PDF document '{path.name}': {exc}") from exc

    combined_parts: list[str] = []
    page_spans: list[dict[str, int]] = []
    current_offset = 0
    for page_number, page_text in page_texts:
        if combined_parts:
            combined_parts.append("\n\n")
            current_offset += 2
        character_start = current_offset
        combined_parts.append(page_text)
        current_offset += len(page_text)
        page_spans.append(
            {
                "page_number": page_number,
                "character_start": character_start,
                "character_end": current_offset,
            }
        )

    return (
        "".join(combined_parts),
        len(reader.pages),
        {"page_spans": page_spans},
    )


def load_document(path: Path, *, root_dir: Path | None = None) -> SourceDocument:
    """Load one supported file and preserve portable provenance metadata.

    Args:
        path: Markdown, UTF-8 text, or PDF file to load.
        root_dir: Optional corpus root used to calculate a portable relative path.

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
        DocumentLoadError: If the path is invalid, unreadable, or unparsable.
        EmptyDocumentError: If normalization or extraction yields no content.
    """
    document_path = Path(path)
    extension = document_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported document type '{extension or '<none>'}' for "
            f"'{document_path.name}'. Supported extensions: {supported}."
        )
    if not document_path.is_file():
        raise DocumentLoadError(f"Document path is not a readable file: '{document_path}'.")

    relative_path = _portable_relative_path(document_path, root_dir)
    page_count: int | None = None
    if extension == ".pdf":
        text, page_count, loader_metadata = _read_pdf(document_path)
    else:
        text, loader_metadata = _read_utf8_text(document_path)

    if not text:
        raise EmptyDocumentError(
            f"Document '{relative_path}' contains no text after normalization."
        )

    checksum = _content_checksum(text)
    metadata = {
        "normalisation_method": NORMALISATION_METHOD,
        **loader_metadata,
    }
    return SourceDocument(
        source_id=_source_id(relative_path, checksum),
        source_name=document_path.name,
        relative_path=relative_path,
        file_type=_FILE_TYPES[extension],
        text=text,
        content_sha256=checksum,
        page_count=page_count,
        metadata=metadata,
    )
