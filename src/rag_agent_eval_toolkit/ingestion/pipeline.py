"""Deterministic directory ingestion and duplicate detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_agent_eval_toolkit.exceptions import DocumentLoadError
from rag_agent_eval_toolkit.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from rag_agent_eval_toolkit.models import SourceDocument


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Documents loaded from a corpus and duplicate groups by checksum."""

    documents: tuple[SourceDocument, ...]
    duplicate_source_ids_by_checksum: dict[str, tuple[str, ...]]

    @property
    def document_count(self) -> int:
        """Return the number of successfully loaded source documents."""
        return len(self.documents)


def find_duplicate_documents(
    documents: Sequence[SourceDocument],
) -> dict[str, tuple[str, ...]]:
    """Return checksum groups containing more than one source document."""
    sources_by_checksum: dict[str, list[str]] = {}
    for document in documents:
        sources_by_checksum.setdefault(document.content_sha256, []).append(document.source_id)
    return {
        checksum: tuple(source_ids)
        for checksum, source_ids in sources_by_checksum.items()
        if len(source_ids) > 1
    }


def load_documents(directory: Path) -> list[SourceDocument]:
    """Recursively load supported files in deterministic relative-path order.

    Unsupported files in a mixed corpus directory are ignored. Calling
    :func:`load_document` directly reports an unsupported requested file.
    """
    corpus_dir = Path(directory)
    if not corpus_dir.exists():
        raise DocumentLoadError(f"Corpus directory does not exist: '{corpus_dir}'.")
    if not corpus_dir.is_dir():
        raise DocumentLoadError(f"Corpus path is not a directory: '{corpus_dir}'.")

    supported_paths = [
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    supported_paths.sort(key=lambda path: path.relative_to(corpus_dir).as_posix().casefold())
    return [load_document(path, root_dir=corpus_dir) for path in supported_paths]


def ingest_directory(directory: Path) -> IngestionResult:
    """Load a corpus directory and report duplicate normalized documents."""
    documents = load_documents(directory)
    return IngestionResult(
        documents=tuple(documents),
        duplicate_source_ids_by_checksum=find_duplicate_documents(documents),
    )
