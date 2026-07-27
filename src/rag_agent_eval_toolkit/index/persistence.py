"""JSON and NumPy persistence for the local vector index."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from rag_agent_eval_toolkit.exceptions import IndexCompatibilityError
from rag_agent_eval_toolkit.index.numpy_store import (
    INDEX_SCHEMA_VERSION,
    NumpyVectorStore,
)
from rag_agent_eval_toolkit.models import DocumentChunk, IndexManifest


def save_index(
    index: NumpyVectorStore,
    directory: str | Path,
    *,
    manifest_file: str = "manifest.json",
) -> Path:
    """Persist an index atomically enough for a small local corpus.

    The manifest is replaced last, so a loader never observes a new manifest
    before both data files have been written successfully.
    """

    root = Path(directory)
    manifest_path = _safe_child_path(root, manifest_file)
    vector_path = _safe_child_path(root, index.manifest.vector_file)
    metadata_path = _safe_child_path(root, index.manifest.metadata_file)
    _validate_distinct_paths(manifest_path, vector_path, metadata_path)

    try:
        root.mkdir(parents=True, exist_ok=True)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        metadata_payload = [asdict(chunk) for chunk in index.chunks]
        metadata_json = json.dumps(
            metadata_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        manifest_json = json.dumps(
            asdict(index.manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise IndexCompatibilityError(
            "Index metadata could not be serialised for persistence."
        ) from exc

    temporary_paths: list[Path] = []
    try:
        vector_temp = _temporary_sibling(vector_path)
        temporary_paths.append(vector_temp)
        with vector_temp.open("wb") as handle:
            np.save(handle, index.vectors, allow_pickle=False)

        metadata_temp = _temporary_sibling(metadata_path)
        temporary_paths.append(metadata_temp)
        metadata_temp.write_text(metadata_json + "\n", encoding="utf-8")

        manifest_temp = _temporary_sibling(manifest_path)
        temporary_paths.append(manifest_temp)
        manifest_temp.write_text(manifest_json + "\n", encoding="utf-8")

        vector_temp.replace(vector_path)
        metadata_temp.replace(metadata_path)
        manifest_temp.replace(manifest_path)
    except OSError as exc:
        raise IndexCompatibilityError(
            "Index files could not be written to the target directory."
        ) from exc
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                # A failed cleanup of our private temporary file must not hide
                # the original persistence result.
                pass

    return manifest_path


def load_index(
    directory: str | Path,
    *,
    manifest_file: str = "manifest.json",
    expected_corpus_checksum: str | None = None,
    strict: bool = True,
) -> NumpyVectorStore:
    """Load an index and reject incompatible schemas, dimensions, or metadata."""

    root = Path(directory)
    manifest_path = _safe_child_path(root, manifest_file)
    manifest_payload = _read_json_object(manifest_path, label="manifest")

    manifest = _manifest_from_payload(manifest_payload)

    if manifest.schema_version != INDEX_SCHEMA_VERSION:
        raise IndexCompatibilityError(
            "Unsupported index schema version "
            f"{manifest.schema_version}; expected {INDEX_SCHEMA_VERSION}."
        )

    if (
        strict
        and expected_corpus_checksum is not None
        and manifest.corpus_checksum != expected_corpus_checksum
    ):
        raise IndexCompatibilityError("Index corpus checksum does not match the expected corpus.")

    vector_path = _safe_child_path(root, manifest.vector_file)
    metadata_path = _safe_child_path(root, manifest.metadata_file)
    _validate_distinct_paths(manifest_path, vector_path, metadata_path)
    if not vector_path.is_file():
        raise IndexCompatibilityError("Index vector file is missing.")
    if not metadata_path.is_file():
        raise IndexCompatibilityError("Index metadata file is missing.")

    try:
        vectors = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexCompatibilityError("Index vector file is unreadable or invalid.") from exc

    metadata_payload = _read_json_array(metadata_path, label="chunk metadata")
    chunks: list[DocumentChunk] = []
    for position, record in enumerate(metadata_payload):
        if not isinstance(record, Mapping):
            raise IndexCompatibilityError(f"Chunk metadata record {position} must be an object.")
        try:
            chunks.append(DocumentChunk(**dict(record)))
        except (TypeError, ValueError) as exc:
            raise IndexCompatibilityError(f"Chunk metadata record {position} is invalid.") from exc

    return NumpyVectorStore(
        vectors=vectors,
        chunks=chunks,
        manifest=manifest,
    )


def _safe_child_path(root: Path, relative_name: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise IndexCompatibilityError("Index file name must not be empty.")

    relative_path = Path(relative_name)
    if relative_path.is_absolute():
        raise IndexCompatibilityError("Index file names must be relative paths.")

    resolved_root = root.resolve()
    resolved_child = (root / relative_path).resolve()
    try:
        resolved_child.relative_to(resolved_root)
    except ValueError as exc:
        raise IndexCompatibilityError(
            "Index file path must remain inside the index directory."
        ) from exc
    if resolved_child == resolved_root:
        raise IndexCompatibilityError("Index file path must identify a file.")
    return resolved_child


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = _read_json(path, label=label)
    if not isinstance(payload, dict):
        raise IndexCompatibilityError(f"Index {label} must be a JSON object.")
    return payload


def _read_json_array(path: Path, *, label: str) -> list[object]:
    payload = _read_json(path, label=label)
    if not isinstance(payload, list):
        raise IndexCompatibilityError(f"Index {label} must be a JSON array.")
    return payload


def _read_json(path: Path, *, label: str) -> object:
    if not path.is_file():
        raise IndexCompatibilityError(f"Index {label} file is missing.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise IndexCompatibilityError(f"Index {label} file is not valid UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise IndexCompatibilityError(f"Index {label} file does not contain valid JSON.") from exc
    except OSError as exc:
        raise IndexCompatibilityError(f"Index {label} file could not be read.") from exc


def _manifest_from_payload(payload: Mapping[str, object]) -> IndexManifest:
    return IndexManifest(
        schema_version=_required_int(payload, "schema_version"),
        created_at_utc=_required_str(payload, "created_at_utc"),
        corpus_checksum=_required_str(payload, "corpus_checksum"),
        document_count=_required_int(payload, "document_count"),
        chunk_count=_required_int(payload, "chunk_count"),
        embedding_provider=_required_str(payload, "embedding_provider"),
        embedding_model=_required_str(payload, "embedding_model"),
        embedding_dimensions=_required_int(payload, "embedding_dimensions"),
        chunk_size=_required_int(payload, "chunk_size"),
        overlap=_required_int(payload, "overlap"),
        normalisation_method=_required_str(payload, "normalisation_method"),
        vector_file=_required_str(payload, "vector_file"),
        metadata_file=_required_str(payload, "metadata_file"),
    )


def _required_int(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndexCompatibilityError(f"Index manifest field '{field_name}' must be an integer.")
    return value


def _required_str(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise IndexCompatibilityError(
            f"Index manifest field '{field_name}' must be a non-empty string."
        )
    return value


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _validate_distinct_paths(*paths: Path) -> None:
    if len(paths) != len(set(paths)):
        raise IndexCompatibilityError(
            "Manifest, vector, and metadata files must use distinct paths."
        )
