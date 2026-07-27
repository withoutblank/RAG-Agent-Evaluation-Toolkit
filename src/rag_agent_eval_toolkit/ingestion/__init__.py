"""Document loading and corpus-ingestion APIs."""

from rag_agent_eval_toolkit.ingestion.loaders import (
    NORMALISATION_METHOD,
    SUPPORTED_EXTENSIONS,
    load_document,
    normalise_text,
)
from rag_agent_eval_toolkit.ingestion.pipeline import (
    IngestionResult,
    find_duplicate_documents,
    ingest_directory,
    load_documents,
)

__all__ = [
    "NORMALISATION_METHOD",
    "SUPPORTED_EXTENSIONS",
    "IngestionResult",
    "find_duplicate_documents",
    "ingest_directory",
    "load_document",
    "load_documents",
    "normalise_text",
]
