"""Application services that compose the explicit RAG components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag_agent_eval_toolkit.agent import (
    AgentTools,
    DeterministicAgentRunner,
    OpenAIToolCallingAgentRunner,
)
from rag_agent_eval_toolkit.chunking import CharacterChunker
from rag_agent_eval_toolkit.config import Settings
from rag_agent_eval_toolkit.embeddings import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from rag_agent_eval_toolkit.exceptions import ConfigurationError, EmptyDocumentError
from rag_agent_eval_toolkit.generation import (
    DeterministicFakeGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from rag_agent_eval_toolkit.index import NumpyVectorStore
from rag_agent_eval_toolkit.ingestion import (
    NORMALISATION_METHOD,
    IngestionResult,
    ingest_directory,
)
from rag_agent_eval_toolkit.models import SourceDocument
from rag_agent_eval_toolkit.rag import DirectRagPipeline
from rag_agent_eval_toolkit.retrieval import Retriever


@dataclass(frozen=True, slots=True)
class BuiltIndex:
    """Result of ingesting, chunking, embedding, and persisting one corpus."""

    manifest_path: Path
    document_count: int
    chunk_count: int
    corpus_checksum: str
    duplicate_group_count: int


@dataclass(frozen=True, slots=True)
class LoadedRuntime:
    """Loaded corpus, index, providers, and direct-RAG pipeline."""

    documents: tuple[SourceDocument, ...]
    index: NumpyVectorStore
    retriever: Retriever
    rag_pipeline: DirectRagPipeline


def calculate_corpus_checksum(documents: tuple[SourceDocument, ...]) -> str:
    """Hash ordered relative paths and normalized content checksums."""

    if not documents:
        raise EmptyDocumentError("The corpus contains no supported non-empty documents.")
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        digest.update(document.relative_path.encode())
        digest.update(b"\0")
        digest.update(document.content_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_and_save_index(settings: Settings) -> BuiltIndex:
    """Build the configured local index and persist its manifest and data."""

    ingestion = ingest_directory(settings.data_dir)
    documents = _require_documents(ingestion)
    chunks = CharacterChunker(settings.chunk_size, settings.overlap).chunk_many(documents)
    if not chunks:
        raise EmptyDocumentError("The corpus produced no searchable chunks.")

    corpus_checksum = calculate_corpus_checksum(documents)
    provider = create_embedding_provider(settings)
    index = NumpyVectorStore.build(
        chunks,
        provider,
        corpus_checksum=corpus_checksum,
        normalisation_method=NORMALISATION_METHOD,
    )
    manifest_path = index.save(settings.index_dir)
    return BuiltIndex(
        manifest_path=manifest_path,
        document_count=len(documents),
        chunk_count=len(chunks),
        corpus_checksum=corpus_checksum,
        duplicate_group_count=len(ingestion.duplicate_source_ids_by_checksum),
    )


def load_runtime(settings: Settings) -> LoadedRuntime:
    """Load and validate the configured corpus/index and compose direct RAG."""

    ingestion = ingest_directory(settings.data_dir)
    documents = _require_documents(ingestion)
    corpus_checksum = calculate_corpus_checksum(documents)
    index = NumpyVectorStore.load(
        settings.index_dir,
        expected_corpus_checksum=corpus_checksum,
        strict=True,
    )
    embedding_provider = create_embedding_provider(
        settings,
        fake_dimensions=index.dimension,
    )
    retriever = Retriever(
        index,
        embedding_provider,
        default_top_k=settings.top_k,
    )
    generation_provider = create_generation_provider(settings)
    rag_pipeline = DirectRagPipeline(
        retriever,
        generation_provider,
        default_top_k=settings.top_k,
        temperature=settings.temperature,
        prompt_version=settings.prompt_version,
    )
    return LoadedRuntime(
        documents=documents,
        index=index,
        retriever=retriever,
        rag_pipeline=rag_pipeline,
    )


def create_embedding_provider(
    settings: Settings,
    *,
    fake_dimensions: int = 4096,
) -> EmbeddingProvider:
    """Create the configured embedding provider without exposing credentials."""

    if settings.provider == "fake":
        return DeterministicFakeEmbeddingProvider(
            dimensions=fake_dimensions,
            seed=settings.seed,
        )
    return OpenAIEmbeddingProvider(
        model=settings.embedding_model,
        api_key=_required_openai_key(settings),
    )


def create_generation_provider(settings: Settings) -> GenerationProvider:
    """Create the configured generation provider without making a request."""

    if settings.provider == "fake":
        return DeterministicFakeGenerationProvider(model_name="fake-generation-v1")
    return OpenAIGenerationProvider(
        model=settings.generation_model,
        api_key=_required_openai_key(settings),
    )


def source_documents_by_id(
    documents: tuple[SourceDocument, ...],
) -> dict[str, SourceDocument]:
    """Index source documents by stable ID for the constrained agent tool."""

    return {document.source_id: document for document in documents}


def create_agent_runner(
    settings: Settings,
    runtime: LoadedRuntime,
) -> DeterministicAgentRunner | OpenAIToolCallingAgentRunner:
    """Compose the constrained two-tool agent for an already loaded runtime."""

    tools = AgentTools(
        runtime.retriever,
        source_documents_by_id(runtime.documents),
    )
    if settings.provider == "fake":
        return DeterministicAgentRunner(
            tools,
            default_top_k=settings.top_k,
        )
    return OpenAIToolCallingAgentRunner(
        tools,
        settings.generation_model,
        api_key=_required_openai_key(settings),
        default_top_k=settings.top_k,
    )


def _required_openai_key(settings: Settings) -> str:
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is required when provider='openai'; "
            "use provider='fake' for offline operation."
        )
    return settings.openai_api_key.get_secret_value()


def _require_documents(ingestion: IngestionResult) -> tuple[SourceDocument, ...]:
    if not ingestion.documents:
        raise EmptyDocumentError("No supported non-empty documents were found in the corpus.")
    return ingestion.documents


__all__ = [
    "BuiltIndex",
    "LoadedRuntime",
    "build_and_save_index",
    "calculate_corpus_checksum",
    "create_agent_runner",
    "create_embedding_provider",
    "create_generation_provider",
    "load_runtime",
    "source_documents_by_id",
]
