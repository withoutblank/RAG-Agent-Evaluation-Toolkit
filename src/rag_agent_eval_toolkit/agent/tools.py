"""Typed, allowlisted tools available to the retrieval agent."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rag_agent_eval_toolkit.exceptions import (
    CitationValidationError,
    RetrievalError,
)
from rag_agent_eval_toolkit.models import (
    RetrievalResult,
    SourceDocument,
)

SEARCH_CORPUS_TOOL_NAME = "search_corpus"
GET_SOURCE_METADATA_TOOL_NAME = "get_source_metadata"
SUPPORTED_TOOL_NAMES = (
    SEARCH_CORPUS_TOOL_NAME,
    GET_SOURCE_METADATA_TOOL_NAME,
)

_WHITESPACE = re.compile(r"\s+")
_MARKDOWN_HEADING_LINE = re.compile(r"(?m)^#{1,6}\s+[^\n]*(?:\n|$)")


class RetrieverLike(Protocol):
    """Minimal retriever surface required by the agent tool boundary."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked evidence for one query."""

        ...


@dataclass(frozen=True, slots=True)
class SearchCorpusArguments:
    """Validated input schema for ``search_corpus``."""

    query: str
    top_k: int

    def __post_init__(self) -> None:
        """Reject empty queries and invalid result limits."""

        if not isinstance(self.query, str) or not self.query.strip():
            raise RetrievalError("search_corpus query must contain non-whitespace text.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise RetrievalError("search_corpus top_k must be a positive integer.")
        object.__setattr__(self, "query", self.query.strip())

    @classmethod
    def from_mapping(cls, arguments: Mapping[str, object]) -> SearchCorpusArguments:
        """Parse a strict JSON-like arguments mapping."""

        _validate_argument_keys(
            arguments,
            expected={"query", "top_k"},
            tool_name=SEARCH_CORPUS_TOOL_NAME,
        )
        return cls(
            query=arguments["query"],  # type: ignore[arg-type]
            top_k=arguments["top_k"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SourceMetadataArguments:
    """Validated input schema for ``get_source_metadata``."""

    source_id: str

    def __post_init__(self) -> None:
        """Reject empty source identifiers."""

        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise RetrievalError("get_source_metadata source_id must contain non-whitespace text.")
        object.__setattr__(self, "source_id", self.source_id.strip())

    @classmethod
    def from_mapping(cls, arguments: Mapping[str, object]) -> SourceMetadataArguments:
        """Parse a strict JSON-like arguments mapping."""

        _validate_argument_keys(
            arguments,
            expected={"source_id"},
            tool_name=GET_SOURCE_METADATA_TOOL_NAME,
        )
        return cls(source_id=arguments["source_id"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SearchCorpusMatch:
    """One concise, citation-ready match returned to an agent."""

    rank: int
    score: float
    chunk_id: str
    source_id: str
    source_name: str
    excerpt: str
    page_number: int | None
    citation_label: str

    def to_payload(self) -> dict[str, object]:
        """Return the allowlisted JSON-compatible representation."""

        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "excerpt": self.excerpt,
            "page_number": self.page_number,
            "citation_label": self.citation_label,
        }


@dataclass(frozen=True, slots=True)
class SearchCorpusResult:
    """Typed output schema for ``search_corpus``."""

    query: str
    matches: tuple[SearchCorpusMatch, ...]

    def to_payload(self) -> dict[str, object]:
        """Return a concise JSON-compatible tool payload."""

        return {
            "query": self.query,
            "result_count": len(self.matches),
            "results": [match.to_payload() for match in self.matches],
        }

    def to_json(self) -> str:
        """Serialize the safe tool payload for a model function result."""

        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def as_retrieval_results(self) -> list[RetrievalResult]:
        """Convert matches to the shared evidence model for citation validation."""

        return [
            RetrievalResult(
                query=self.query,
                rank=match.rank,
                score=match.score,
                chunk_id=match.chunk_id,
                source_id=match.source_id,
                source_name=match.source_name,
                text=match.excerpt,
                page_number=match.page_number,
                citation_label=match.citation_label,
            )
            for match in self.matches
        ]


@dataclass(frozen=True, slots=True)
class SourceMetadataResult:
    """Typed, allowlisted output schema for ``get_source_metadata``.

    Source text and arbitrary metadata values are intentionally excluded. They
    can be large or confidential and are unnecessary for source attribution.
    """

    source_id: str
    source_name: str
    relative_path: str
    file_type: str
    content_sha256: str
    page_count: int | None

    def to_payload(self) -> dict[str, object]:
        """Return the allowlisted JSON-compatible representation."""

        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "relative_path": self.relative_path,
            "file_type": self.file_type,
            "content_sha256": self.content_sha256,
            "page_count": self.page_count,
        }

    def to_json(self) -> str:
        """Serialize the safe tool payload for a model function result."""

        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


AgentToolResult: TypeAlias = SearchCorpusResult | SourceMetadataResult


def search_corpus(
    query: str,
    top_k: int,
    *,
    retriever: RetrieverLike,
    max_excerpt_chars: int = 700,
) -> SearchCorpusResult:
    """Retrieve concise, ranked evidence through an injected retriever."""

    arguments = SearchCorpusArguments(query=query, top_k=top_k)
    _validate_excerpt_limit(max_excerpt_chars)
    try:
        retrieval_results = retriever.retrieve(
            arguments.query,
            top_k=arguments.top_k,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError("search_corpus failed unexpectedly.") from exc

    matches = tuple(
        _match_from_retrieval(result, max_excerpt_chars=max_excerpt_chars)
        for result in retrieval_results[: arguments.top_k]
    )
    return SearchCorpusResult(query=arguments.query, matches=matches)


def get_source_metadata(
    source_id: str,
    *,
    sources: Mapping[str, SourceDocument],
) -> SourceMetadataResult:
    """Return allowlisted metadata for one known source identifier."""

    arguments = SourceMetadataArguments(source_id=source_id)
    source = sources.get(arguments.source_id)
    if source is None:
        raise RetrievalError(f"No source metadata was found for source_id {arguments.source_id!r}.")
    if source.source_id != arguments.source_id:
        raise RetrievalError("Source metadata mapping key does not match the document source_id.")
    return SourceMetadataResult(
        source_id=source.source_id,
        source_name=source.source_name,
        relative_path=source.relative_path,
        file_type=source.file_type,
        content_sha256=source.content_sha256,
        page_count=source.page_count,
    )


class AgentTools:
    """Bind the two supported agent tools to injected local dependencies."""

    def __init__(
        self,
        retriever: RetrieverLike,
        sources: Mapping[str, SourceDocument],
        *,
        max_excerpt_chars: int = 700,
    ) -> None:
        """Validate and retain the retrieval and source metadata dependencies."""

        _validate_excerpt_limit(max_excerpt_chars)
        copied_sources = dict(sources)
        for source_id, source in copied_sources.items():
            if source_id != source.source_id:
                raise RetrievalError(
                    "Source metadata mapping keys must match document source_id values."
                )
        self._retriever = retriever
        self._sources = copied_sources
        self._max_excerpt_chars = max_excerpt_chars

    @property
    def tool_names(self) -> tuple[str, str]:
        """Return the complete allowlist exposed to an agent."""

        return SUPPORTED_TOOL_NAMES

    def search_corpus(self, query: str, top_k: int) -> SearchCorpusResult:
        """Run the typed retrieval tool."""

        return search_corpus(
            query,
            top_k,
            retriever=self._retriever,
            max_excerpt_chars=self._max_excerpt_chars,
        )

    def get_source_metadata(self, source_id: str) -> SourceMetadataResult:
        """Run the typed source metadata tool."""

        return get_source_metadata(source_id, sources=self._sources)

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> AgentToolResult:
        """Invoke one allowlisted tool after strict argument validation."""

        if tool_name == SEARCH_CORPUS_TOOL_NAME:
            parsed = SearchCorpusArguments.from_mapping(arguments)
            return self.search_corpus(parsed.query, parsed.top_k)
        if tool_name == GET_SOURCE_METADATA_TOOL_NAME:
            parsed_metadata = SourceMetadataArguments.from_mapping(arguments)
            return self.get_source_metadata(parsed_metadata.source_id)
        allowed = ", ".join(SUPPORTED_TOOL_NAMES)
        raise RetrievalError(f"Unsupported agent tool {tool_name!r}; allowed tools are: {allowed}.")

    def openai_tool_definitions(self) -> list[dict[str, object]]:
        """Return fresh strict function definitions for the Responses API."""

        return openai_tool_definitions()


def openai_tool_definitions() -> list[dict[str, object]]:
    """Return exactly the two allowlisted OpenAI function definitions."""

    return [
        {
            "type": "function",
            "name": SEARCH_CORPUS_TOOL_NAME,
            "description": (
                "Search the indexed corpus for ranked evidence. Use the returned "
                "citation_label verbatim in supported final claims."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A non-empty evidence search query.",
                        "minLength": 1,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of ranked chunks to return.",
                        "minimum": 1,
                    },
                },
                "required": ["query", "top_k"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": GET_SOURCE_METADATA_TOOL_NAME,
            "description": (
                "Look up allowlisted provenance metadata for a source_id returned by search_corpus."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "A source_id returned by search_corpus.",
                        "minLength": 1,
                    }
                },
                "required": ["source_id"],
                "additionalProperties": False,
            },
        },
    ]


def _validate_argument_keys(
    arguments: Mapping[str, object],
    *,
    expected: set[str],
    tool_name: str,
) -> None:
    actual = set(arguments)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise RetrievalError(f"{tool_name} received invalid arguments ({'; '.join(details)}).")


def _validate_excerpt_limit(max_excerpt_chars: int) -> None:
    if (
        isinstance(max_excerpt_chars, bool)
        or not isinstance(max_excerpt_chars, int)
        or max_excerpt_chars <= 0
    ):
        raise RetrievalError("max_excerpt_chars must be a positive integer.")


def _match_from_retrieval(
    result: RetrievalResult,
    *,
    max_excerpt_chars: int,
) -> SearchCorpusMatch:
    expected_label = f"[{result.source_name}#{result.chunk_id}]"
    if result.citation_label != expected_label:
        raise CitationValidationError("Retrieved evidence contains a non-canonical citation label.")
    if isinstance(result.rank, bool) or not isinstance(result.rank, int) or result.rank <= 0:
        raise RetrievalError("search_corpus received an invalid retrieval rank.")
    if (
        isinstance(result.score, bool)
        or not isinstance(result.score, (int, float))
        or not math.isfinite(float(result.score))
    ):
        raise RetrievalError("search_corpus received a non-finite retrieval score.")
    if not isinstance(result.text, str) or not result.text.strip():
        raise RetrievalError("search_corpus received empty retrieval text.")
    return SearchCorpusMatch(
        rank=result.rank,
        score=float(result.score),
        chunk_id=result.chunk_id,
        source_id=result.source_id,
        source_name=result.source_name,
        excerpt=_truncate_excerpt(result.text, max_chars=max_excerpt_chars),
        page_number=result.page_number,
        citation_label=result.citation_label,
    )


def _truncate_excerpt(text: str, *, max_chars: int) -> str:
    without_headings = _MARKDOWN_HEADING_LINE.sub("", text)
    normalised = _WHITESPACE.sub(" ", without_headings).strip()
    if len(normalised) <= max_chars:
        return normalised
    if max_chars <= 3:
        return "." * max_chars
    return f"{normalised[: max_chars - 3].rstrip()}..."
