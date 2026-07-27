"""Citation parsing and resolution against retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_agent_eval_toolkit.exceptions import CitationValidationError
from rag_agent_eval_toolkit.models import Citation, RetrievalResult

_CITATION_PATTERN = re.compile(r"\[[^\[\]#\r\n]+#[^\[\]#\s\r\n]+\]")


def parse_citation_labels(answer: str) -> list[str]:
    """Return unique citation labels in their first-occurrence order."""

    if not isinstance(answer, str):
        raise CitationValidationError("Answer must be text before citations can be parsed.")
    return list(dict.fromkeys(match.group(0) for match in _CITATION_PATTERN.finditer(answer)))


def validate_citations(
    answer: str,
    retrieval_results: Sequence[RetrievalResult],
    *,
    require_citation: bool = True,
) -> list[Citation]:
    """Resolve answer citations and reject labels absent from retrieved evidence."""

    evidence_by_label: dict[str, RetrievalResult] = {}
    for result in retrieval_results:
        expected_label = f"[{result.source_name}#{result.chunk_id}]"
        if result.citation_label != expected_label:
            raise CitationValidationError(
                "Retrieved evidence contains a non-canonical citation label."
            )
        existing = evidence_by_label.get(result.citation_label)
        if existing is not None and existing != result:
            raise CitationValidationError(
                "Retrieved evidence contains a duplicated citation label."
            )
        evidence_by_label[result.citation_label] = result

    labels = parse_citation_labels(answer)
    if require_citation and not labels:
        raise CitationValidationError(
            "A non-abstaining RAG answer must include at least one citation."
        )

    invalid_labels = [label for label in labels if label not in evidence_by_label]
    if invalid_labels:
        joined = ", ".join(invalid_labels)
        raise CitationValidationError(f"Answer cited evidence that was not retrieved: {joined}")

    return [
        Citation(
            citation_label=label,
            source_id=evidence_by_label[label].source_id,
            source_name=evidence_by_label[label].source_name,
            chunk_id=evidence_by_label[label].chunk_id,
            page_number=evidence_by_label[label].page_number,
        )
        for label in labels
    ]
