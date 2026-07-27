"""Pure deterministic metrics for retrieval, answers, citations, and latency."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_agent_eval_toolkit.models import RetrievalResult

P95_MIN_SAMPLE_SIZE = 20

_CITATION_CANDIDATE_PATTERN = re.compile(r"\[[^\[\]\r\n]*#[^\[\]\r\n]*\]")
_CANONICAL_CITATION_PATTERN = re.compile(r"^\[(?P<source>[^\[\]#\r\n]+)#(?P<chunk>[0-9a-f]{20})\]$")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_ABSTENTION_SENTENCE_PATTERNS = (
    re.compile(
        r"^(?:the )?(?:provided )?(?:documents|corpus) "
        r"(?:do|does) not (?:contain|provide|identify|state|specify|include)(?: .+)?$"
    ),
    re.compile(r"^(?:there is )?insufficient evidence(?: .+)?$"),
    re.compile(
        r"^i (?:do not|don't) have enough evidence"
        r"(?: in (?:the )?(?:retrieved context|provided documents|corpus))?"
        r"(?: to .+)?$"
    ),
    re.compile(
        r"^i (?:cannot|can't) determine(?: .+)? from "
        r"(?:the )?(?:provided )?(?:documents|corpus)$"
    ),
)


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Per-question retrieval metrics and supporting ranked evidence."""

    hit_rate_at_k: float
    reciprocal_rank: float
    source_recall_at_k: float
    first_relevant_rank: int | None
    retrieved_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationScore:
    """Per-question citation metrics and parsed citation evidence."""

    validity: float
    precision: float
    citation_labels: tuple[str, ...]
    valid_labels: tuple[str, ...]
    invalid_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AbstentionAssessment:
    """Structured result of the deterministic abstention policy."""

    expected_to_abstain: bool
    explicit_abstention: bool
    contains_additional_claims: bool
    correct: bool | None


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Aggregate latency values with an explicit p95 sample-size policy."""

    count: int
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    warnings: tuple[str, ...]


def retrieval_score(
    results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
    *,
    top_k: int,
) -> RetrievalScore:
    """Calculate Hit Rate@k, reciprocal rank, and deduplicated Source Recall@k."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    expected = _deduplicate_non_empty(expected_sources, name="expected_sources")
    if not expected:
        raise ValueError("retrieval metrics require at least one expected source")

    ranked = sorted(results, key=lambda result: result.rank)
    considered = [result for result in ranked if result.rank <= top_k]
    retrieved_sources = tuple(
        _deduplicate_preserving_order(result.source_name for result in considered)
    )
    expected_set = set(expected)

    first_rank = next(
        (result.rank for result in considered if result.source_name in expected_set),
        None,
    )
    retrieved_expected_sources = expected_set.intersection(retrieved_sources)
    return RetrievalScore(
        hit_rate_at_k=1.0 if first_rank is not None else 0.0,
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        source_recall_at_k=len(retrieved_expected_sources) / len(expected_set),
        first_relevant_rank=first_rank,
        retrieved_sources=retrieved_sources,
    )


def normalize_for_fact_matching(text: str) -> str:
    """Lowercase text, replace Unicode punctuation with spaces, and collapse whitespace."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text.lower()
    )
    return " ".join(without_punctuation.split())


def required_fact_coverage(answer: str, required_facts: Sequence[str]) -> float | None:
    """Return normalized substring fact coverage, or ``None`` when no facts apply."""

    facts = _deduplicate_non_empty(required_facts, name="required_facts")
    if not facts:
        return None
    normalized_answer = normalize_for_fact_matching(answer)
    matches = sum(normalize_for_fact_matching(fact) in normalized_answer for fact in facts)
    return matches / len(facts)


def extract_citation_labels(answer: str) -> tuple[str, ...]:
    """Extract citation-like bracket tokens containing a source/chunk separator."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    return tuple(match.group(0) for match in _CITATION_CANDIDATE_PATTERN.finditer(answer))


def citation_score(
    answer: str,
    retrieval_results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
) -> CitationScore:
    """Score citation validity and expected-source precision for one answer.

    Both metrics use all citation occurrences as their denominator. A citation is
    valid only when it has canonical syntax and exactly resolves to a chunk that
    was retrieved and supplied to generation. An answer with no citations scores
    zero for both metrics.
    """

    expected = set(_deduplicate_non_empty(expected_sources, name="expected_sources"))
    if not expected:
        raise ValueError("citation metrics require at least one expected source")

    labels = extract_citation_labels(answer)
    retrieved_labels = {result.citation_label for result in retrieval_results}
    valid_labels: list[str] = []
    invalid_labels: list[str] = []
    valid_expected_count = 0

    for label in labels:
        match = _CANONICAL_CITATION_PATTERN.fullmatch(label)
        if match is None or label not in retrieved_labels:
            invalid_labels.append(label)
            continue
        valid_labels.append(label)
        if match.group("source") in expected:
            valid_expected_count += 1

    denominator = len(labels)
    validity = len(valid_labels) / denominator if denominator else 0.0
    precision = valid_expected_count / denominator if denominator else 0.0
    return CitationScore(
        validity=validity,
        precision=precision,
        citation_labels=labels,
        valid_labels=tuple(valid_labels),
        invalid_labels=tuple(invalid_labels),
    )


def assess_abstention(answer: str, *, answerable: bool) -> AbstentionAssessment:
    """Apply the explicit, abstention-only sentence policy.

    A correct unanswerable response consists solely of one or more recognized
    abstention sentences. Any additional sentence is treated as a possible
    unsupported factual claim. For answerable questions, ``correct`` is not an
    unanswerable metric and is therefore ``None``.
    """

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    without_citations = _CITATION_CANDIDATE_PATTERN.sub("", answer)
    sentences = [
        " ".join(sentence.strip().lower().split())
        for sentence in _SENTENCE_SPLIT_PATTERN.split(without_citations)
        if sentence.strip()
    ]
    recognized = [
        any(pattern.fullmatch(sentence) is not None for pattern in _ABSTENTION_SENTENCE_PATTERNS)
        for sentence in sentences
    ]
    explicit_abstention = bool(recognized) and any(recognized)
    contains_additional_claims = explicit_abstention and not all(recognized)
    correct = explicit_abstention and not contains_additional_claims if not answerable else None
    return AbstentionAssessment(
        expected_to_abstain=not answerable,
        explicit_abstention=explicit_abstention,
        contains_additional_claims=contains_additional_claims,
        correct=correct,
    )


def latency_summary(
    values_ms: Iterable[float],
    *,
    metric_name: str = "latency",
    p95_min_sample_size: int = P95_MIN_SAMPLE_SIZE,
) -> LatencySummary:
    """Aggregate mean/median and nearest-rank p95 with a sample-size caveat."""

    if (
        isinstance(p95_min_sample_size, bool)
        or not isinstance(p95_min_sample_size, int)
        or p95_min_sample_size < 1
    ):
        raise ValueError("p95_min_sample_size must be a positive integer")
    values = tuple(_validated_latency(value) for value in values_ms)
    if not values:
        return LatencySummary(
            count=0,
            mean_ms=None,
            median_ms=None,
            p95_ms=None,
            warnings=(f"{metric_name} has no successful observations.",),
        )

    warnings: list[str] = []
    p95: float | None = None
    if len(values) >= p95_min_sample_size:
        ordered = sorted(values)
        nearest_rank_index = math.ceil(0.95 * len(ordered)) - 1
        p95 = ordered[nearest_rank_index]
    else:
        warnings.append(
            f"{metric_name} p95 omitted: {len(values)} observations; "
            f"at least {p95_min_sample_size} are required."
        )

    return LatencySummary(
        count=len(values),
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=p95,
        warnings=tuple(warnings),
    )


def _deduplicate_non_empty(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain only non-empty strings")
        parsed.append(value)
    return tuple(_deduplicate_preserving_order(parsed))


def _deduplicate_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validated_latency(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency values must be finite non-negative numbers")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("latency values must be finite non-negative numbers")
    return parsed


__all__ = [
    "P95_MIN_SAMPLE_SIZE",
    "AbstentionAssessment",
    "CitationScore",
    "LatencySummary",
    "RetrievalScore",
    "assess_abstention",
    "citation_score",
    "extract_citation_labels",
    "latency_summary",
    "normalize_for_fact_matching",
    "required_fact_coverage",
    "retrieval_score",
]
