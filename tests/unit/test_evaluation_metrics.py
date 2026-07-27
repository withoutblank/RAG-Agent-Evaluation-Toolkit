"""Hand-calculated deterministic metric tests."""

from __future__ import annotations

import pytest

from rag_agent_eval_toolkit.evaluation.metrics import (
    assess_abstention,
    citation_score,
    latency_summary,
    normalize_for_fact_matching,
    required_fact_coverage,
    retrieval_score,
)
from rag_agent_eval_toolkit.models import RetrievalResult

_A_CHUNK = "a" * 20
_B_CHUNK = "b" * 20
_X_CHUNK = "c" * 20


def _result(rank: int, source_name: str, chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        query="question",
        rank=rank,
        score=1.0 / rank,
        chunk_id=chunk_id,
        source_id=f"source-{source_name}",
        source_name=source_name,
        text=f"Evidence from {source_name}.",
        page_number=None,
        citation_label=f"[{source_name}#{chunk_id}]",
    )


def test_retrieval_metrics_are_hand_calculated_and_sources_are_deduplicated() -> None:
    results = [
        _result(1, "irrelevant.md", _X_CHUNK),
        _result(2, "a.md", _A_CHUNK),
        _result(3, "a.md", "d" * 20),
        _result(4, "b.md", _B_CHUNK),
    ]

    score_at_three = retrieval_score(results, ["a.md", "b.md", "a.md"], top_k=3)
    score_at_four = retrieval_score(results, ["a.md", "b.md"], top_k=4)

    assert score_at_three.hit_rate_at_k == 1.0
    assert score_at_three.reciprocal_rank == 0.5
    assert score_at_three.source_recall_at_k == 0.5
    assert score_at_three.first_relevant_rank == 2
    assert score_at_three.retrieved_sources == ("irrelevant.md", "a.md")
    assert score_at_four.source_recall_at_k == 1.0


def test_retrieval_metrics_score_a_complete_miss_as_zero() -> None:
    score = retrieval_score(
        [_result(1, "irrelevant.md", _X_CHUNK)],
        ["a.md"],
        top_k=1,
    )

    assert score.hit_rate_at_k == 0.0
    assert score.reciprocal_rank == 0.0
    assert score.source_recall_at_k == 0.0
    assert score.first_relevant_rank is None


def test_fact_normalization_and_coverage_are_deterministic() -> None:
    answer = "Inspect the connector; then FOLLOW safety-instructions."
    facts = ["inspect the connector", "follow safety instructions", "confirm compatibility"]

    assert normalize_for_fact_matching("Safety—Instructions!") == "safety instructions"
    assert required_fact_coverage(answer, facts) == pytest.approx(2 / 3)
    assert required_fact_coverage(answer, []) is None


def test_citation_validity_and_precision_use_all_occurrences() -> None:
    retrieval_results = [
        _result(1, "a.md", _A_CHUNK),
        _result(2, "b.md", _B_CHUNK),
    ]
    answer = (
        f"Expected [{retrieval_results[0].source_name}#{_A_CHUNK}] "
        f"unexpected-but-valid [b.md#{_B_CHUNK}] "
        f"not-retrieved [a.md#{'e' * 20}] and malformed [a.md#short]."
    )

    score = citation_score(answer, retrieval_results, ["a.md"])

    assert score.validity == 0.5
    assert score.precision == 0.25
    assert len(score.citation_labels) == 4
    assert score.valid_labels == (f"[a.md#{_A_CHUNK}]", f"[b.md#{_B_CHUNK}]")
    assert score.invalid_labels == (f"[a.md#{'e' * 20}]", "[a.md#short]")


def test_missing_citations_score_zero_for_answerable_questions() -> None:
    score = citation_score("An uncited answer.", [_result(1, "a.md", _A_CHUNK)], ["a.md"])

    assert score.validity == 0.0
    assert score.precision == 0.0


@pytest.mark.parametrize(
    ("answer", "expected_correct", "additional_claims"),
    [
        ("The provided documents do not contain an insurance premium.", True, False),
        ("Insufficient evidence in the provided corpus.", True, False),
        (
            "I don't have enough evidence in the retrieved context to answer that question.",
            True,
            False,
        ),
        ("I cannot determine this from the corpus.", True, False),
        (
            "The provided documents do not contain the price. It costs 500 dollars.",
            False,
            True,
        ),
        ("It costs 500 dollars.", False, False),
    ],
)
def test_structured_abstention_policy(
    answer: str,
    expected_correct: bool,
    additional_claims: bool,
) -> None:
    assessment = assess_abstention(answer, answerable=False)

    assert assessment.correct is expected_correct
    assert assessment.contains_additional_claims is additional_claims


def test_answerable_abstention_is_tracked_but_not_scored_as_unanswerable() -> None:
    assessment = assess_abstention(
        "The provided documents do not contain that information.",
        answerable=True,
    )

    assert assessment.explicit_abstention is True
    assert assessment.correct is None


def test_latency_aggregation_mean_median_and_caveated_nearest_rank_p95() -> None:
    small = latency_summary([1.0, 2.0, 3.0], metric_name="retrieval")
    sufficient = latency_summary(range(1, 21), metric_name="total")

    assert small.count == 3
    assert small.mean_ms == 2.0
    assert small.median_ms == 2.0
    assert small.p95_ms is None
    assert "p95 omitted" in small.warnings[0]

    assert sufficient.count == 20
    assert sufficient.mean_ms == 10.5
    assert sufficient.median_ms == 10.5
    assert sufficient.p95_ms == 19.0
    assert sufficient.warnings == ()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_latency_aggregation_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        latency_summary([value])
