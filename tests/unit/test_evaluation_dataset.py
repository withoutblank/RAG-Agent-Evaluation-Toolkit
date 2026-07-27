"""Strict evaluation JSONL loader tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_agent_eval_toolkit.evaluation.dataset import (
    EVALUATION_DATASET_SCHEMA_VERSION,
    load_evaluation_dataset,
)
from rag_agent_eval_toolkit.exceptions import EvaluationDatasetError


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "ev-test",
        "question": "What is the documented fact?",
        "expected_answer": "The documented fact is one.",
        "expected_sources": ["guide.md"],
        "required_facts": ["fact is one"],
        "answerable": True,
        "tags": ["test"],
        "difficulty": "easy",
    }
    record.update(overrides)
    return record


def _write_records(path: Path, records: list[dict[str, object]]) -> bytes:
    raw = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    path.write_text(raw, encoding="utf-8", newline="\n")
    return raw.encode()


def test_load_committed_evaluation_dataset_and_validate_mvp() -> None:
    path = Path("evals/fictional_ev_support_questions.jsonl")
    raw = path.read_bytes()

    dataset = load_evaluation_dataset(path, require_mvp_coverage=True)

    assert dataset.schema_version == EVALUATION_DATASET_SCHEMA_VERSION
    assert dataset.checksum_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(dataset.questions) == 20
    assert sum(not question.answerable for question in dataset.questions) == 4
    assert dataset.questions[0].id == "ev-001"


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record(extra="not allowed")], "unknown: extra"),
        ([_record(required_facts=[])], "must define expected_sources and required_facts"),
        (
            [
                _record(
                    answerable=False,
                    expected_sources=["guide.md"],
                    required_facts=[],
                )
            ],
            "must have empty",
        ),
        ([_record(tags=["test", "test"])], "must not contain duplicates"),
        ([_record(difficulty="extreme")], "difficulty"),
        ([_record(expected_sources=["../guide.md"])], "non-portable"),
    ],
)
def test_loader_rejects_schema_and_consistency_errors(
    tmp_path: Path,
    records: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "questions.jsonl"
    _write_records(path, records)

    with pytest.raises(EvaluationDatasetError, match=message):
        load_evaluation_dataset(path)


def test_loader_rejects_duplicate_ids_and_blank_lines(tmp_path: Path) -> None:
    duplicate_path = tmp_path / "duplicate.jsonl"
    _write_records(duplicate_path, [_record(), _record(question="A different question?")])

    with pytest.raises(EvaluationDatasetError, match="Duplicate evaluation question id"):
        load_evaluation_dataset(duplicate_path)

    blank_path = tmp_path / "blank.jsonl"
    blank_path.write_text(
        json.dumps(_record()) + "\n\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(EvaluationDatasetError, match="Blank lines"):
        load_evaluation_dataset(blank_path)


def test_loader_rejects_unsupported_version_bom_and_invalid_utf8(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.jsonl"
    _write_records(valid_path, [_record()])
    with pytest.raises(EvaluationDatasetError, match="Unsupported evaluation"):
        load_evaluation_dataset(valid_path, schema_version=2)

    bom_path = tmp_path / "bom.jsonl"
    bom_path.write_text(
        "\ufeff" + json.dumps(_record()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationDatasetError, match="must not contain a UTF-8 BOM"):
        load_evaluation_dataset(bom_path)

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_bytes(b"\xff")
    with pytest.raises(EvaluationDatasetError, match="valid UTF-8"):
        load_evaluation_dataset(invalid_path)


def test_mvp_validation_reports_incomplete_fixture(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _write_records(path, [_record()])

    with pytest.raises(EvaluationDatasetError, match="at least 20 questions"):
        load_evaluation_dataset(path, require_mvp_coverage=True)
