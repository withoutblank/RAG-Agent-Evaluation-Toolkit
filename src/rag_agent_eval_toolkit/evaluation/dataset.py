"""Strict loading and validation for the versioned evaluation JSONL dataset."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_agent_eval_toolkit.exceptions import EvaluationDatasetError
from rag_agent_eval_toolkit.models import EvaluationQuestion

EVALUATION_DATASET_SCHEMA_VERSION = 1

_QUESTION_FIELDS = frozenset(
    {
        "id",
        "question",
        "expected_answer",
        "expected_sources",
        "required_facts",
        "answerable",
        "tags",
        "difficulty",
    }
)
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A validated evaluation dataset and its immutable provenance."""

    schema_version: int
    path: Path
    checksum_sha256: str
    questions: tuple[EvaluationQuestion, ...]

    def __len__(self) -> int:
        """Return the number of labelled questions."""

        return len(self.questions)


def load_evaluation_dataset(
    path: Path,
    *,
    schema_version: int = EVALUATION_DATASET_SCHEMA_VERSION,
    require_mvp_coverage: bool = False,
) -> EvaluationDataset:
    """Load a UTF-8 JSONL dataset using the exact schema-version-1 contract.

    The checksum is SHA-256 over the file's raw bytes. The schema version belongs
    to this parser because JSONL question records deliberately contain only the
    fields defined in :mod:`DATA_MODEL_AND_INTERFACES.md`.
    """

    if isinstance(schema_version, bool) or schema_version != EVALUATION_DATASET_SCHEMA_VERSION:
        raise EvaluationDatasetError(
            f"Unsupported evaluation dataset schema version: {schema_version!r}. "
            f"Supported version: {EVALUATION_DATASET_SCHEMA_VERSION}."
        )

    resolved_path = Path(path)
    if resolved_path.suffix.lower() != ".jsonl":
        raise EvaluationDatasetError(
            f"Evaluation dataset must use the .jsonl extension: {resolved_path}"
        )
    try:
        raw_bytes = resolved_path.read_bytes()
    except OSError as exc:
        raise EvaluationDatasetError(
            f"Unable to read evaluation dataset {resolved_path}: {exc}"
        ) from exc
    if not raw_bytes:
        raise EvaluationDatasetError(f"Evaluation dataset is empty: {resolved_path}")

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset must be valid UTF-8: {resolved_path}"
        ) from exc
    if content.startswith("\ufeff"):
        raise EvaluationDatasetError("Evaluation dataset must not contain a UTF-8 BOM.")

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise EvaluationDatasetError(
                f"Blank lines are not allowed in evaluation JSONL (line {line_number})."
            )
        try:
            loaded: object = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvaluationDatasetError(
                f"Invalid JSON on evaluation dataset line {line_number}: {exc}"
            ) from exc
        question = _parse_question(loaded, line_number=line_number)
        if question.id in seen_ids:
            raise EvaluationDatasetError(
                f"Duplicate evaluation question id {question.id!r} on line {line_number}."
            )
        normalized_question = " ".join(question.question.casefold().split())
        if normalized_question in seen_questions:
            raise EvaluationDatasetError(
                f"Duplicate evaluation question text on line {line_number}."
            )
        seen_ids.add(question.id)
        seen_questions.add(normalized_question)
        questions.append(question)

    if not questions:
        raise EvaluationDatasetError(f"Evaluation dataset has no records: {resolved_path}")

    dataset = EvaluationDataset(
        schema_version=schema_version,
        path=resolved_path,
        checksum_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        questions=tuple(questions),
    )
    if require_mvp_coverage:
        validate_mvp_dataset(dataset)
    return dataset


def validate_mvp_dataset(dataset: EvaluationDataset) -> None:
    """Validate the minimum coverage required by the MVP benchmark."""

    problems: list[str] = []
    if len(dataset.questions) < 20:
        problems.append("at least 20 questions are required")

    unanswerable_count = sum(not question.answerable for question in dataset.questions)
    if unanswerable_count < 4:
        problems.append("at least four unanswerable questions are required")

    represented_sources = {
        source for question in dataset.questions for source in question.expected_sources
    }
    if len(represented_sources) < 8:
        problems.append("at least eight expected source documents must be represented")

    represented_difficulties = {question.difficulty for question in dataset.questions}
    missing_difficulties = sorted(_DIFFICULTIES - represented_difficulties)
    if missing_difficulties:
        problems.append(
            "all difficulty levels must be represented; missing " + ", ".join(missing_difficulties)
        )

    if not any(len(set(question.expected_sources)) >= 2 for question in dataset.questions):
        problems.append("at least one multi-source question is required")

    if problems:
        raise EvaluationDatasetError("Dataset does not meet MVP coverage: " + "; ".join(problems))


def _parse_question(value: object, *, line_number: int) -> EvaluationQuestion:
    mapping = _require_mapping(value, line_number=line_number)
    supplied_fields = frozenset(mapping)
    missing_fields = sorted(_QUESTION_FIELDS - supplied_fields)
    unknown_fields = sorted(supplied_fields - _QUESTION_FIELDS)
    if missing_fields or unknown_fields:
        details: list[str] = []
        if missing_fields:
            details.append("missing: " + ", ".join(missing_fields))
        if unknown_fields:
            details.append("unknown: " + ", ".join(unknown_fields))
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} has invalid fields ({'; '.join(details)})."
        )

    identifier = _require_text(mapping["id"], field_name="id", line_number=line_number)
    if _IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field 'id' has an invalid format."
        )
    question_text = _require_text(
        mapping["question"], field_name="question", line_number=line_number
    )
    expected_answer = _require_text(
        mapping["expected_answer"],
        field_name="expected_answer",
        line_number=line_number,
    )
    expected_sources = _require_unique_text_list(
        mapping["expected_sources"],
        field_name="expected_sources",
        line_number=line_number,
    )
    for source_name in expected_sources:
        if (
            "/" in source_name
            or "\\" in source_name
            or source_name in {".", ".."}
            or Path(source_name).name != source_name
        ):
            raise EvaluationDatasetError(
                f"Evaluation dataset line {line_number} contains a non-portable "
                "expected source name."
            )

    required_facts = _require_unique_text_list(
        mapping["required_facts"],
        field_name="required_facts",
        line_number=line_number,
    )
    tags = _require_unique_text_list(
        mapping["tags"],
        field_name="tags",
        line_number=line_number,
    )
    if not tags:
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field 'tags' must not be empty."
        )

    answerable_value = mapping["answerable"]
    if not isinstance(answerable_value, bool):
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field 'answerable' must be a boolean."
        )
    difficulty = _require_text(
        mapping["difficulty"], field_name="difficulty", line_number=line_number
    )
    if difficulty not in _DIFFICULTIES:
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field 'difficulty' must be one of "
            f"{', '.join(sorted(_DIFFICULTIES))}."
        )

    if answerable_value and (not expected_sources or not required_facts):
        raise EvaluationDatasetError(
            f"Answerable evaluation dataset line {line_number} must define expected_sources "
            "and required_facts."
        )
    if not answerable_value and (expected_sources or required_facts):
        raise EvaluationDatasetError(
            f"Unanswerable evaluation dataset line {line_number} must have empty "
            "expected_sources and required_facts."
        )

    return EvaluationQuestion(
        id=identifier,
        question=question_text,
        expected_answer=expected_answer,
        expected_sources=expected_sources,
        required_facts=required_facts,
        answerable=answerable_value,
        tags=tags,
        difficulty=difficulty,
    )


def _require_mapping(value: object, *, line_number: int) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} must be a JSON object with string keys."
        )
    return {str(key): item for key, item in value.items()}


def _require_text(value: object, *, field_name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field {field_name!r} must be "
            "a non-empty, trimmed string."
        )
    return value


def _require_unique_text_list(
    value: object,
    *,
    field_name: str,
    line_number: int,
) -> list[str]:
    if not isinstance(value, list):
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field {field_name!r} must be a list."
        )
    parsed: list[str] = []
    for item in value:
        parsed.append(_require_text(item, field_name=field_name, line_number=line_number))
    if len(parsed) != len(set(parsed)):
        raise EvaluationDatasetError(
            f"Evaluation dataset line {line_number} field {field_name!r} "
            "must not contain duplicates."
        )
    return parsed


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


__all__ = [
    "EVALUATION_DATASET_SCHEMA_VERSION",
    "EvaluationDataset",
    "load_evaluation_dataset",
    "validate_mvp_dataset",
]
