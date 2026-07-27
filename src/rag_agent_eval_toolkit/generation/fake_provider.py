"""Deterministic local answer generation for tests and offline demonstrations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from rag_agent_eval_toolkit.exceptions import GenerationProviderError
from rag_agent_eval_toolkit.generation.base import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerationResult,
    _validate_generation_request,
)

_CONTEXT_PATTERN = re.compile(
    r"<retrieved_context>\s*(.*?)\s*</retrieved_context>",
    flags=re.DOTALL,
)
_QUESTION_PATTERN = re.compile(
    r"<question_json>\s*(.*?)\s*</question_json>",
    flags=re.DOTALL,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_LEADING_MARKDOWN_HEADING = re.compile(r"^(?:#{1,6}\s+[^\n]+\n+)+")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "does",
        "driver",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "vehicle",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)
_TOKEN_ALIASES = {
    "charge": "charge",
    "charged": "charge",
    "charger": "charge",
    "chargers": "charge",
    "charging": "charge",
    "check": "inspect",
    "checked": "inspect",
    "checking": "inspect",
    "checks": "inspect",
    "connect": "connect",
    "connected": "connect",
    "connecting": "connect",
    "connection": "connect",
    "connector": "connect",
    "connectors": "connect",
    "inspect": "inspect",
    "inspected": "inspect",
    "inspecting": "inspect",
    "inspection": "inspect",
    "inspections": "inspect",
}


class DeterministicFakeGenerationProvider:
    """Return reproducible cited text without making a network request.

    With no fixed response, the provider extracts the first evidence record from
    the toolkit's JSON context and cites it. This is useful for testing pipeline
    plumbing, but it is not a semantic language model and must not be presented
    as answer-quality evidence.
    """

    provider_name = "fake"

    def __init__(
        self,
        response: str | None = None,
        *,
        model_name: str = "fake-generation-v1",
    ) -> None:
        """Configure an optional fixed response and a stable model identifier."""

        if response is not None and (not isinstance(response, str) or not response.strip()):
            raise GenerationProviderError("Fake generation response must not be empty.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise GenerationProviderError("Fake generation model name must not be empty.")
        self._response = response.strip() if response is not None else None
        self._model_name = model_name.strip()
        self._prompts: list[str] = []

    @property
    def model_name(self) -> str:
        """Return the configured fake model identifier."""

        return self._model_name

    @property
    def prompts(self) -> tuple[str, ...]:
        """Return prompts received by this instance for offline test inspection."""

        return tuple(self._prompts)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate a fixed or first-evidence response deterministically."""

        validated_prompt = _validate_generation_request(prompt, temperature)
        self._prompts.append(validated_prompt)
        if self._response is not None:
            return GenerationResult(text=self._response)
        return GenerationResult(text=_answer_from_context(validated_prompt))


def _answer_from_context(prompt: str) -> str:
    question = _question_from_prompt(prompt)
    match = _CONTEXT_PATTERN.search(prompt)
    if match is None:
        return INSUFFICIENT_EVIDENCE_ANSWER
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise GenerationProviderError("Fake generator received malformed context JSON.") from exc
    if not isinstance(payload, list) or not payload:
        return INSUFFICIENT_EVIDENCE_ANSWER

    question_terms = _normalised_terms(question)
    candidates: list[tuple[tuple[int, int, int, int], str, str]] = []
    for evidence_index, evidence in enumerate(payload):
        if not isinstance(evidence, Mapping):
            raise GenerationProviderError("Fake generator received an invalid evidence record.")
        text = evidence.get("text")
        citation_label = evidence.get("citation_label")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(citation_label, str)
            or not citation_label.strip()
        ):
            raise GenerationProviderError("Fake generator received incomplete evidence.")

        sentences = [
            _LEADING_MARKDOWN_HEADING.sub("", sentence.strip()).strip()
            for sentence in _SENTENCE_BOUNDARY.split(text.strip())
            if _LEADING_MARKDOWN_HEADING.sub("", sentence.strip()).strip()
        ]
        for sentence_index, sentence in enumerate(sentences):
            overlap = len(question_terms.intersection(_normalised_terms(sentence)))
            starts_cleanly = int(sentence[0].isupper() or sentence[0].isdigit())
            score = (
                overlap,
                starts_cleanly,
                -evidence_index,
                -sentence_index,
            )
            candidates.append((score, sentence, citation_label.strip()))

    if not candidates:
        return INSUFFICIENT_EVIDENCE_ANSWER
    _, sentence, citation_label = max(candidates, key=lambda candidate: candidate[0])
    return f"{sentence} {citation_label}"


def _question_from_prompt(prompt: str) -> str:
    match = _QUESTION_PATTERN.search(prompt)
    if match is None:
        return ""
    try:
        question = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise GenerationProviderError("Fake generator received malformed question JSON.") from exc
    if not isinstance(question, str):
        raise GenerationProviderError("Fake generator received an invalid question.")
    return question


def _normalised_terms(text: str) -> set[str]:
    return {
        _TOKEN_ALIASES.get(token, token)
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOP_WORDS
    }


FakeGenerationProvider = DeterministicFakeGenerationProvider
