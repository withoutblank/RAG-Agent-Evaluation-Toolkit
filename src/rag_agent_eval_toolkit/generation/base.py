"""Shared interfaces and result types for answer-generation providers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from rag_agent_eval_toolkit.exceptions import GenerationProviderError

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough evidence in the retrieved context to answer that question."
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Text and optional provider-reported token usage from one generation call."""

    text: str
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Validate provider output before it reaches the RAG boundary."""

        if not isinstance(self.text, str) or not self.text.strip():
            raise GenerationProviderError("Generation result text must not be empty.")
        if self.usage is None:
            return
        for key, value in self.usage.items():
            if not isinstance(key, str) or not key:
                raise GenerationProviderError("Generation usage keys must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GenerationProviderError(
                    "Generation usage values must be non-negative integers."
                )
        object.__setattr__(self, "usage", dict(self.usage))


@runtime_checkable
class GenerationProvider(Protocol):
    """Protocol implemented by deterministic and hosted generation providers."""

    @property
    def model_name(self) -> str:
        """Return the stable model identifier recorded with generated answers."""

        ...

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate one response for a complete prompt."""

        ...


def _validate_generation_request(prompt: str, temperature: float) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationProviderError("Generation prompt must contain non-whitespace text.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise GenerationProviderError("temperature must be a finite number between 0 and 2.")
    return prompt
