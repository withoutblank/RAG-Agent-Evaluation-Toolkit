"""Direct retrieval-augmented generation with strict citation resolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from math import isfinite
from time import perf_counter

from rag_agent_eval_toolkit.exceptions import (
    GenerationProviderError,
    RetrievalError,
)
from rag_agent_eval_toolkit.generation.base import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerationProvider,
    GenerationResult,
)
from rag_agent_eval_toolkit.models import RagAnswer
from rag_agent_eval_toolkit.rag.citations import validate_citations
from rag_agent_eval_toolkit.rag.prompts import (
    DEFAULT_PROMPT_VERSION,
    build_rag_prompt,
)
from rag_agent_eval_toolkit.retrieval import Retriever

logger = logging.getLogger(__name__)


class DirectRagPipeline:
    """Retrieve evidence, generate a constrained answer, and resolve citations."""

    def __init__(
        self,
        retriever: Retriever,
        generation_provider: GenerationProvider,
        *,
        default_top_k: int = 3,
        temperature: float = 0.0,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        """Configure deterministic direct-RAG behavior with injected providers."""

        _validate_top_k(default_top_k)
        _validate_temperature(temperature)
        if prompt_version != DEFAULT_PROMPT_VERSION:
            raise GenerationProviderError(
                f"Unsupported RAG prompt version {prompt_version!r}; "
                f"expected {DEFAULT_PROMPT_VERSION!r}."
            )
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._default_top_k = default_top_k
        self._temperature = float(temperature)
        self._prompt_version = prompt_version
        self._clock = clock

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> RagAnswer:
        """Answer one question from ranked evidence or explicitly abstain."""

        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("RAG question must contain non-whitespace text.")
        requested_top_k = self._default_top_k if top_k is None else top_k
        _validate_top_k(requested_top_k)

        total_started = self._clock()
        retrieval_started = self._clock()
        retrieval_results = self._retriever.retrieve(
            question,
            top_k=requested_top_k,
        )
        retrieval_finished = self._clock()
        retrieval_latency_ms = _elapsed_ms(retrieval_started, retrieval_finished)

        if not retrieval_results:
            total_latency_ms = _elapsed_ms(total_started, self._clock())
            answer = RagAnswer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                retrieval_results=[],
                model_name=self._generation_provider.model_name,
                prompt_version=self._prompt_version,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                usage=None,
                warnings=["No evidence was retrieved; generation was skipped."],
            )
            _log_completion(answer)
            return answer

        prompt = build_rag_prompt(
            question,
            retrieval_results,
            prompt_version=self._prompt_version,
        )
        generation_started = self._clock()
        try:
            generation_result = self._generation_provider.generate(
                prompt,
                temperature=self._temperature,
            )
        except GenerationProviderError:
            raise
        except Exception as exc:
            raise GenerationProviderError("Generation failed unexpectedly.") from exc
        if not isinstance(generation_result, GenerationResult):
            raise GenerationProviderError("Generation provider returned an invalid result type.")
        generation_finished = self._clock()
        generation_latency_ms = _elapsed_ms(generation_started, generation_finished)

        answer_text = generation_result.text.strip()
        warnings: list[str] = []
        if answer_text == INSUFFICIENT_EVIDENCE_ANSWER:
            citations = []
            warnings.append("The generation provider reported insufficient evidence.")
        else:
            citations = validate_citations(answer_text, retrieval_results)

        total_latency_ms = _elapsed_ms(total_started, self._clock())
        answer = RagAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieval_results=list(retrieval_results),
            model_name=self._generation_provider.model_name,
            prompt_version=self._prompt_version,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            usage=(dict(generation_result.usage) if generation_result.usage is not None else None),
            warnings=warnings,
        )
        _log_completion(answer)
        return answer

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> RagAnswer:
        """Alias for :meth:`ask` for callers that use answer-oriented naming."""

        return self.ask(question, top_k=top_k)


RagPipeline = DirectRagPipeline
DirectRAGPipeline = DirectRagPipeline
RAGPipeline = DirectRagPipeline


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer.")


def _validate_temperature(temperature: float) -> None:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise GenerationProviderError("temperature must be a finite number between 0 and 2.")


def _elapsed_ms(started: float, finished: float) -> float:
    return max(0.0, (finished - started) * 1000.0)


def _log_completion(answer: RagAnswer) -> None:
    logger.info(
        "direct_rag_completed",
        extra={
            "citation_count": len(answer.citations),
            "generation_latency_ms": answer.generation_latency_ms,
            "retrieval_latency_ms": answer.retrieval_latency_ms,
            "total_latency_ms": answer.total_latency_ms,
        },
    )
