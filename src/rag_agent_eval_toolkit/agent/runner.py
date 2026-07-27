"""Constrained deterministic and OpenAI tool-calling agent runners."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol, cast

from rag_agent_eval_toolkit.agent.tools import (
    GET_SOURCE_METADATA_TOOL_NAME,
    SEARCH_CORPUS_TOOL_NAME,
    AgentToolResult,
    AgentTools,
    SearchCorpusResult,
)
from rag_agent_eval_toolkit.exceptions import (
    GenerationProviderError,
    RetrievalError,
)
from rag_agent_eval_toolkit.generation.base import INSUFFICIENT_EVIDENCE_ANSWER
from rag_agent_eval_toolkit.models import (
    AgentRunResult,
    RetrievalResult,
    ToolCall,
)
from rag_agent_eval_toolkit.rag.citations import validate_citations

logger = logging.getLogger(__name__)

DEFAULT_AGENT_MODEL = "fake-agent-v1"
DEFAULT_AGENT_TOP_K = 3
DEFAULT_MAX_TOOL_ROUNDS = 4

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_LEADING_MARKDOWN_HEADING = re.compile(r"^(?:#{1,6}\s+[^\n]+\n+)+")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_ANSWER_SELECTION_STOP_WORDS = frozenset(
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
_ANSWER_SELECTION_ALIASES = {
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


class _ResponsesResource(Protocol):
    def create(self, **kwargs: object) -> object: ...


class OpenAIAgentClient(Protocol):
    """Small official-client surface required by the agent runner."""

    responses: _ResponsesResource


@dataclass(frozen=True, slots=True)
class _FunctionCall:
    name: str
    arguments: Mapping[str, object]
    call_id: str


@dataclass(frozen=True, slots=True)
class _ExecutedToolCall:
    output_json: str
    trace: ToolCall
    result: AgentToolResult | None
    error: Exception | None


class DeterministicAgentRunner:
    """Run a fixed, offline two-tool plan for reproducible tests and demos."""

    def __init__(
        self,
        tools: AgentTools,
        *,
        model_name: str = DEFAULT_AGENT_MODEL,
        default_top_k: int = DEFAULT_AGENT_TOP_K,
        clock: Callable[[], float] = perf_counter,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        """Configure an offline agent with injected tools and clocks."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise GenerationProviderError("Agent model name must not be empty.")
        _validate_top_k(default_top_k)
        self._tools = tools
        self._model_name = model_name.strip()
        self._default_top_k = default_top_k
        self._clock = clock
        self._utc_now = utc_now or (lambda: datetime.now(UTC))

    @property
    def model_name(self) -> str:
        """Return the stable offline agent model identifier."""

        return self._model_name

    @property
    def tool_names(self) -> tuple[str, str]:
        """Return the complete tool allowlist available to this runner."""

        return self._tools.tool_names

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> AgentRunResult:
        """Search, inspect source provenance, and return a cited deterministic answer."""

        validated_question = _validate_question(question)
        requested_top_k = self._default_top_k if top_k is None else top_k
        _validate_top_k(requested_top_k)

        total_started = self._clock()
        tool_calls: list[ToolCall] = []
        warnings: list[str] = []

        search_execution = _execute_tool(
            self._tools,
            SEARCH_CORPUS_TOOL_NAME,
            {"query": validated_question, "top_k": requested_top_k},
            clock=self._clock,
            utc_now=self._utc_now,
        )
        tool_calls.append(search_execution.trace)
        if search_execution.error is not None:
            raise RetrievalError(
                f"Agent search_corpus call failed: {search_execution.trace.error}"
            ) from search_execution.error
        if not isinstance(search_execution.result, SearchCorpusResult):
            raise RetrievalError("Agent search_corpus returned an invalid result type.")

        search_result = search_execution.result
        evidence = search_result.as_retrieval_results()
        if not evidence:
            result = AgentRunResult(
                question=validated_question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                tool_calls=tool_calls,
                model_name=self._model_name,
                total_latency_ms=_elapsed_ms(total_started, self._clock()),
                usage=None,
                warnings=["No evidence was retrieved; the offline agent abstained."],
            )
            _log_completion(result)
            return result

        seen_sources: set[str] = set()
        for retrieval_result in evidence:
            if retrieval_result.source_id in seen_sources:
                continue
            seen_sources.add(retrieval_result.source_id)
            metadata_execution = _execute_tool(
                self._tools,
                GET_SOURCE_METADATA_TOOL_NAME,
                {"source_id": retrieval_result.source_id},
                clock=self._clock,
                utc_now=self._utc_now,
            )
            tool_calls.append(metadata_execution.trace)
            if metadata_execution.error is not None:
                warnings.append(
                    "Source metadata lookup failed for "
                    f"{retrieval_result.source_id!r}; retrieved evidence was retained."
                )

        answer_text = _deterministic_answer(search_result)
        citations = validate_citations(answer_text, evidence)
        result = AgentRunResult(
            question=validated_question,
            answer=answer_text,
            citations=citations,
            tool_calls=tool_calls,
            model_name=self._model_name,
            total_latency_ms=_elapsed_ms(total_started, self._clock()),
            usage=None,
            warnings=warnings,
        )
        _log_completion(result)
        return result

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> AgentRunResult:
        """Alias for :meth:`run` for answer-oriented callers."""

        return self.run(question, top_k=top_k)


DeterministicOfflineAgentRunner = DeterministicAgentRunner


class OpenAIToolCallingAgentRunner:
    """Run a constrained agent through the official OpenAI Responses API."""

    def __init__(
        self,
        tools: AgentTools,
        model: str,
        *,
        client: OpenAIAgentClient | None = None,
        api_key: str | None = None,
        default_top_k: int = DEFAULT_AGENT_TOP_K,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_output_tokens: int = 600,
        timeout_seconds: float = 30.0,
        clock: Callable[[], float] = perf_counter,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        """Configure an injectable, output-bounded OpenAI tool-calling runner."""

        if not isinstance(model, str) or not model.strip():
            raise GenerationProviderError("OpenAI agent model must not be empty.")
        _validate_top_k(default_top_k)
        if (
            isinstance(max_tool_rounds, bool)
            or not isinstance(max_tool_rounds, int)
            or max_tool_rounds <= 0
        ):
            raise GenerationProviderError("max_tool_rounds must be a positive integer.")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise GenerationProviderError("max_output_tokens must be a positive integer.")
        if timeout_seconds <= 0:
            raise GenerationProviderError("timeout_seconds must be greater than 0.")

        self._tools = tools
        self._model = model.strip()
        self._default_top_k = default_top_k
        self._max_tool_rounds = max_tool_rounds
        self._max_output_tokens = max_output_tokens
        self._clock = clock
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._client = client or self._create_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _create_client(
        *,
        api_key: str | None,
        timeout_seconds: float,
    ) -> OpenAIAgentClient:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency setup
            raise GenerationProviderError(
                "The 'openai' package is required for the OpenAI agent."
            ) from exc

        return cast(
            OpenAIAgentClient,
            OpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=timeout_seconds,
            ),
        )

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model identifier."""

        return self._model

    @property
    def tool_names(self) -> tuple[str, str]:
        """Return the complete tool allowlist available to this runner."""

        return self._tools.tool_names

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> AgentRunResult:
        """Execute bounded function-call rounds and validate the final citations."""

        validated_question = _validate_question(question)
        requested_top_k = self._default_top_k if top_k is None else top_k
        _validate_top_k(requested_top_k)
        total_started = self._clock()
        instructions = _agent_instructions(requested_top_k)
        definitions = self._tools.openai_tool_definitions()

        response = self._create_response(
            input=validated_question,
            instructions=instructions,
            max_output_tokens=self._max_output_tokens,
            model=self._model,
            tools=definitions,
        )
        usage = _usage_from_response(response)
        tool_calls: list[ToolCall] = []
        evidence_by_label: dict[str, RetrievalResult] = {}
        successful_search = False
        warnings: list[str] = []

        for round_index in range(self._max_tool_rounds + 1):
            function_calls = _function_calls_from_response(response)
            if not function_calls:
                if not successful_search:
                    raise GenerationProviderError(
                        "OpenAI agent completed without a successful search_corpus call."
                    )
                answer_text = _text_from_response(response)
                evidence = list(evidence_by_label.values())
                if answer_text == INSUFFICIENT_EVIDENCE_ANSWER:
                    citations = []
                    warnings.append("The OpenAI agent reported insufficient evidence.")
                else:
                    citations = validate_citations(answer_text, evidence)

                result = AgentRunResult(
                    question=validated_question,
                    answer=answer_text,
                    citations=citations,
                    tool_calls=tool_calls,
                    model_name=self._model,
                    total_latency_ms=_elapsed_ms(total_started, self._clock()),
                    usage=usage,
                    warnings=warnings,
                )
                _log_completion(result)
                return result

            if round_index >= self._max_tool_rounds:
                raise GenerationProviderError(
                    "OpenAI agent exceeded the configured tool-call round limit."
                )

            response_id = _required_text_field(response, "id", context="response")
            function_outputs: list[dict[str, object]] = []
            for function_call in function_calls:
                if function_call.name not in self._tools.tool_names:
                    raise GenerationProviderError(
                        "OpenAI agent requested an unsupported tool; no tool was executed."
                    )
                execution = _execute_tool(
                    self._tools,
                    function_call.name,
                    function_call.arguments,
                    clock=self._clock,
                    utc_now=self._utc_now,
                )
                tool_calls.append(execution.trace)
                if execution.error is not None:
                    warnings.append(f"{function_call.name} returned a tool error to the agent.")
                elif isinstance(execution.result, SearchCorpusResult):
                    successful_search = True
                    for item in execution.result.as_retrieval_results():
                        evidence_by_label.setdefault(item.citation_label, item)
                function_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": execution.output_json,
                    }
                )

            response = self._create_response(
                input=function_outputs,
                instructions=instructions,
                max_output_tokens=self._max_output_tokens,
                model=self._model,
                previous_response_id=response_id,
                tools=definitions,
            )
            usage = _merge_usage(usage, _usage_from_response(response))

        raise GenerationProviderError("OpenAI agent loop ended unexpectedly.")

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> AgentRunResult:
        """Alias for :meth:`run` for answer-oriented callers."""

        return self.run(question, top_k=top_k)

    def _create_response(self, **request: object) -> object:
        try:
            return self._client.responses.create(**request)
        except GenerationProviderError:
            raise
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            status_suffix = f", HTTP {status_code}" if isinstance(status_code, int) else ""
            raise GenerationProviderError(
                f"OpenAI agent request failed ({exc.__class__.__name__}{status_suffix})."
            ) from exc


OpenAIAgentRunner = OpenAIToolCallingAgentRunner


def _execute_tool(
    tools: AgentTools,
    tool_name: str,
    arguments: Mapping[str, object],
    *,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
) -> _ExecutedToolCall:
    started_at = _format_timestamp(utc_now())
    started = clock()
    safe_arguments = _safe_trace_arguments(tool_name, arguments)
    try:
        result = tools.invoke(tool_name, arguments)
    except Exception as exc:
        error = _safe_tool_error(exc)
        trace = ToolCall(
            tool_name=tool_name,
            arguments=safe_arguments,
            started_at=started_at,
            duration_ms=_elapsed_ms(started, clock()),
            result_summary="Tool call failed; no result content was retained.",
            error=error,
        )
        return _ExecutedToolCall(
            output_json=json.dumps(
                {"ok": False, "error": error},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            trace=trace,
            result=None,
            error=exc,
        )

    trace = ToolCall(
        tool_name=tool_name,
        arguments=safe_arguments,
        started_at=started_at,
        duration_ms=_elapsed_ms(started, clock()),
        result_summary=_result_summary(result),
        error=None,
    )
    return _ExecutedToolCall(
        output_json=result.to_json(),
        trace=trace,
        result=result,
        error=None,
    )


def _agent_instructions(default_top_k: int) -> str:
    return (
        "Answer using only evidence returned by the provided tools. "
        "You have exactly two tools: search_corpus and get_source_metadata. "
        "Call search_corpus before answering and use top_k "
        f"{default_top_k} unless the question clearly needs fewer results. "
        "You may use get_source_metadata only with a source_id returned by search_corpus. "
        "For supported claims, copy citation_label values verbatim into the answer. "
        "Never invent or alter citations. If the evidence is insufficient, return exactly: "
        f"{INSUFFICIENT_EVIDENCE_ANSWER}"
    )


def _deterministic_answer(search_result: SearchCorpusResult) -> str:
    question_terms = _answer_selection_terms(search_result.query)
    candidates: list[tuple[tuple[int, int, int, int], str, str]] = []
    for match_index, match in enumerate(search_result.matches):
        for sentence_index, raw_sentence in enumerate(
            _SENTENCE_BOUNDARY.split(match.excerpt.strip())
        ):
            sentence = _LEADING_MARKDOWN_HEADING.sub("", raw_sentence.strip()).strip()
            if not sentence:
                continue
            score = (
                len(question_terms.intersection(_answer_selection_terms(sentence))),
                int(sentence[0].isupper() or sentence[0].isdigit()),
                -match_index,
                -sentence_index,
            )
            candidates.append((score, sentence, match.citation_label))
    if not candidates:
        return INSUFFICIENT_EVIDENCE_ANSWER
    _, sentence, citation_label = max(candidates, key=lambda candidate: candidate[0])
    return f"{sentence} {citation_label}"


def _answer_selection_terms(text: str) -> set[str]:
    return {
        _ANSWER_SELECTION_ALIASES.get(token, token)
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _ANSWER_SELECTION_STOP_WORDS
    }


def _function_calls_from_response(response: object) -> list[_FunctionCall]:
    output = _read_field(response, "output", default=())
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise GenerationProviderError("OpenAI agent response contained invalid output items.")

    calls: list[_FunctionCall] = []
    for item in output:
        if _read_field(item, "type", default=None) != "function_call":
            continue
        name = _required_text_field(item, "name", context="function call")
        call_id = _required_text_field(item, "call_id", context="function call")
        raw_arguments = _read_field(item, "arguments", default=None)
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
        else:
            parsed_arguments = raw_arguments
        if not isinstance(parsed_arguments, Mapping):
            parsed_arguments = {}
        calls.append(
            _FunctionCall(
                name=name,
                arguments={str(key): value for key, value in parsed_arguments.items()},
                call_id=call_id,
            )
        )
    return calls


def _text_from_response(response: object) -> str:
    output_text = _read_field(response, "output_text", default=None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _read_field(response, "output", default=())
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise GenerationProviderError("OpenAI agent response did not contain final text.")
    text_parts: list[str] = []
    for item in output:
        content = _read_field(item, "content", default=())
        if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
            continue
        for part in content:
            text_value = _read_field(part, "text", default=None)
            if isinstance(text_value, str) and text_value.strip():
                text_parts.append(text_value.strip())
    answer = "\n".join(text_parts).strip()
    if not answer:
        raise GenerationProviderError("OpenAI agent response did not contain final text.")
    return answer


_MISSING = object()


def _read_field(
    value: object,
    name: str,
    *,
    default: object = _MISSING,
) -> object:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise GenerationProviderError(f"OpenAI agent response was missing the {name!r} field.")


def _required_text_field(value: object, name: str, *, context: str) -> str:
    field = _read_field(value, name, default=None)
    if not isinstance(field, str) or not field.strip():
        raise GenerationProviderError(
            f"OpenAI agent {context} was missing a non-empty {name!r} field."
        )
    return field.strip()


def _usage_from_response(response: object) -> dict[str, int] | None:
    usage = _read_field(response, "usage", default=None)
    if usage is None:
        return None
    parsed: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = _read_field(usage, name, default=None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            parsed[name] = value
    return parsed or None


def _merge_usage(
    left: dict[str, int] | None,
    right: dict[str, int] | None,
) -> dict[str, int] | None:
    if left is None:
        return dict(right) if right is not None else None
    if right is None:
        return dict(left)
    keys = set(left) | set(right)
    return {key: left.get(key, 0) + right.get(key, 0) for key in keys}


def _safe_trace_arguments(
    tool_name: str,
    arguments: Mapping[str, object],
) -> dict[str, object]:
    if tool_name == SEARCH_CORPUS_TOOL_NAME:
        query = arguments.get("query")
        top_k = arguments.get("top_k")
        return {
            "query": _truncate_trace_text(query),
            "top_k": top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else None,
        }
    if tool_name == GET_SOURCE_METADATA_TOOL_NAME:
        return {"source_id": _truncate_trace_text(arguments.get("source_id"))}
    return {}


def _truncate_trace_text(value: object, *, limit: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."


def _safe_tool_error(exc: Exception) -> str:
    if isinstance(exc, (RetrievalError, GenerationProviderError)):
        return f"{exc.__class__.__name__}: {exc}"
    return f"{exc.__class__.__name__}: tool execution failed unexpectedly."


def _result_summary(result: AgentToolResult) -> str:
    if isinstance(result, SearchCorpusResult):
        return f"Returned {len(result.matches)} ranked evidence result(s)."
    return f"Returned allowlisted metadata for source_id {result.source_id!r}."


def _validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise RetrievalError("Agent question must contain non-whitespace text.")
    return question.strip()


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer.")


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _elapsed_ms(started: float, finished: float) -> float:
    return max(0.0, (finished - started) * 1000.0)


def _log_completion(result: AgentRunResult) -> None:
    logger.info(
        "agent_run_completed",
        extra={
            "citation_count": len(result.citations),
            "tool_call_count": len(result.tool_calls),
            "total_latency_ms": result.total_latency_ms,
        },
    )
