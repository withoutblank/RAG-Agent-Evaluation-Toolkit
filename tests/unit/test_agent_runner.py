"""Unit tests for deterministic and injectable OpenAI agent runners."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rag_agent_eval_toolkit.agent import (
    GET_SOURCE_METADATA_TOOL_NAME,
    SEARCH_CORPUS_TOOL_NAME,
    AgentTools,
    DeterministicAgentRunner,
    OpenAIToolCallingAgentRunner,
)
from rag_agent_eval_toolkit.exceptions import (
    CitationValidationError,
    GenerationProviderError,
    RetrievalError,
)
from rag_agent_eval_toolkit.generation import INSUFFICIENT_EVIDENCE_ANSWER
from rag_agent_eval_toolkit.models import RetrievalResult, SourceDocument, ToolCall


class _StubRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


class _FakeResponses:
    def __init__(
        self,
        responses: list[object],
        *,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses)
        self._error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("The fake client received an unexpected request.")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _source() -> SourceDocument:
    return SourceDocument(
        source_id="charging-source",
        source_name="charging.md",
        relative_path="fictional_ev_support/charging.md",
        file_type=".md",
        text="Inspect the connector before charging.",
        content_sha256="b" * 64,
        page_count=None,
        metadata={},
    )


def _result() -> RetrievalResult:
    return RetrievalResult(
        query="What should I inspect?",
        rank=1,
        score=0.9,
        chunk_id="charging-001",
        source_id="charging-source",
        source_name="charging.md",
        text="Inspect the connector before charging. Follow the safety instructions.",
        page_number=None,
        citation_label="[charging.md#charging-001]",
    )


def _tools(
    *,
    results: list[RetrievalResult] | None = None,
    include_source: bool = True,
    retriever_error: Exception | None = None,
) -> tuple[AgentTools, _StubRetriever]:
    retriever = _StubRetriever(
        [_result()] if results is None else results,
        error=retriever_error,
    )
    source = _source()
    sources = {source.source_id: source} if include_source else {}
    return AgentTools(retriever, sources), retriever


def _fixed_now() -> datetime:
    return datetime(2026, 7, 27, 3, 4, 5, tzinfo=UTC)


def _function_response(
    *,
    response_id: str,
    name: str,
    arguments: dict[str, object],
    call_id: str,
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "id": response_id,
        "output": [
            {
                "type": "function_call",
                "name": name,
                "arguments": json.dumps(arguments),
                "call_id": call_id,
            }
        ],
        "usage": usage,
    }


def _final_response(
    text: str,
    *,
    response_id: str = "response-final",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "id": response_id,
        "output": [],
        "output_text": text,
        "usage": usage,
    }


def test_deterministic_runner_uses_both_tools_and_captures_safe_traces() -> None:
    tools, retriever = _tools()
    runner = DeterministicAgentRunner(
        tools,
        clock=lambda: 1.0,
        utc_now=_fixed_now,
    )

    result = runner.run("  What should I inspect?  ", top_k=1)

    assert retriever.calls == [("What should I inspect?", 1)]
    assert result.question == "What should I inspect?"
    assert result.answer == ("Inspect the connector before charging. [charging.md#charging-001]")
    assert [citation.citation_label for citation in result.citations] == [
        "[charging.md#charging-001]"
    ]
    assert [call.tool_name for call in result.tool_calls] == [
        SEARCH_CORPUS_TOOL_NAME,
        GET_SOURCE_METADATA_TOOL_NAME,
    ]
    assert all(isinstance(call, ToolCall) for call in result.tool_calls)
    assert all(call.error is None for call in result.tool_calls)
    assert all(call.started_at == "2026-07-27T03:04:05Z" for call in result.tool_calls)
    assert all(call.duration_ms == 0.0 for call in result.tool_calls)
    assert all(
        "Follow the safety instructions" not in call.result_summary for call in result.tool_calls
    )
    assert result.model_name == "fake-agent-v1"
    assert result.usage is None
    assert result.warnings == []


def test_deterministic_runner_abstains_without_evidence() -> None:
    tools, _ = _tools(results=[])
    runner = DeterministicAgentRunner(tools)

    result = runner.run("Question with no evidence")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []
    assert [call.tool_name for call in result.tool_calls] == [SEARCH_CORPUS_TOOL_NAME]
    assert result.warnings == ["No evidence was retrieved; the offline agent abstained."]


def test_deterministic_runner_skips_a_truncated_leading_fragment() -> None:
    result = _result()
    truncated = RetrievalResult(
        query=result.query,
        rank=result.rank,
        score=result.score,
        chunk_id=result.chunk_id,
        source_id=result.source_id,
        source_name=result.source_name,
        text=(
            "ld inspect an outlet used for charging.\n\n"
            "## Before connecting\n\n"
            "Before every charge, inspect the cable and connector for damage."
        ),
        page_number=result.page_number,
        citation_label=result.citation_label,
    )
    tools, _ = _tools(results=[truncated])
    runner = DeterministicAgentRunner(tools)

    answer = runner.run("What should I check before connecting to a charger?")

    assert answer.answer == (
        "Before every charge, inspect the cable and connector for damage. "
        "[charging.md#charging-001]"
    )


def test_deterministic_runner_records_optional_metadata_failure() -> None:
    tools, _ = _tools(include_source=False)
    runner = DeterministicAgentRunner(tools)

    result = runner.run("What should I inspect?")

    assert result.citations
    assert result.tool_calls[1].tool_name == GET_SOURCE_METADATA_TOOL_NAME
    assert result.tool_calls[1].error is not None
    assert "No source metadata" in result.tool_calls[1].error
    assert result.warnings == [
        "Source metadata lookup failed for 'charging-source'; retrieved evidence was retained."
    ]


def test_deterministic_runner_rejects_invalid_inputs_and_sanitises_search_failures() -> None:
    tools, _ = _tools(retriever_error=RuntimeError("private vector database detail"))
    runner = DeterministicAgentRunner(tools)

    with pytest.raises(RetrievalError, match="non-whitespace"):
        runner.run(" ")
    with pytest.raises(RetrievalError, match="top_k"):
        runner.run("question", top_k=False)
    with pytest.raises(RetrievalError, match="search_corpus") as exc_info:
        runner.run("question")

    assert "private vector database detail" not in str(exc_info.value)


def test_openai_runner_executes_bounded_tool_loop_and_validates_citations() -> None:
    tools, retriever = _tools()
    fake_responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "What should I inspect?", "top_k": 1},
                call_id="search-call",
                usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            _function_response(
                response_id="response-2",
                name=GET_SOURCE_METADATA_TOOL_NAME,
                arguments={"source_id": "charging-source"},
                call_id="metadata-call",
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            ),
            _final_response(
                "Inspect the connector. [charging.md#charging-001]",
                usage={"input_tokens": 4, "output_tokens": 4, "total_tokens": 8},
            ),
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(fake_responses),
        clock=lambda: 2.0,
        utc_now=_fixed_now,
    )

    result = runner.run("What should I inspect?", top_k=1)

    assert retriever.calls == [("What should I inspect?", 1)]
    assert result.answer == "Inspect the connector. [charging.md#charging-001]"
    assert [citation.chunk_id for citation in result.citations] == ["charging-001"]
    assert [call.tool_name for call in result.tool_calls] == [
        SEARCH_CORPUS_TOOL_NAME,
        GET_SOURCE_METADATA_TOOL_NAME,
    ]
    assert result.usage == {
        "input_tokens": 17,
        "output_tokens": 8,
        "total_tokens": 25,
    }
    assert result.model_name == "gpt-test"
    first_request = fake_responses.calls[0]
    assert [tool["name"] for tool in first_request["tools"]] == [  # type: ignore[index,union-attr]
        SEARCH_CORPUS_TOOL_NAME,
        GET_SOURCE_METADATA_TOOL_NAME,
    ]
    assert "shell" not in json.dumps(first_request["tools"])
    assert fake_responses.calls[1]["previous_response_id"] == "response-1"
    assert fake_responses.calls[2]["previous_response_id"] == "response-2"
    search_output = fake_responses.calls[1]["input"][0]["output"]  # type: ignore[index]
    assert "citation_label" in search_output
    assert "Full source text must never" not in search_output


def test_openai_runner_returns_tool_validation_error_to_model_and_can_recover() -> None:
    tools, _ = _tools()
    fake_responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 0},
                call_id="bad-search",
            ),
            _function_response(
                response_id="response-2",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="good-search",
            ),
            _final_response("Inspect it. [charging.md#charging-001]"),
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(fake_responses),
    )

    result = runner.run("question")

    assert len(result.tool_calls) == 2
    assert "positive integer" in (result.tool_calls[0].error or "")
    assert result.tool_calls[1].error is None
    error_payload = json.loads(
        fake_responses.calls[1]["input"][0]["output"]  # type: ignore[index]
    )
    assert error_payload["ok"] is False
    assert result.warnings == ["search_corpus returned a tool error to the agent."]


@pytest.mark.parametrize(
    "final_text",
    [
        "Invented answer. [unknown.md#unknown-001]",
        "An answer without a citation.",
    ],
)
def test_openai_runner_rejects_invalid_final_citations(final_text: str) -> None:
    tools, _ = _tools()
    responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="search-call",
            ),
            _final_response(final_text),
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(responses),
    )

    with pytest.raises(CitationValidationError):
        runner.run("question")


def test_openai_runner_allows_exact_abstention_after_empty_search() -> None:
    tools, _ = _tools(results=[])
    responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="search-call",
            ),
            _final_response(INSUFFICIENT_EVIDENCE_ANSWER),
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(responses),
    )

    result = runner.run("question")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []
    assert result.warnings == ["The OpenAI agent reported insufficient evidence."]


def test_openai_runner_rejects_cited_answer_after_empty_search() -> None:
    tools, _ = _tools(results=[])
    responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="search-call",
            ),
            _final_response("Invented answer. [charging.md#charging-001]"),
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(responses),
    )

    with pytest.raises(CitationValidationError, match="not retrieved"):
        runner.run("question")


def test_openai_runner_never_executes_an_unlisted_tool() -> None:
    tools, retriever = _tools()
    responses = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name="run_shell",
                arguments={"command": "whoami"},
                call_id="unsafe-call",
            )
        ]
    )
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(responses),
    )

    with pytest.raises(GenerationProviderError, match="unsupported tool"):
        runner.run("question")

    assert retriever.calls == []


def test_openai_runner_requires_search_and_enforces_round_limit() -> None:
    tools, _ = _tools()
    no_search = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(
            _FakeResponses([_final_response("Unsupported answer [charging.md#charging-001]")])
        ),
    )

    with pytest.raises(GenerationProviderError, match="without a successful"):
        no_search.run("question")

    repeated_calls = _FakeResponses(
        [
            _function_response(
                response_id="response-1",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="search-1",
            ),
            _function_response(
                response_id="response-2",
                name=SEARCH_CORPUS_TOOL_NAME,
                arguments={"query": "question", "top_k": 1},
                call_id="search-2",
            ),
        ]
    )
    limited = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(repeated_calls),
        max_tool_rounds=1,
    )

    with pytest.raises(GenerationProviderError, match="round limit"):
        limited.run("question")


def test_openai_client_errors_are_sanitised_and_no_live_client_is_needed() -> None:
    tools, _ = _tools()
    fake_responses = _FakeResponses([], error=RuntimeError("secret request detail"))
    runner = OpenAIToolCallingAgentRunner(
        tools,
        "gpt-test",
        client=_FakeClient(fake_responses),
    )

    with pytest.raises(GenerationProviderError, match="OpenAI agent request failed") as exc_info:
        runner.run("question")

    assert "secret request detail" not in str(exc_info.value)
    assert len(fake_responses.calls) == 1
