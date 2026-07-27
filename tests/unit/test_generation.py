"""Unit tests for deterministic and injected generation providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_agent_eval_toolkit.exceptions import GenerationProviderError
from rag_agent_eval_toolkit.generation import (
    FakeGenerationProvider,
    GenerationProvider,
    OpenAIGenerationProvider,
)
from rag_agent_eval_toolkit.models import RetrievalResult
from rag_agent_eval_toolkit.rag import build_rag_prompt


def _evidence() -> RetrievalResult:
    return RetrievalResult(
        query="What should I inspect?",
        rank=1,
        score=0.9,
        chunk_id="charging-001",
        source_id="charging-source",
        source_name="charging.md",
        text="Inspect the connector before fast charging. Confirm compatibility.",
        page_number=2,
        citation_label="[charging.md#charging-001]",
    )


class _Client:
    def __init__(self, responses: object) -> None:
        self.responses = responses


class _RecordingResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def test_fake_provider_is_deterministic_and_cites_first_evidence() -> None:
    prompt = build_rag_prompt("What should I inspect?", [_evidence()])
    provider = FakeGenerationProvider()

    first = provider.generate(prompt)
    second = provider.generate(prompt)

    assert first == second
    assert first.text == ("Inspect the connector before fast charging. [charging.md#charging-001]")
    assert first.usage is None
    assert provider.prompts == (prompt, prompt)
    assert isinstance(provider, GenerationProvider)


def test_fake_provider_selects_relevant_complete_sentence() -> None:
    evidence = RetrievalResult(
        query="What should I check before connecting to a charger?",
        rank=1,
        score=0.8,
        chunk_id="charging-002",
        source_id="charging-source",
        source_name="charging.md",
        text=(
            "ld inspect an outlet used for charging. "
            "Before every charge, inspect the cable and connector for damage. "
            "Store the vehicle between 50 and 60 percent."
        ),
        page_number=None,
        citation_label="[charging.md#charging-002]",
    )
    prompt = build_rag_prompt(
        "What should I check before connecting to a charger?",
        [evidence],
    )

    result = FakeGenerationProvider().generate(prompt)

    assert result.text == (
        "Before every charge, inspect the cable and connector for damage. "
        "[charging.md#charging-002]"
    )


def test_fake_provider_supports_an_injected_fixed_response() -> None:
    provider = FakeGenerationProvider(
        "Use the supplied guidance. [charging.md#charging-001]",
        model_name="fixture-generator",
    )

    result = provider.generate("A complete test prompt")

    assert result.text == "Use the supplied guidance. [charging.md#charging-001]"
    assert provider.model_name == "fixture-generator"


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: FakeGenerationProvider(response=" "), "response"),
        (lambda: FakeGenerationProvider(model_name=" "), "model name"),
        (lambda: FakeGenerationProvider().generate(" "), "prompt"),
        (
            lambda: FakeGenerationProvider().generate("prompt", temperature=True),
            "temperature",
        ),
        (
            lambda: FakeGenerationProvider().generate("prompt", temperature=2.1),
            "temperature",
        ),
    ],
)
def test_fake_provider_rejects_invalid_configuration(
    operation: object,
    message: str,
) -> None:
    assert callable(operation)
    with pytest.raises(GenerationProviderError, match=message):
        operation()


def test_openai_provider_uses_injected_responses_client_and_usage() -> None:
    endpoint = _RecordingResponses(
        SimpleNamespace(
            output_text="Inspect it. [charging.md#charging-001]",
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=8,
                total_tokens=28,
            ),
        )
    )
    provider = OpenAIGenerationProvider(
        "gpt-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
        max_output_tokens=80,
    )

    result = provider.generate("safe prompt", temperature=0)

    assert result.text == "Inspect it. [charging.md#charging-001]"
    assert result.usage == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }
    assert endpoint.calls == [
        {
            "input": "safe prompt",
            "max_output_tokens": 80,
            "model": "gpt-test",
            "temperature": 0.0,
        }
    ]


def test_openai_provider_reads_structured_output_when_output_text_is_absent() -> None:
    endpoint = _RecordingResponses(
        {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Cited answer. [charging.md#charging-001]",
                        }
                    ]
                }
            ]
        }
    )
    provider = OpenAIGenerationProvider(
        "gpt-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
    )

    result = provider.generate("safe prompt")

    assert result.text == "Cited answer. [charging.md#charging-001]"
    assert result.usage is None


def test_openai_provider_retries_only_transient_failures() -> None:
    class APITimeoutError(Exception):
        pass

    class _FlakyResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise APITimeoutError("prompt body must not leak")
            return SimpleNamespace(output_text="Recovered answer.", usage=None)

    endpoint = _FlakyResponses()
    sleeps: list[float] = []
    provider = OpenAIGenerationProvider(
        "gpt-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
        max_retries=2,
        retry_base_seconds=0.5,
        sleeper=sleeps.append,
    )

    result = provider.generate("safe prompt")

    assert result.text == "Recovered answer."
    assert endpoint.calls == 2
    assert sleeps == [0.5]


def test_openai_provider_does_not_retry_invalid_responses() -> None:
    endpoint = _RecordingResponses(SimpleNamespace(output_text="", output=[]))
    provider = OpenAIGenerationProvider(
        "gpt-test",
        client=_Client(endpoint),  # type: ignore[arg-type]
        max_retries=2,
        sleeper=lambda _: pytest.fail("invalid responses must not be retried"),
    )

    with pytest.raises(GenerationProviderError, match="output text"):
        provider.generate("safe prompt")

    assert len(endpoint.calls) == 1


def test_openai_provider_sanitises_non_transient_errors() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    class _FailingResponses:
        def create(self, **kwargs: object) -> object:
            raise AuthenticationError("secret prompt body")

    provider = OpenAIGenerationProvider(
        "gpt-test",
        client=_Client(_FailingResponses()),  # type: ignore[arg-type]
        max_retries=2,
        sleeper=lambda _: pytest.fail("authentication errors must not be retried"),
    )

    with pytest.raises(GenerationProviderError) as exc_info:
        provider.generate("safe prompt")

    assert "AuthenticationError" in str(exc_info.value)
    assert "secret prompt body" not in str(exc_info.value)
