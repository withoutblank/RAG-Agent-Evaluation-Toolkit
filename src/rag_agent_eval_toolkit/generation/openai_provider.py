"""Official OpenAI generation adapter with injectable client support."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from rag_agent_eval_toolkit.exceptions import GenerationProviderError
from rag_agent_eval_toolkit.generation.base import (
    GenerationResult,
    _validate_generation_request,
)


class _ResponsesResource(Protocol):
    def create(self, **kwargs: object) -> object: ...


class OpenAIGenerationClient(Protocol):
    """Small client surface required from the official OpenAI SDK."""

    responses: _ResponsesResource


class OpenAIGenerationProvider:
    """Generate text through the official OpenAI Responses API.

    A compatible client can be injected for tests. When no client is supplied,
    the official SDK is imported lazily with its built-in retries disabled so
    this adapter's bounded retry behavior remains explicit.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: OpenAIGenerationClient | None = None,
        api_key: str | None = None,
        max_output_tokens: int = 600,
        max_retries: int = 2,
        retry_base_seconds: float = 0.25,
        timeout_seconds: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise an injectable, output-bounded OpenAI adapter."""

        if not isinstance(model, str) or not model.strip():
            raise GenerationProviderError("OpenAI generation model must not be empty.")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise GenerationProviderError("max_output_tokens must be a positive integer.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise GenerationProviderError("max_retries must be a non-negative integer.")
        if retry_base_seconds < 0:
            raise GenerationProviderError("retry_base_seconds must be non-negative.")
        if timeout_seconds <= 0:
            raise GenerationProviderError("timeout_seconds must be greater than 0.")

        self._model = model.strip()
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._sleeper = sleeper
        self._client = client or self._create_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _create_client(
        *,
        api_key: str | None,
        timeout_seconds: float,
    ) -> OpenAIGenerationClient:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is runtime setup
            raise GenerationProviderError(
                "The 'openai' package is required for OpenAI generation."
            ) from exc

        return cast(
            OpenAIGenerationClient,
            OpenAI(
                api_key=api_key,
                max_retries=0,
                timeout=timeout_seconds,
            ),
        )

    @property
    def model_name(self) -> str:
        """Return the configured OpenAI model name."""

        return self._model

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate one response with bounded transient-error retries."""

        validated_prompt = _validate_generation_request(prompt, temperature)
        request: dict[str, object] = {
            "input": validated_prompt,
            "max_output_tokens": self._max_output_tokens,
            "model": self._model,
            "temperature": float(temperature),
        }

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.responses.create(**request)
                return GenerationResult(
                    text=_text_from_response(response),
                    usage=_usage_from_response(response),
                )
            except GenerationProviderError:
                raise
            except Exception as exc:
                should_retry = attempt < self._max_retries and _is_transient_error(exc)
                if not should_retry:
                    raise _sanitised_provider_error(
                        exc,
                        attempts=attempt + 1,
                    ) from exc
                self._sleeper(self._retry_base_seconds * (2**attempt))

        raise GenerationProviderError("OpenAI generation retry loop ended unexpectedly.")


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
    raise GenerationProviderError(f"OpenAI generation response was missing the '{name}' field.")


def _text_from_response(response: object) -> str:
    output_text = _read_field(response, "output_text", default=None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _read_field(response, "output", default=None)
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        raise GenerationProviderError("OpenAI generation response did not contain output text.")

    text_parts: list[str] = []
    for item in output:
        content = _read_field(item, "content", default=())
        if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
            continue
        for part in content:
            text = _read_field(part, "text", default=None)
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
    combined = "\n".join(text_parts).strip()
    if not combined:
        raise GenerationProviderError("OpenAI generation response did not contain output text.")
    return combined


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


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and (status_code in {408, 409, 429} or status_code >= 500):
        return True

    transient_class_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
    return isinstance(exc, (ConnectionError, TimeoutError)) or (
        exc.__class__.__name__ in transient_class_names
    )


def _sanitised_provider_error(
    exc: Exception,
    *,
    attempts: int,
) -> GenerationProviderError:
    status_code = getattr(exc, "status_code", None)
    status_suffix = f", HTTP {status_code}" if isinstance(status_code, int) else ""
    return GenerationProviderError(
        "OpenAI generation request failed "
        f"after {attempts} attempt(s) ({exc.__class__.__name__}{status_suffix})."
    )
